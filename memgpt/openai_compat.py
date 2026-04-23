"""Wire-layer translation between MemGPT's pre-v1 OpenAI functions API
vocabulary and the current OpenAI tools API protocol.

MemGPT was written against the 2023 OpenAI functions API
(functions=[...], function_call="auto", role: "function" for results).
That API is deprecated in openai SDK v2.x and, more importantly, is not
supported by the translation paths through OpenAI-compatible proxies
(LiteLLM, vLLM, Ollama) that target non-OpenAI providers. Without
translation, MemGPT's pre-v1 payloads hit Anthropic's Messages API as
structurally-invalid tool_use/tool_result pairings and are rejected.

This module translates at the SDK boundary:
  - Request:  MemGPT vocabulary -> modern tools API (what the wire expects)
  - Response: modern tools API  -> MemGPT vocabulary (what agent.py expects)

Category (b) adapter-layer port. Agent loop, system prompt, message
vocabulary, persistence, and function schemas in memgpt/ are
preserved verbatim; only the bytes on the wire are modernised.

The translator is provider-agnostic: it speaks the shared protocol,
not any specific provider's dialect. Provider-specific quirks live in
a deployment-layer shim above this layer, not here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ---------- request translation: MemGPT vocab -> modern tools API ----------


def _synthesise_tool_call_id(index: int, name: str) -> str:
    """Deterministic ID for pairing assistant.tool_calls[i] with its
    corresponding role:tool response. Determinism matters for idempotent
    replays during debugging; the wire protocol only requires uniqueness
    within the request, not cross-request stability."""
    h = hashlib.sha256(f"{index}:{name}".encode()).hexdigest()[:16]
    return f"call_{h}"


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Walk the message sequence, rewriting MemGPT's pre-v1 shape to
    modern tools shape. Pairing rule: each assistant.function_call at
    index i pairs with the next role:function at index i+1 (MemGPT's
    construction is strictly sequential — verified in system.py and
    agent.py handle_ai_response)."""
    out: list[dict] = []
    pending_tool_call_id: str | None = None

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role == "assistant" and "function_call" in msg and msg["function_call"] is not None:
            fc = msg["function_call"]
            tool_call_id = _synthesise_tool_call_id(i, fc["name"])
            translated = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["arguments"],
                        },
                    }
                ],
            }
            out.append(translated)
            pending_tool_call_id = tool_call_id

        elif role == "function":
            if pending_tool_call_id is None:
                raise ValueError(
                    f"Orphan role:function message at index {i} "
                    f"(name={msg.get('name')!r}): no preceding "
                    f"assistant.function_call to pair with. "
                    f"MemGPT's construction should prevent this; if "
                    f"it happens, the agent loop shape has changed and "
                    f"the translator needs updating."
                )
            out.append({
                "role": "tool",
                "tool_call_id": pending_tool_call_id,
                "content": msg["content"],
            })
            pending_tool_call_id = None

        else:
            # user / system / assistant-without-function_call -> passthrough
            out.append(dict(msg))
            pending_tool_call_id = None

    return out


def _translate_functions_to_tools(functions: list[dict]) -> list[dict]:
    """functions=[{name, description, parameters}] -> tools=[{type, function}]."""
    return [
        {
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            },
        }
        for f in functions
    ]


def _translate_function_call_to_tool_choice(function_call: Any) -> Any:
    """function_call: 'auto' | 'none' | {'name': X} -> tool_choice shape."""
    if function_call in ("auto", "none", None):
        return function_call
    if isinstance(function_call, dict) and "name" in function_call:
        return {"type": "function", "function": {"name": function_call["name"]}}
    return function_call  # unexpected shape; let the SDK reject it loudly


def translate_request(kwargs: dict) -> dict:
    """Translate outgoing chat.completions.create kwargs from MemGPT's
    pre-v1 vocabulary to the modern tools API. Returns a new dict;
    does not mutate input."""
    new_kwargs = dict(kwargs)

    if "messages" in new_kwargs:
        new_kwargs["messages"] = _translate_messages(new_kwargs["messages"])

    if "functions" in new_kwargs:
        new_kwargs["tools"] = _translate_functions_to_tools(new_kwargs.pop("functions"))

    if "function_call" in new_kwargs:
        new_kwargs["tool_choice"] = _translate_function_call_to_tool_choice(
            new_kwargs.pop("function_call")
        )

    return new_kwargs


# ---------- response translation: modern tools API -> MemGPT vocab --------


class _AttrDict(dict):
    """Dict that also supports attribute access. Required because agent.py
    accesses response.choices[0].message.function_call.name (attribute
    path) AND response_message.get('function_call') (dict path) in
    different places — see handle_ai_response (line 512) and step
    (line 937). Both must work on the same object."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return _wrap(v)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self) -> "_AttrDict":
        return _AttrDict(dict(self))


def _wrap(v: Any) -> Any:
    if isinstance(v, dict) and not isinstance(v, _AttrDict):
        return _AttrDict(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def translate_response(response: Any) -> Any:
    """Translate an incoming chat completion response from the modern
    tools API to MemGPT's pre-v1 vocabulary. Given a pydantic ChatCompletion
    (openai SDK v2.x), returns an _AttrDict mirror that exposes
    .choices[0].message.function_call in the shape agent.py expects."""
    # model_dump() handles both pydantic v1 and v2; fall back for plain dicts.
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = dict(response)
    else:
        # Unknown shape — let the caller see the raw object.
        return response

    for choice in data.get("choices", []):
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            # Collapse first tool_call -> function_call. MemGPT's agent
            # loop is single-function-call-per-turn (see agent.py
            # handle_ai_response); if a provider ever returns multiple
            # parallel tool_calls, the additional ones are dropped here.
            # Flagging this as a known simplification consistent with
            # MemGPT's architecture.
            first = tool_calls[0]
            fn = first.get("function", {})
            msg["function_call"] = {
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            }
            # Also map finish_reason: "tool_calls" -> "function_call"
            # so agent.py's finish_reason check at line 125/156 passes.
            if choice.get("finish_reason") == "tool_calls":
                choice["finish_reason"] = "function_call"
        else:
            # No tool call -> ensure function_call key exists as None so
            # response_message.get("function_call") returns falsy cleanly.
            msg.setdefault("function_call", None)

        # Content may be a list of blocks (some providers) or a string.
        # agent.py expects a string; flatten block list to concatenated text.
        content = msg.get("content")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            msg["content"] = "".join(texts) if texts else None

        choice["message"] = msg

    return _AttrDict(data)
