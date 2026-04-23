"""Tests for memgpt.openai_compat — the wire-layer translator between
MemGPT's pre-v1 OpenAI functions vocabulary and the modern tools API.

Regression guard: if these fail, memgpt run will 400 at the first
chat completion when routing through any translation layer that
targets non-OpenAI providers (LiteLLM -> Anthropic, etc.). The
FAILING_REQUEST constant below is the exact shape captured from a
live memgpt run against LiteLLM -> Anthropic during the openai SDK
v0.28 -> v2.x port; it is the concrete failure we are translating
away from.
"""

import pytest

from memgpt.openai_compat import translate_request, translate_response


# ---------------------------------------------------------------------------
# Captured fixtures
# ---------------------------------------------------------------------------

# Minimal reproduction of the request shape MemGPT sends on the first
# turn of memgpt run. Pre-v1 vocabulary: functions=[...],
# function_call="auto", role:"function" for the boot-sequence result.
FAILING_REQUEST = {
    "model": "claude-haiku-dev",
    "messages": [
        {"role": "system", "content": "<baked system prompt>"},
        {
            "role": "assistant",
            "content": "Bootup sequence complete.",
            "function_call": {
                "name": "send_message",
                "arguments": '{\n  "message": "Hi"\n}',
            },
        },
        {
            "role": "function",
            "name": "send_message",
            "content": '{"status": "OK", "message": null}',
        },
        {"role": "user", "content": '{"type": "login"}'},
    ],
    "functions": [
        {
            "name": "send_message",
            "description": "Sends a message",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ],
    "function_call": "auto",
}


class _FakeCompletion:
    """Stand-in for an openai.types.ChatCompletion. We only need
    model_dump() to look right; the translator flattens via dict ops."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _tool_call_response(fn_name, fn_args, finish_reason="tool_calls", content=None):
    return _FakeCompletion({
        "id": "chatcmpl-test",
        "choices": [{
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }],
            },
        }],
    })


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def test_functions_removed_tools_added():
    out = translate_request(FAILING_REQUEST)
    assert "functions" not in out
    assert "tools" in out
    assert len(out["tools"]) == 1
    assert out["tools"][0]["type"] == "function"
    assert out["tools"][0]["function"]["name"] == "send_message"


def test_function_call_auto_becomes_tool_choice_auto():
    out = translate_request(FAILING_REQUEST)
    assert "function_call" not in out
    assert out["tool_choice"] == "auto"


def test_function_call_named_becomes_tool_choice_function():
    req = dict(FAILING_REQUEST, function_call={"name": "send_message"})
    out = translate_request(req)
    assert out["tool_choice"] == {"type": "function", "function": {"name": "send_message"}}


def test_tool_call_id_pairs_assistant_to_tool():
    """Headline invariant. This is the failure that drove the whole
    translator: LiteLLM generated different IDs for the assistant
    tool_use block and the tool_result block, Anthropic rejected."""
    out = translate_request(FAILING_REQUEST)
    msgs = out["messages"]
    assert msgs[1]["role"] == "assistant"
    assert "function_call" not in msgs[1]
    assistant_id = msgs[1]["tool_calls"][0]["id"]
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == assistant_id, (
        "assistant.tool_calls[0].id must equal the following "
        "tool.tool_call_id or Anthropic will 400"
    )


def test_user_and_system_messages_pass_through():
    out = translate_request(FAILING_REQUEST)
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][-1]["role"] == "user"
    assert out["messages"][-1]["content"] == '{"type": "login"}'


def test_orphan_role_function_raises():
    """role:function without a preceding assistant.function_call is a
    construction error MemGPT should never produce; if it does, the
    translator should fail loud rather than send a malformed request."""
    req = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "function", "name": "x", "content": "orphaned"},
    ]}
    with pytest.raises(ValueError, match="Orphan role:function"):
        translate_request(req)


def test_translate_request_does_not_mutate_input():
    req = dict(FAILING_REQUEST)
    original_messages_ref = req["messages"]
    translate_request(req)
    assert "functions" in req, "input must not lose 'functions' key"
    assert "function_call" in req, "input must not lose 'function_call' key"
    assert req["messages"] is original_messages_ref, "input messages list must be untouched"
    # And the original assistant message must still have function_call, not tool_calls
    assert "function_call" in req["messages"][1]


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------

def test_response_attribute_access():
    """agent.py line 944: response_message.function_call.name"""
    resp = translate_response(_tool_call_response(
        "core_memory_append",
        '{"name":"human","content":"favourite colour: teal"}',
    ))
    assert resp.choices[0].message.function_call.name == "core_memory_append"


def test_response_dict_style_access():
    """agent.py line 512: response_message.get('function_call')"""
    resp = translate_response(_tool_call_response("send_message", '{"message":"hi"}'))
    msg = resp.choices[0].message
    fc = msg.get("function_call")
    assert fc is not None
    assert fc["name"] == "send_message"


def test_response_finish_reason_remapped():
    """agent.py line 125/156 requires finish_reason in {'stop',
    'function_call'}; modern API returns 'tool_calls'."""
    resp = translate_response(_tool_call_response("send_message", "{}"))
    assert resp.choices[0].finish_reason == "function_call"


def test_response_content_block_list_flattened():
    """Some providers return assistant.content as a list of text
    blocks rather than a string; agent.py expects a string."""
    resp = translate_response(_FakeCompletion({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            },
        }],
    }))
    assert resp.choices[0].message.content == "Hello world"


def test_response_no_tool_call_sets_function_call_none():
    """msg.get('function_call') must return falsy cleanly for non-tool
    responses — agent.py branches on its truthiness."""
    resp = translate_response(_FakeCompletion({
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "just chatting"},
        }],
    }))
    assert resp.choices[0].message.get("function_call") is None


def test_response_passthrough_for_unknown_type():
    """Unexpected response shape (no model_dump, not a dict) returns
    raw so the caller can surface the mismatch with full context."""

    class Opaque:
        pass

    sentinel = Opaque()
    assert translate_response(sentinel) is sentinel
