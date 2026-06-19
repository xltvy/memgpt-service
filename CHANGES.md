# Modifications from upstream MemGPT

This file records the modifications made in this fork relative to upstream MemGPT, as required by
the Apache License, Version 2.0, §4(b). The fork's baseline is upstream commit `f46cc3b`
(`pymemgpt` v0.1.18-alpha.1). Each modification is summarised below; the fork's git history is the
authoritative, line-level record, and each commit names its change class.

The governing principle of the fork is **modernise plumbing, preserve architecture**: the MemGPT
memory architecture (core / archival / recall tiers, the function-calling schemas, the recursive
summariser, the message wire-format, and the persistence model) is preserved. The changes below are
limited to dependency and compatibility plumbing required to run that architecture on the current
ecosystem.

## Compatibility patches

- **Wire-format modernisation / Anthropic-Claude tool-use translation** — added
  `memgpt/openai_compat.py`, a wire-layer translator between MemGPT's pre-v1 OpenAI *functions* API
  vocabulary and the modern OpenAI *tools* API protocol. Requests are rewritten from MemGPT's
  vocabulary to the modern tools shape on the way out, and responses are rewritten back to the
  vocabulary `agent.py` expects on the way in. This lets MemGPT's payloads reach Anthropic/Claude
  (and other providers behind OpenAI-compatible proxies such as LiteLLM) as structurally-valid
  tool-use exchanges. The translator is provider-agnostic; provider-specific quirks are handled by a
  deployment-layer shim above this module, not here.
- **Modern OpenAI SDK port** — `memgpt/openai_tools.py` ported from the 2023 `openai` SDK v0.28 API
  to the v2.x SDK (client construction, call signatures, response shapes).

## Dependency modernisation

- **Pin loosening** — dependency version ranges in `pyproject.toml` updated for the current
  ecosystem (`openai`, `tiktoken`, `setuptools`, and others), and the supported Python range
  narrowed to 3.11.
- **llama_index migration** — migrated from `llama_index` 0.8 to 0.14 across the modules that touch
  it: `memgpt/utils.py`, `memgpt/memory.py`, `memgpt/embeddings.py`, `memgpt/cli/cli.py`,
  `memgpt/cli/cli_load.py`, and `memgpt/connectors/local.py`.

## Adapter-layer additions

These are behaviour-preserving ports and reuse-oriented refactors — extraction and de-duplication
that leave the memory architecture's behaviour identical to upstream:

- **Cutoff-selection factoring** (`memgpt/agent.py`) — the summarisation message-cutoff selection
  logic was extracted from `Agent.summarize_messages_inplace` (and its `AgentAsync` twin) into a
  standalone, pure `select_cutoff(...)` function, de-duplicating the two implementations. The
  algorithm is unchanged; an equivalence test pins the factored function to upstream behaviour.
- **Recall reference-repair at load** (`memgpt/persistence_manager.py`) — `LocalStateManager.load`
  re-points `recall_memory._message_logs` to the freshly-loaded `all_messages` list after unpickle,
  hardening the in-memory-path reference-sharing invariant on the reload path. Happy-path behaviour
  is unchanged.
- **Compatibility module** (`memgpt/openai_compat.py`) — see "Compatibility patches" above; this is
  the one substantive new source module.

For the precise, line-level diff against upstream, see the fork's commit history
(`git diff f46cc3b -- <file>`) and the per-commit change-class annotations.
