# openclaw-memgpt-sidecar

A maintained fork of [MemGPT](https://github.com/letta-ai/letta) that packages the
MemGPT memory architecture as a Python sidecar for the
[openclaw-memgpt](https://github.com/xltvy/openclaw-memgpt) plugin.

It preserves MemGPT's memory design — the core / archival / recall memory tiers, the
function-calling syscall vocabulary, and the recursive summariser — while modernising the
surrounding plumbing so the architecture runs on the current Python LLM ecosystem and against
Anthropic/Claude models through OpenAI-compatible proxies.

## Relationship to upstream MemGPT

This project is a fork of MemGPT (originally `cpacker/MemGPT`, now maintained as
[`letta-ai/letta`](https://github.com/letta-ai/letta)), described in
[Packer et al., 2024](https://arxiv.org/abs/2310.08560). Originally authored by Packer et al.
(2024); maintained as a fork by Altay Acar. All credit for the memory architecture belongs to the
original authors:

- Charles Packer
- Vivian Fang
- Sarah Wooders
- Shishir Patil
- Kevin Lin

The fork exists because the original `pymemgpt` release was written against the 2023 single-vendor
OpenAI ecosystem (the pre-v1 functions API, `openai` SDK v0.28, `llama_index` 0.8). Running that
architecture unchanged against today's SDKs and against non-OpenAI providers (e.g. Claude via
LiteLLM) is not possible without compatibility work. This fork supplies exactly that work and
nothing more: **the memory architecture is preserved; only the plumbing is modernised.**

## Scope of modifications

The changes are deliberately narrow and limited to compatibility and dependency plumbing:

- **Wire-format modernisation (Anthropic/Claude compatibility)** — translation at the SDK boundary
  from MemGPT's pre-v1 OpenAI functions vocabulary to the modern OpenAI tools API, so payloads
  reach Claude (and other providers behind OpenAI-compatible proxies) as valid tool-use exchanges.
- **Dependency modernisation** — `openai` SDK v0.28 → v2.x port, `llama_index` 0.8 → 0.14
  migration, and accompanying dependency-pin loosening.
- **Adapter-layer ports** — small, behaviour-preserving ports and reuse-oriented refactors that let
  the existing code run on the modern dependencies without altering the memory architecture.

A full, file-level account of what changed relative to upstream is in
[CHANGES.md](CHANGES.md), per the Apache 2.0 modification-notice requirement.

## Use

This package is primarily intended as the sidecar dependency of the
[openclaw-memgpt](https://github.com/xltvy/openclaw-memgpt) plugin. Independent use is possible,
but API stability is scoped to the plugin's needs and is not guaranteed for other consumers.

## Installation

```sh
pip install openclaw-memgpt-sidecar==1.0.0
```

Python 3.11 is required.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text. The
license is preserved unchanged from upstream MemGPT.
