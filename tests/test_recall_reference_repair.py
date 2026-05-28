"""Recall reference-repair test — F2 viva defence.

Verifies that ``LocalStateManager.load`` re-points ``recall_memory._message_logs``
to the freshly-loaded ``all_messages`` list after unpickle, restoring the
reference-sharing invariant that the in-memory path establishes at init:

    # persistence_manager.py:157 (in-memory path — init)
    self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)

Without the repair, if the two objects were serialised as independent lists (e.g.
because the invariant diverged before save, or as a future-proofing concern), a
post-load ``all_messages.extend`` is invisible to ``recall_memory.text_search``.
The repair, one line inserted after the unpickle assignments, makes the invariant
explicit and load-path-guaranteed rather than relying solely on pickle's reference-
preservation behaviour.

Test structure
--------------
1. ``test_recall_reference_bug_scenario``
   — Demonstrates the bug directly: when ``_message_logs`` and ``all_messages`` are
     independent objects, a post-append is invisible to recall search.

2. ``test_recall_reference_repair_one_line_fix``
   — Demonstrates that the one-line repair (``recall._message_logs = all_messages``)
     eliminates the bug.  This test FAILS if that line is removed.

3. ``test_recall_reference_repair_via_load``
   — End-to-end via ``LocalStateManager.load``.  The pickle is deliberately
     constructed with *independent* ``_message_logs`` and ``all_messages`` (to force
     the bug scenario regardless of pickle's reference-preservation behaviour).
     Without the repair in ``LocalStateManager.load`` this test FAILS; with it,
     it PASSES.  Also asserts the pre-save message is still findable (the repair
     does not corrupt existing messages).
"""

import pickle
import pytest
from unittest.mock import MagicMock, patch

from memgpt.persistence_manager import LocalStateManager
from memgpt.memory import DummyRecallMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role, content, ts="t0"):
    return {"timestamp": ts, "message": {"role": role, "content": content}}


def _write_independent_pickle(path, pre_save_content="XZEBRA_PRESAVE"):
    """Write a pickle where recall_memory._message_logs and all_messages are
    independent list objects — the diverged-reference bug scenario.

    The key: DummyRecallMemory is constructed with ``list(original)`` (a copy),
    and ``all_messages`` is a separate ``list(original)`` copy.  After pickling
    and re-loading, they remain independent because they were never the same
    Python object at serialisation time.
    """
    original = [_msg("user", pre_save_content)]
    recall = DummyRecallMemory(message_database=list(original))  # copy A
    all_messages = list(original)                                 # copy B — independent

    assert recall._message_logs is not all_messages  # confirm independence before pickle

    messages = [_msg("system", "You are MemGPT.")]
    with open(path, "wb") as f:
        pickle.dump(
            {"recall_memory": recall, "messages": messages, "all_messages": all_messages},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ---------------------------------------------------------------------------
# Case 1 — bug scenario: diverged reference causes invisible post-load appends
# ---------------------------------------------------------------------------

def test_recall_reference_bug_scenario():
    """Diverged _message_logs → post-append to all_messages is invisible to recall search."""
    original = [_msg("user", "XZEBRA_PRESAVE")]
    all_messages = list(original)
    recall = DummyRecallMemory(message_database=list(original))  # independent — the bug state

    assert recall._message_logs is not all_messages

    # Simulate append_to_messages extending all_messages in-place
    all_messages.append(_msg("user", "XZEBRA_POSTLOAD", ts="t1"))

    _, count = recall.text_search("XZEBRA_POSTLOAD", count=10, start=0)
    assert count == 0, (
        "expected: with diverged reference, XZEBRA_POSTLOAD is invisible to recall search"
    )


# ---------------------------------------------------------------------------
# Case 2 — one-line repair: re-pointing _message_logs eliminates the bug
# ---------------------------------------------------------------------------

def test_recall_reference_repair_one_line_fix():
    """Re-pointing recall._message_logs = all_messages makes post-appends visible.

    This test FAILS if the repair line below is removed.
    """
    original = [_msg("user", "XZEBRA_PRESAVE")]
    all_messages = list(original)
    recall = DummyRecallMemory(message_database=list(original))  # independent — bug state

    # THE REPAIR — the exact one line inserted in LocalStateManager.load
    recall._message_logs = all_messages

    # Post-repair: append is visible
    all_messages.append(_msg("user", "XZEBRA_POSTLOAD", ts="t1"))

    _, count = recall.text_search("XZEBRA_POSTLOAD", count=10, start=0)
    assert count > 0, "XZEBRA_POSTLOAD should be findable after reference-repair"

    # Pre-existing message is also still findable (repair did not corrupt prior messages)
    _, count_pre = recall.text_search("XZEBRA_PRESAVE", count=10, start=0)
    assert count_pre > 0, "XZEBRA_PRESAVE should still be findable after repair"


# ---------------------------------------------------------------------------
# Case 3 — end-to-end via LocalStateManager.load
# ---------------------------------------------------------------------------

def test_recall_reference_repair_via_load(tmp_path):
    """LocalStateManager.load re-points _message_logs to all_messages.

    The pickle is constructed with *independent* lists to force the bug scenario.
    Without the repair line in LocalStateManager.load this test FAILS.
    """
    save_path = str(tmp_path / "pm.pkl")
    _write_independent_pickle(save_path, pre_save_content="XZEBRA_PRESAVE")

    # Load via the real load path; mock EmbeddingArchivalMemory to avoid heavy deps
    agent_config = MagicMock()
    with patch("memgpt.persistence_manager.EmbeddingArchivalMemory") as mock_ea:
        mock_ea.return_value = MagicMock(save=MagicMock())
        pm = LocalStateManager.load(save_path, agent_config)

    # The repair must have re-pointed the reference
    assert pm.recall_memory._message_logs is pm.all_messages, (
        "repair failed: recall_memory._message_logs is not pm.all_messages after load"
    )

    # Simulate append_to_messages: extend all_messages in-place
    pm.all_messages.append(_msg("user", "XZEBRA_POSTLOAD", ts="t1"))

    # Post-repair: XZEBRA_POSTLOAD is visible via recall search
    _, count = pm.recall_memory.text_search("XZEBRA_POSTLOAD", count=10, start=0)
    assert count > 0, (
        "XZEBRA_POSTLOAD not found — LocalStateManager.load reference-repair may be missing"
    )

    # Pre-save message is still findable (repair did not corrupt existing messages)
    _, count_pre = pm.recall_memory.text_search("XZEBRA_PRESAVE", count=10, start=0)
    assert count_pre > 0, "XZEBRA_PRESAVE should still be findable after load + repair"
