# A failed classification is a failure, and records are never rewritten

When the model is unreachable, times out, or returns output that does not validate, the Routing run is recorded as failed with no Route. It is never turned into `invalid`, retried, or sent to another model, because a guessed Route would be indistinguishable from a real one in the audit. Human evaluations are appended beside the model's decision and never replace it, policy versions are immutable, and the exact transcript sent to the model is stored.

We accepted that an outage shows the engineer a visible failure instead of a graceful-looking answer. If a conversation exceeds the request size cap, the run fails with `context_limit` before any model call; history is never silently truncated or summarized.
