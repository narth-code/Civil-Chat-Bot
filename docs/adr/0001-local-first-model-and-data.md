# Model and data stay on the engineer's machine

Inference runs on a local Ollama model (the exact `qwen3:8b` tag) and data lives in a local Docker PostgreSQL with pgvector. A hosted model provider, a hosted database, and SQLite were rejected. Saved conversations stay on the machine, and vector search and full-text search share one store. The one exception is web search, which sends a query off the machine (see ADR-0005). The browser holds no provider or database credentials, and a model that is missing or fails is reported as a failure, never replaced by a different model.

Consequence: answer quality and speed are bounded by what a local 8B-class model can do on the engineer's hardware, and moving to a hosted model reverses this decision rather than changing a setting.
