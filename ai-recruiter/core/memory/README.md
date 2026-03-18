# Memory Module

Shared memory layer for agents - Redis + Vector DB.

## Purpose

Provide persistent and semantic memory for interview sessions:
- **Session Memory**: Redis hash/list for conversation state
- **Vector Memory**: Embeddings for semantic search over history
- **Context Window**: Manage context for LLM calls

## Key Files

| File | Purpose |
|------|---------|
| `redis_client.py` | Redis connection pool and operations |
| `vector_store.py` | Vector DB client (Qdrant/Pinecone/Chroma) |
| `session_memory.py` | Interview session state storage |
| `conversation_memory.py` | Conversation history management |
| `embeddings.py` | Embedding model wrapper |

## Memory Types

1. **Short-term**: Current interview context (Redis)
2. **Long-term**: Historical interview patterns (Vector DB)
3. **Semantic**: Retrieved similar Q&A pairs for context
