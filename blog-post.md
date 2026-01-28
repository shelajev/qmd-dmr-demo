# Building a Local RAG System with QMD and Docker Model Runner

Your personal knowledge base deserves better than being locked in cloud services. What if you could search your notes, meeting transcripts, and documentation with the same intelligence as ChatGPT—but entirely on your own machine?

In this post, I'll show you how to combine two tools—**QMD** and **Docker Model Runner**—to build a private, local RAG (Retrieval-Augmented Generation) system that searches your markdown files and generates intelligent responses without any data leaving your machine.

## What We're Building

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local RAG Pipeline                           │
└─────────────────────────────────────────────────────────────────┘

  Your Question                                        Answer
       │                                                  ▲
       ▼                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     QMD      │────▶│   Context    │────▶│Docker Model  │
│   (Search)   │     │  Assembly    │     │   Runner     │
│              │     │              │     │  (Generate)  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                         │
       ▼                                         ▼
┌──────────────┐                         ┌──────────────┐
│  Your Docs   │                         │  Local LLM   │
│  (Markdown)  │                         │   (GGUF)     │
└──────────────┘                         └──────────────┘
```

**The result:** Ask questions about your documents in natural language and get contextual answers—all running locally.

## The Tools

### QMD: Hybrid Search for Your Knowledge Base

[QMD](https://github.com/tobi/qmd) (Quick Markdown Search) by Tobi Lütke is a local search engine that goes beyond simple keyword matching. It combines three search approaches:

1. **BM25 full-text search** - Fast keyword matching via SQLite FTS5
2. **Vector semantic search** - Understands meaning, not just words
3. **LLM re-ranking** - Uses a local LLM to rank results by true relevance

What makes QMD special is its **position-aware fusion strategy**. It doesn't just combine search results—it intelligently blends them, preserving exact matches while allowing semantic understanding to surface conceptually relevant documents.

### Docker Model Runner: LLMs Without the Infrastructure Headaches

[Docker Model Runner](https://docs.docker.com/ai/model-runner/) brings the Docker experience to AI models. Pull models like you pull images, run them with a single command, and access them via an OpenAI-compatible API.

```bash
# It's as simple as this
docker model pull hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL
docker model run hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL "Hello!"
```

Docker Model Runner handles:
- Model downloading and caching
- GPU acceleration (Apple Silicon, NVIDIA, AMD)
- Memory management and model lifecycle
- An OpenAI-compatible API at `localhost:12434`

## Setup

### Prerequisites

- Docker Desktop with Model Runner enabled
- [Bun](https://bun.sh) runtime (for QMD)
- macOS with Homebrew SQLite (`brew install sqlite`)

### Step 1: Install QMD

```bash
bun install -g https://github.com/tobi/qmd
```

### Step 2: Pull a Model with Docker Model Runner

```bash
# Check Docker Model Runner status
docker model status

# Pull a capable model (20GB, great for RAG)
docker model pull hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL

# Or use a smaller model for testing
docker model pull huggingface.co/unsloth/ministral-3-14b-reasoning-2512-gguf:q4_k_xl
```

### Step 3: Index Your Knowledge Base

```bash
# Add collections for your documents
qmd collection add ~/Documents/notes --name notes
qmd collection add ~/work/docs --name work-docs
qmd collection add ~/meetings --name meetings

# Add context to help search understand your content
qmd context add qmd://notes "Personal notes, ideas, and journal entries"
qmd context add qmd://work-docs "Technical documentation and project specs"
qmd context add qmd://meetings "Meeting transcripts and action items"

# Generate embeddings for semantic search (one-time)
qmd embed
```

## The Magic: A Streamlit RAG App

Here's where it gets interesting. We'll build a web app that:
1. Takes your question
2. Uses QMD to find relevant documents
3. Streams the response from Docker Model Runner

### The App

```python
# app.py - Local RAG with Streamlit
import streamlit as st
import subprocess
import json
import requests

DMR_API_URL = "http://localhost:12434/engines/llama.cpp/v1"
DEFAULT_MODEL = "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL"

def search_with_qmd(query: str, num_results: int = 5) -> dict:
    """Use QMD's hybrid search."""
    result = subprocess.run(
        ["qmd", "query", query, "--json", "-n", str(num_results)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else {"results": []}

def generate_response(context: str, question: str, model: str):
    """Stream response from Docker Model Runner."""
    return requests.post(
        f"{DMR_API_URL}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Answer based on the provided documents."},
                {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}
            ],
            "stream": True
        },
        stream=True
    )

# UI
st.title("🔍 Local RAG Assistant")
question = st.text_input("Ask about your documents")

if st.button("Search & Answer") and question:
    # Search
    results = search_with_qmd(question)["results"]
    context = "\n\n".join([f"## {d['title']}\n{d['snippet']}" for d in results])

    with st.expander(f"📚 Found {len(results)} documents"):
        for doc in results:
            st.markdown(f"**{doc['title']}** ({int(doc['score']*100)}%)")

    # Generate with streaming
    response = generate_response(context, question, DEFAULT_MODEL)
    full_text = ""
    placeholder = st.empty()

    for line in response.iter_lines():
        if line and line.startswith(b"data: ") and line != b"data: [DONE]":
            chunk = json.loads(line[6:])
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            full_text += content
            placeholder.markdown(full_text + "▌")

    placeholder.markdown(full_text)
```

### Running the App

```bash
# Run with uv (recommended)
uv run streamlit run app.py

# Or install and run traditionally
uv pip install streamlit requests
streamlit run app.py
```

Open `http://localhost:8501` and start asking questions about your documents.

![Local RAG App](app-screenshot.png)

## Going Further: API Integration

Docker Model Runner exposes an OpenAI-compatible API, making it easy to integrate with existing tools:

```bash
# The API endpoint
curl -X POST http://localhost:12434/engines/llama.cpp/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

This means you can integrate with any tool that supports OpenAI's API format—just point it at `localhost:12434/engines/llama.cpp/v1`.

### Minimal Python RAG Function

```python
import subprocess, json, requests

def ask(question: str) -> str:
    """Complete RAG pipeline in a few lines."""
    # Retrieve with QMD
    result = subprocess.run(
        ["qmd", "query", question, "--json", "-n", "5"],
        capture_output=True, text=True
    )
    docs = json.loads(result.stdout).get("results", [])
    context = "\n".join([f"# {d['title']}\n{d['snippet']}" for d in docs])

    # Generate with Docker Model Runner
    response = requests.post(
        "http://localhost:12434/engines/llama.cpp/v1/chat/completions",
        json={
            "model": "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL",
            "messages": [{"role": "user", "content": f"{context}\n\nQ: {question}"}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Usage
print(ask("What authentication methods do we support?"))
```

## MCP Server Integration

Both tools play nicely with AI agents. QMD exposes an MCP (Model Context Protocol) server, making it available to Claude Desktop, Claude Code, and other MCP-compatible tools.

**Claude Desktop configuration** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

Now Claude can search your knowledge base directly, using QMD's hybrid search to find relevant documents before responding.

## Why This Combination Works

### What Makes QMD Special

1. **Hybrid search done right** - Not just combining results, but intelligently fusing them with position-aware blending
2. **Query expansion** - Automatically generates search variations to improve recall
3. **Local LLM reranking** - Uses a small, fast model to sort results by true relevance
4. **Zero cloud dependencies** - All models run via node-llama-cpp locally

### What Makes Docker Model Runner Special

1. **Developer experience** - Same mental model as Docker images
2. **No infrastructure** - Just `docker model run`, no servers to manage
3. **Hardware optimization** - Automatically uses your GPU (Metal, CUDA, ROCm)
4. **OpenAI compatibility** - Works with existing tools and libraries

### Together: The Private AI Stack

| Component | Role | Why It Matters |
|-----------|------|----------------|
| QMD | Retrieval | State-of-the-art search without cloud APIs |
| Docker Model Runner | Generation | Run any GGUF model with one command |
| Your filesystem | Storage | Your data stays yours |

## Under the Hood: Shared Embedding Models

Here's something interesting: **QMD and Docker Model Runner use the same embedding model**.

QMD uses three local models via node-llama-cpp:

| Purpose | Model | Size |
|---------|-------|------|
| Embeddings | EmbeddingGemma 300M | ~300MB |
| Re-ranking | Qwen3-Reranker 0.6B | ~640MB |
| Query expansion | Qwen3 1.7B | ~2.2GB |

Docker Model Runner has the same EmbeddingGemma available:

```bash
# Pull the embedding model
docker model pull ai/embeddinggemma

# Use the embeddings API
curl -X POST http://localhost:12434/engines/llama.cpp/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "ai/embeddinggemma", "input": "Hello world"}'
```

This means you can use DMR's embeddings API for other applications while QMD handles its own embeddings internally. Both produce compatible 768-dimensional vectors from the same model architecture.

### Using DMR Embeddings in Python

```python
import requests

def get_embedding(text: str) -> list[float]:
    """Get embeddings from Docker Model Runner."""
    response = requests.post(
        "http://localhost:12434/engines/llama.cpp/v1/embeddings",
        json={"model": "ai/embeddinggemma", "input": text}
    )
    return response.json()["data"][0]["embedding"]

# Use for similarity search, clustering, etc.
embedding = get_embedding("How do I configure authentication?")
print(f"Vector dimension: {len(embedding)}")  # 768
```

## Performance Considerations

### Model Selection

| Model | Size | Use Case |
|-------|------|----------|
| GLM-4.7-Flash | ~20GB | Best quality for RAG responses |
| Ministral-3-14B | ~8GB | Good balance of speed and quality |
| Smaller models | <4GB | Quick responses, simpler queries |

### Search Tuning

QMD offers three search modes for different needs:

```bash
# Fast keyword search (BM25 only)
qmd search "authentication" -n 10

# Semantic search (vector similarity)
qmd vsearch "how to log in" -n 10

# Hybrid with reranking (best quality, slower)
qmd query "user authentication flow" -n 10
```

For RAG, `query` gives the best results, but `search` is faster for simple keyword lookups.

## Wrapping Up

Building a local RAG system used to require cobbling together multiple tools, managing model downloads, and writing lots of glue code. With QMD and Docker Model Runner, the entire stack is:

- **Local** - No data leaves your machine
- **Fast** - Both tools are optimized for local inference
- **Simple** - A few commands to set up, a few lines to query

Your personal knowledge base—your notes, documents, meeting transcripts—can now be searchable and queryable with AI, without giving up privacy or paying per-token API costs.

Give it a try. Index your docs with QMD, pull a model with Docker Model Runner, and start asking questions. Your future self will thank you for having an AI-powered memory that actually respects your privacy.

---

**Resources:**
- [QMD on GitHub](https://github.com/tobi/qmd)
- [Docker Model Runner Documentation](https://docs.docker.com/ai/model-runner/)
- [GGUF Models on HuggingFace](https://huggingface.co/models?library=gguf)
