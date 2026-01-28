"""
Local RAG App - Search your knowledge base with AI

A Streamlit app combining:
- QMD for hybrid search (BM25 + vector + LLM reranking)
- Docker Model Runner for LLM generation

Run with: uv run streamlit run app.py
"""

import streamlit as st
import subprocess
import json
import requests
from typing import Optional

# Configuration
DMR_API_URL = "http://localhost:12434/engines/llama.cpp/v1"
DEFAULT_MODEL = "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL"
EMBEDDING_MODEL = "ai/embeddinggemma"


def get_embedding(text: str) -> list[float] | None:
    """Get embeddings from Docker Model Runner."""
    try:
        response = requests.post(
            f"{DMR_API_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30
        )
        if response.ok:
            return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException:
        pass
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


def get_available_models() -> list[dict]:
    """Fetch available models from Docker Model Runner."""
    try:
        response = requests.get(f"{DMR_API_URL}/models", timeout=5)
        if response.ok:
            return response.json().get("data", [])
    except requests.exceptions.RequestException:
        pass
    return []


def search_with_qmd(query: str, num_results: int = 5, min_score: float = 0.3) -> dict:
    """Use QMD to search for relevant documents."""
    try:
        result = subprocess.run(
            [
                "qmd", "query", query,
                "--json",
                "-n", str(num_results),
                "--min-score", str(min_score)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        st.error(f"QMD search error: {e}")
    return {"results": []}


def generate_response(
    context: str,
    question: str,
    model: str,
    stream: bool = True
) -> Optional[str]:
    """Use Docker Model Runner to generate a response."""

    system_prompt = """You are a helpful assistant answering questions based on the user's personal knowledge base.
Answer based on the provided documents. If the documents don't contain enough information, acknowledge what you can answer and what requires more context.
Be concise but thorough. Reference specific documents when relevant."""

    user_prompt = f"""RELEVANT DOCUMENTS FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

Your response:"""

    try:
        response = requests.post(
            f"{DMR_API_URL}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": stream
            },
            stream=stream,
            timeout=120
        )

        if not response.ok:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None

        if stream:
            return response  # Return response object for streaming
        else:
            return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        st.error(f"Docker Model Runner error: {e}")
        return None


def get_qmd_status() -> dict:
    """Get QMD index status."""
    try:
        result = subprocess.run(
            ["qmd", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return {"available": True, "output": result.stdout}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return {"available": False, "output": "QMD not available"}


def main():
    st.set_page_config(
        page_title="Local RAG",
        page_icon="\U0001F433",
        layout="wide"
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("\u2699\ufe0f Configuration")

        # Model selection
        models = get_available_models()
        if models:
            model_ids = [m["id"] for m in models]
            selected_model = st.selectbox(
                "Model",
                model_ids,
                index=model_ids.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_ids else 0
            )
        else:
            selected_model = DEFAULT_MODEL
            st.warning("\u26a0\ufe0f Docker Model Runner not available")

        st.divider()

        # Search settings
        st.subheader("Search Settings")
        num_results = st.slider("Number of results", 1, 10, 5)
        min_score = st.slider("Minimum relevance score", 0.0, 1.0, 0.3, 0.1)

        st.divider()

        # Status indicators
        st.subheader("Status")

        qmd_status = get_qmd_status()
        if qmd_status["available"]:
            st.success("\u2713 QMD ready")
        else:
            st.error("\u2717 QMD not found")
            st.code("bun install -g https://github.com/tobi/qmd", language="bash")

        if models:
            st.success(f"\u2713 {len(models)} models available")
        else:
            st.error("\u2717 Docker Model Runner not running")

        test_embedding = get_embedding("test")
        if test_embedding:
            st.success(f"\u2713 Embeddings ready ({len(test_embedding)}d)")
        else:
            st.warning("\u26a0 No embedding model")
            st.code("docker model pull ai/embeddinggemma", language="bash")

    # Main content
    st.title("\U0001F433 Local RAG Assistant")
    st.caption("Search your knowledge base and get AI-powered answers \u2014 powered by Docker Model Runner")

    # Question input with search button on same row
    col1, col2 = st.columns([5, 1])

    with col1:
        question = st.text_input(
            "Ask a question about your documents",
            placeholder="e.g., What were the key decisions from the last planning meeting?",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("\U0001F50D Search", type="primary", use_container_width=True)

    st.divider()

    if search_button and question:
        # Search phase
        with st.status("Searching knowledge base...", expanded=True) as status:
            st.write("Running hybrid search (BM25 + vector + LLM reranking)...")
            search_results = search_with_qmd(question, num_results, min_score)

            results = search_results.get("results", [])

            if not results:
                status.update(label="No results found", state="error")
                st.warning("No relevant documents found. Try adjusting your search or indexing more documents.")
                st.code("""# Index your documents with QMD
qmd collection add ~/Documents/notes --name notes
qmd embed""", language="bash")
                return

            status.update(label=f"Found {len(results)} relevant documents", state="complete")

        # Display search results
        with st.expander(f"\U0001F4DA Retrieved Documents ({len(results)})", expanded=False):
            for i, doc in enumerate(results):
                score_pct = int(doc.get("score", 0) * 100)
                score_color = "green" if score_pct > 70 else "orange" if score_pct > 40 else "blue"

                st.markdown(f"""
**{i+1}. {doc.get('title', 'Untitled')}** :{score_color}[{score_pct}%]

`{doc.get('file', 'unknown')}`

{doc.get('snippet', '')[:300]}...
""")
                st.divider()

        # Build context
        context = "\n\n".join([
            f"## {doc.get('title', 'Untitled')}\nSource: {doc.get('file', 'unknown')}\n\n{doc.get('snippet', '')}"
            for doc in results
        ])

        # Generation phase
        st.subheader("\U0001F4AC Response")

        response_container = st.empty()

        with st.spinner("Generating response..."):
            response = generate_response(context, question, selected_model, stream=True)

            if response:
                full_response = ""

                # Stream the response
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            data = line_text[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    response_container.markdown(full_response + "\u258c")
                            except json.JSONDecodeError:
                                continue

                response_container.markdown(full_response)

                # Show sources
                st.caption(f"**Sources:** {', '.join([doc.get('file', '') for doc in results])}")

        # Optional: Show embedding similarities using DMR
        with st.expander("\U0001F9EE Embedding Similarities (via Docker Model Runner)", expanded=False):
            question_embedding = get_embedding(question)
            if question_embedding:
                st.caption("Cosine similarity between your question and each document (using DMR's EmbeddingGemma):")
                for doc in results:
                    doc_embedding = get_embedding(doc.get('snippet', '')[:500])
                    if doc_embedding:
                        similarity = cosine_similarity(question_embedding, doc_embedding)
                        st.progress(similarity, text=f"{doc.get('title', 'Untitled')}: {similarity:.2%}")
            else:
                st.info("Pull the embedding model to see similarities: `docker model pull ai/embeddinggemma`")

    # Footer
    st.divider()
    st.caption("""
    **How it works:** Your question is searched against your local documents using QMD's hybrid search
    (BM25 + vector similarity + LLM reranking). The most relevant documents are then sent to a local LLM
    via Docker Model Runner to generate a contextual answer. All processing happens on your machine.
    """)


if __name__ == "__main__":
    main()
