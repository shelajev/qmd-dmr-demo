# Local RAG with QMD and Docker (so you don't have to upload your life to the cloud)

Look, I've got a lot of markdown notes. Years of them. And searching them with `grep` or whatever Spotlight is doing these days wasn't cutting it. I wanted to ask questions about my own stuff without pasting everything into ChatGPT.

So I spent the weekend figuring out how to run this locally. It uses **QMD** for searching and **Docker Model Runner** for the LLM part. It works, it's free (after you buy the hardware I guess), and nothing leaves your laptop.

## The basic idea

It's a RAG pipeline. You've probably seen this diagram a thousand times but here it is again:

```
┌─────────────────────────────────────────────────────────────────┐
│                    The "My Laptop" Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

  "Where did I save that API key?"                     "It's in..."
             │                                              ▲
             ▼                                              │
      ┌──────────────┐                          ┌──────────────┐
      │     QMD      │──(Find docs)────────────▶│ Docker Model │
      │              │                          │    Runner    │
      └──────────────┘                          └──────────────┘
             │                                         ▲
             ▼                                         │
      ┌──────────────┐                          ┌──────────────┐
      │  My messy    │                          │   Local LLM  │
      │  notes       │                          │  (GGUF)      │
      └──────────────┘                          └──────────────┘
```

## The stuff you need

I'm using two main things here:

1.  **QMD** (Quick Markdown Search): It's a tool Tobi Lütke wrote. It does the search part. It's actually pretty clever—uses full-text search *and* vector search, then ranks them. Better than my regex attempts.
2.  **Docker Model Runner**: Docker can run AI models now. I like it because I didn't have to mess around with Python venvs or compiling llama.cpp from source again.

### Prerequisites

You'll need:
*   Docker Desktop (make sure the AI/Model Runner stuff is turned on)
*   [Bun](https://bun.sh) (QMD needs it)
*   A Mac (I haven't tested this on Linux/Windows, sorry. You'll probably need `brew install sqlite` too).

## Setting it up

First, get QMD.
```bash
bun install -g https://github.com/tobi/qmd
```

Then grab a model. I'm using unsloth's GLM-4.7 because it's decent and fits in memory.
```bash
# This might take a bit depending on your internet
docker model pull hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL
```

Now you need to index your files. This is the boring part where you tell QMD where your stuff is.
```bash
# Point it at your folders
qmd collection add ~/Documents/notes --name notes
qmd collection add ~/work/docs --name work-docs

# Give it a hint about what's inside so it knows which pile to look in
qmd context add qmd://notes "Random thoughts and journals"
qmd context add qmd://work-docs "Serious work stuff"

# This builds the index. Go get coffee.
qmd embed
```

## Making it actually usable

I wrote a quick Streamlit script to slap a UI on this. Running commands in the terminal is fine but sometimes I just want a text box.

It sends the search query to `qmd`, formats the results, and dumps them into the local LLM running in Docker.

```python
# app.py
# It's not pretty code, but it works.
import streamlit as st
import subprocess
import json
import requests

# Docker exposes this port
DMR_API = "http://localhost:12434/engines/llama.cpp/v1"
MODEL = "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL"

def get_context(query):
    # Ask QMD for relevant snippets
    try:
        cmd = ["qmd", "query", query, "--json", "-n", "5"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)["results"]
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def stream_answer(context, question):
    # Send to the model
    prompt = f"Docs:\n{context}\n\nQuestion: {question}"
    
    return requests.post(
        f"{DMR_API}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Answer using the context provided. Be brief."},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        },
        stream=True
    )

st.title("Local RAG thing")
q = st.text_input("What are we looking for?")

if st.button("Go") and q:
    docs = get_context(q)
    
    # Mash the docs together
    context_str = "\n".join([f"--- {d['title']} ---\n{d['snippet']}" for d in docs])
    
    # Show what we found (sanity check)
    with st.expander(f"Found {len(docs)} sources"):
        st.code(context_str)

    # Stream the output
    placeholder = st.empty()
    text = ""
    resp = stream_answer(context_str, q)
    
    for line in resp.iter_lines():
        if line and line.startswith(b"data: ") and line != b"data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                content = chunk["choices"][0]["delta"].get("content", "")
                text += content
                placeholder.markdown(text + "▌")
            except:
                pass
    
    placeholder.markdown(text)
```

Run it with:
```bash
uv run streamlit run app.py
# or just `pip install streamlit requests` and run it normally if you don't use uv
```

## Other random stuff

### API
Since Docker Model Runner just exposes an OpenAI-compatible API, you can curl it if you really want to.
```bash
curl -X POST http://localhost:12434/engines/llama.cpp/v1/chat/completions \
  -d '{"model": "hf.co/unsloth/glm-4.7-flash-gguf:Q5_K_XL", "messages": [{"role": "user", "content": "hi"}]}' \
  -H "Content-Type: application/json"
```

### Claude Integration
If you use Claude Desktop, you can add QMD as an MCP server. I found this useful a couple of times.
Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "qmd": { "command": "qmd", "args": ["mcp"] }
  }
}
```

## Why I bothered

Honestly? Privacy and cost. I don't want to pay monthly fees to search my own text files, and I don't want to upload my journal to a server somewhere.

QMD is good because the hybrid search actually finds stuff (unlike pure vector search which sometimes hallucinates relevance). Docker Model Runner is good because it's easy to set up.

Is it as good as GPT-4? No. Is it good enough to find "that one meeting note from last October"? Yeah, usually.

Anyway, the code is up there if you want to try it. Good luck.