# Agentic RAG with LangGraph

This is a hands-on implementation of Agentic RAG — a pattern that makes retrieval-augmented generation smarter by adding a feedback loop on top of the basic retrieve-then-generate flow. The goal of this repo is to share what that actually looks like in code, what problems it solves, and how you can run it yourself.

**GitHub:** https://github.com/Yashraj2007

---

## What problem does this solve?

Standard RAG is simple: take a question, find related documents, feed them to an LLM, get an answer. That works reasonably well, but it breaks in two common ways:

- The retrieved documents aren't actually relevant to the question — and the LLM generates an answer anyway, based on whatever it retrieved.
- The LLM generates an answer that isn't grounded in the retrieved documents — it uses its own training knowledge, or makes something up.

This implementation adds checks for both of these. After retrieval, a grader decides if the docs are relevant. If not, it rewrites the query and tries again. After generation, a hallucination checker verifies the answer is grounded in what was retrieved.

---

## How the pipeline works

```
User question
      │
      ▼
   Agent  ──calls──►  Retrieve (FAISS MMR)
      ▲                      │
      │                      ▼
   Rewrite  ◄── no ──  Grade docs  ── yes ──► Generate ──► Hallucination check ──► END
   (max 2x)                                                       │
                                                          not grounded → END
                                                      (accepts answer if budget exhausted)
```

Each box is a node in the LangGraph `StateGraph`. The graph is compiled with a checkpointer so state is preserved across nodes. The rewrite loop runs at most `max_rewrites` times (default: 2), and the pipeline always terminates — no infinite loops.

---

## What's different from a basic RAG tutorial

Most tutorials wire up a retriever and a prompt template and stop there. Here's what this adds:

### Corrective loop with a hard budget
Query rewriting is capped at exactly 2 attempts. The comparison uses `>=` not `>` — a small but important distinction. Getting this wrong causes an off-by-one that allows one extra rewrite.

### Hallucination grader with structured output
The hallucination check uses Pydantic's structured output to force the LLM to return a typed `grounded: yes | no` field plus reasoning. More reliable than parsing free-text responses.

### Dynamic tool registration
The system prompt is built from whatever retriever tools actually got built — not a hardcoded list. If a vectorstore fails to build, its tool simply doesn't appear. Prevents the model from hallucinating calls to non-existent tools.

### Fallback seed text for offline use
If the documentation URLs are unreachable (bot-check, rate limits, no internet), each knowledge source has hardcoded seed text that gets indexed instead. The pipeline always builds and runs.

### Primary + fallback LLM with exponential backoff
If `llama-3.3-70b-versatile` hits a rate limit, the pipeline automatically switches to a smaller fallback model and backs off with exponential delay (2s, 4s, 8s...). No crashes, no manual retry.

### Disk-cached vectorstores
FAISS indexes are saved to `vectorstore_cache/` after first build. Restarting the notebook doesn't re-scrape and re-embed. Saves several minutes per restart.

### Two graph instances — in-memory and persistent
The same graph is compiled twice: once with `MemorySaver` (fast, resets on restart) and once with `SqliteSaver` (persists conversations to a local `.db` file). Both use `thread_id` so multiple conversations run independently.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get a Groq API key**

Free at https://console.groq.com — no credit card required. The free tier is enough to run the demos.

**3. Create a `.env` file**
```
GROQ_API_KEY=gsk_your_key_here
```

**4. Run the notebook**

Works in Jupyter or Google Colab. First run downloads the embedding model (~90 MB, cached after that), scrapes the docs, and builds the FAISS indexes. Subsequent runs load from the cache.

> **Note:** The notebook currently has a hardcoded API key. Revoke it at console.groq.com before pushing to GitHub, and use the `.env` file instead.

---

## Usage

```python
# ask() — returns the final answer
answer = ask("How do LangChain runnables work?", show_trace=True)

# stream_ask() — prints each node's output as the graph runs
stream_ask("Explain LangGraph state management")

# use the persistent graph (survives restarts)
answer = ask("...", graph=persistent_rag, thread_id="session_1")

# inspect a conversation thread
inspect_rag_thread("session_1")

# add your own documents to a source
add_documents(["your custom text"], source_tag="langchain")
```

---

## Key configuration

All tunable parameters live in the `CONFIG` dict near the top of the notebook:

```python
CONFIG = {
    "model":            "llama-3.3-70b-versatile",  # primary LLM
    "fallback_model":   "meta-llama/llama-4-scout-17b-16e-instruct",
    "embedding_model":  "all-MiniLM-L6-v2",         # local, no API key needed
    "top_k":            3,                           # docs retrieved per query
    "max_rewrites":     2,                           # hard cap on rewrites
    "chunk_size":       800,
    "chunk_overlap":    150,
    "retry_base_delay": 2.0,                         # seconds, doubles each retry
}
```

---

## Knowledge sources

Two sources are pre-configured: LangGraph docs and LangChain docs. Each gets its own FAISS vectorstore and retriever tool. To add your own:

```python
KNOWLEDGE_SOURCES["my_source"] = {
    "urls": ["https://your-docs-url.com/page"],
    "description": "What this source covers — the agent reads this to decide when to use it",
    "tool_name": "search_my_source",
    "fallback_text": "Optional hardcoded text if the URL is unreachable",
}
```

---



---

## Further reading

- [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/) — StateGraph, nodes, edges, checkpointing
- [LangChain runnables](https://python.langchain.com/docs/concepts/runnables/) — how LCEL chains work
- [Groq console](https://console.groq.com)
