# ⚡ CodeMate

Ask questions about any public GitHub repo and get answers grounded in the actual source code.

CodeMate clones a repo, indexes its code using RAG (Retrieval-Augmented Generation), and answers your questions with citations to the exact file and line numbers — instead of guessing.

## How it works

1. **Index** — paste a repo URL → code is chunked and converted into embeddings
2. **Store** — embeddings saved in a ChromaDB vector database
3. **Retrieve** — your question is matched against stored chunks by *meaning*, not keywords
4. **Generate** — the matched chunks + your question are sent to an LLM (Llama 3.3 via Groq), which writes the answer

## Tech stack

`sentence-transformers` (embeddings) · `ChromaDB` (vector store) · `Groq API` (LLM) · `Streamlit` (UI) · `GitPython` (cloning)

## Run it locally

```bash
git clone https://github.com/POORVIKA0608/CodeMate.git
cd CodeMate
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file with your [Groq API key](https://console.groq.com):
```
GROQ_API_KEY=your_key_here
```

Then run:
```bash
streamlit run app.py
```

## Example

> **Q:** How does this library handle timeouts?
> **A:** Accepts a `timeout` param (float or connect/read tuple). Raises `ConnectTimeout` or `ReadTimeout` if exceeded.
> **Sources:** `adapters.py` (141–180), `sessions.py` (596–635)

## Limitations

- Only indexes `.py` files
- No conversation memory between questions
- Larger repos take a few minutes to index