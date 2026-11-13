# Getting Started

A step-by-step guide to running the simple-rag project.

## What this project does

Simple RAG answers questions from PDF documents through four stages:

1. **Ingest** - `PDFReader` extracts text, `text2chunk` splits it into overlapping chunks.
2. **Retrieve** - `BM25Retrieval` finds the most relevant chunks for a query.
3. **Rerank** - `CrossEncoderRerank` re-scores the candidates with a cross-encoder model.
4. **Generate** - an LLM (`OllamaLLM`, `OpenRouterLLM`, or `GeminiLLM`) writes the answer from the retrieved context.

## Requirements

- Python 3.10 or newer (3.11 recommended; the pinned dependencies do not support 3.14)
- Poppler (required to build the `pdftotext` package)
- One LLM backend:
  - Ollama with `llama3:instruct` (local, no API key), or
  - An OpenRouter API key (cloud), or
  - A Google API key (the Gemini provider needs a fix before use)

## First-time setup

### 1. Install system dependencies (macOS)

```bash
brew install pkg-config poppler
```

On Debian/Ubuntu:

```bash
sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install makes the `rag` package importable from anywhere in the project.

### 4. Set up an LLM backend

**Option A - Ollama (local):**

```bash
brew install ollama
brew services start ollama
ollama pull llama3:instruct
```

Check that it works:

```bash
ollama list
curl http://localhost:11434/api/tags
```

**Option B - OpenRouter (cloud):**

Create a `.env` file in the project root (it is gitignored):

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

See `.env.example` for the available variables.

## Running the project

Always activate the environment first:

```bash
source .venv/bin/activate
```

### CLI with Ollama

```bash
python examples/simple_rag_bm25_ollama.py
```

The script loads `examples/sample.pdf`, runs a sample query, then waits for your questions. Press `Ctrl-D` or `Ctrl-C` to exit.

### CLI with OpenRouter

```bash
python examples/simple_rag_bm25_openrouter.py
```

### Web UI with Streamlit

The UI example currently uses Ollama:

```bash
pip install streamlit
streamlit run examples/simple_rag_bm25_ollama_ui.py
```

Upload a PDF, click **Ingest PDF**, then ask questions.

### Use your own PDF

Change the `sample_pdf` path in the example script:

```python
sample_pdf = "/absolute/path/to/your/document.pdf"
```

Or upload it through the Streamlit UI.

## Quick non-interactive test

Useful for checking the setup without asking questions manually:

```bash
printf 'What can Ollama do?\n' | python examples/simple_rag_bm25_ollama.py
```

A healthy run prints `Number of chunks: ...`, a sample answer, and then exits.

## Tuning

| Setting | Location | Effect |
| --- | --- | --- |
| `chunk_size` | example scripts, `text2chunk(...)` | Words per chunk. Larger chunks keep more context but make retrieval less precise. |
| `overlap` | example scripts, `text2chunk(...)` | Words shared between adjacent chunks. Prevents splitting answers across boundaries. |
| `retrieval_top_k` | `SimpleRAGPipeline(...)` | Number of BM25 candidates passed to the reranker. Default is 100. |
| `rerank_top_k` | `SimpleRAGPipeline(...)` | Number of chunks sent to the LLM. Default is 3. |
| `model_name` | reranker | Cross-encoder model. Default is `cross-encoder/ms-marco-MiniLM-L-6-v2`. |

## Project structure

```text
simple-rag/
├── examples/
│   ├── sample.pdf
│   ├── simple_rag_bm25_ollama.py       # CLI, local Ollama
│   ├── simple_rag_bm25_ollama_ui.py    # Streamlit UI, local Ollama
│   └── simple_rag_bm25_openrouter.py   # CLI, OpenRouter
├── rag/
│   ├── data_helper.py                  # PDF reading
│   ├── llm.py                          # Ollama, OpenRouter, Gemini providers
│   ├── pipeline.py                     # retrieve -> rerank -> generate
│   ├── prompt.py                       # prompt template
│   ├── rerank.py                       # cross-encoder reranker
│   ├── retrieval.py                    # BM25 retrieval
│   └── text_utils.py                   # chunking
├── test/                               # empty, tests welcome
├── .env                                # secrets, gitignored
├── .env.example
├── requirements.txt
└── setup.py
```

## Troubleshooting

**`ERROR: Could not build wheels for pdftotext`**
Poppler or pkg-config is missing. Run `brew install pkg-config poppler` and retry.

**Dependencies fail to install on Python 3.14**
Several pinned packages (torch, sentence-transformers) do not support 3.14. Use Python 3.11:

```bash
pyenv install 3.11.9   # only if not already installed
~/.pyenv/versions/3.11.9/bin/python3 -m venv .venv
```

**Ollama request times out or returns an empty answer**
Make sure the server is running and the model exists:

```bash
brew services start ollama
ollama list
```

The example expects the model name `llama3:instruct` exactly.

**OpenRouter returns `0 endpoints out of N requested are available`**
Your account privacy settings (ZDR) block the requested model's endpoints. Either change the privacy settings at https://openrouter.ai/settings/privacy or choose another `OPENROUTER_MODEL`.

**`ValueError: Please set GOOGLE_API_KEY` or a Gemini model error**
`GeminiLLM` currently references the invalid model name `text-davinci-003`. Use Ollama or OpenRouter until it is fixed.

**`EOFError` when piping input into the examples**
Already handled: the scripts exit cleanly when stdin closes.

## Security notes

- Never commit `.env` or API keys. It is already listed in `.gitignore`.
- Rotate any key that is exposed in logs, screenshots, or chat.
- `.venv/` is also gitignored; recreate it with the steps above when needed.
