# Legal AI
A chatbot to help you navigate through the complicated paths of AI regulations inside the EU.

Technically it is a RAG system implementation, using:
- **LLM** - Llama 3 8B via Groq (free tier)
- **VectorDB** - ChromaDB
- **Embedding** - HuggingFace `all-MiniLM-L6-v2` (free, runs locally)
- **Orchestration** - LangChain

For this demo, the context is the **Artificial Intelligence Act**, adopted by the EU Parliament on 13 March 2024.

---

## Setup & Run

### 1. Prerequisites
- Python 3.10 or 3.11 recommended (Python 3.13 may have compatibility issues with some packages)
- A free Groq API key from https://console.groq.com

### 2. Clone / Download the project
Place all files in a folder, e.g. `legalai/`

### 3. Create a virtual environment (strongly recommended)
```bash
cd legalai
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

> If you get errors about `torch` or `sentence-transformers`, run:
> ```bash
> pip install sentence-transformers
> ```

### 5. Set your Groq API key

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_key_here"
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_key_here
```

**Mac/Linux:**
```bash
export GROQ_API_KEY=your_key_here
```

Or create a `.env` file in the project folder:
```
GROQ_API_KEY=your_key_here
```

### 6. Run the app
```bash
python -m streamlit run app.py
```

The first run will automatically download the EU AI Act PDF and build the ChromaDB vector store (this takes 1-2 minutes). Subsequent runs will be instant.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'langchain.chains'`**
Run `pip install -r requirements.txt` again with the updated requirements file — versions are now pinned to compatible releases.

**`ModuleNotFoundError: No module named 'langchain_core.retrievers'`**
Make sure `langchain-core>=0.3.0` is installed: `pip install langchain-core==0.3.58`

**Slow first startup**
Normal — it's downloading the HuggingFace embedding model (~80MB) and the EU AI Act PDF, then building the vector index.

---

## Demo
https://huggingface.co/spaces/firica/legalai