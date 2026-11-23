# JFK Reader

**Search and question-answering across archival JFK documents using Python, OCR, embeddings, and retrieval-augmented generation (RAG).**

JFK Reader is a reconstructed and modernized version of an earlier archival research project. The original scripts, preserved in [`legacy/`](legacy/), experimented with downloading National Archives JFK records, extracting scanned document text, summarizing files with OpenAI, and asking questions across multiple PDFs using LangChain and FAISS. This revision consolidates those ideas into a command-line application with page-aware retrieval and source references.

> **Project status:** Reconstructed implementation. Automated unit tests are provided; live National Archives downloads, OpenAI responses, and OCR accuracy depend on external services and local setup and have not been end-to-end verified in this release.

## Features

- Collect PDF links from the [National Archives 2025 JFK release listing](https://www.archives.gov/research/jfk/release-2025), with configurable page limits and delays.
- Extract text from PDFs while retaining file and page numbers.
- Optionally OCR pages with little extractable text using Tesseract and PyMuPDF.
- Chunk documents, generate OpenAI embeddings, and store a local normalized vector index using NumPy.
- Retrieve relevant excerpts with cosine similarity and generate answers grounded in those excerpts.
- Display source references (`[S1]`, `[S2]`) with original PDF filenames and page numbers.
- Ask one-off questions or start an interactive Terminal session.

## Quick start

Requires **Python 3.10+** and an OpenAI API key. API usage may incur charges.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your own key. **Do not commit `.env` or share credentials.**

### 1. Add PDFs

Copy your existing JFK PDFs into `data/pdfs/`, or download some linked records:

```bash
jfk-reader download --max-pages 2
```

The downloader scans a limited number of listing pages by default; it does **not** guarantee that it captures the entire archive. The National Archives website structure may change. Respect the source website's access policies and avoid excessive requests.

### 2. Build an index

```bash
jfk-reader index
```

For scanned PDFs with no extractable text, install the optional OCR dependencies and the **Tesseract executable**:

```bash
pip install -e '.[ocr]'
# macOS (Homebrew): brew install tesseract
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
jfk-reader index --ocr
```

`--ocr` applies OCR only to pages with fewer than 50 extracted characters. OCR can misread low-quality historical scans; verify important quotations against the original PDFs. Indexing makes OpenAI embedding API calls and may incur costs. Re-running `index` rebuilds the index and repeats those calls.

### 3. Ask questions

```bash
jfk-reader ask "What do the indexed records say about the investigation?"
```

Or start an interactive Terminal session:

```text
jfk-reader ask
JFK Reader > What do the records say about this subject?
JFK Reader > exit
```

Responses cite retrieved excerpts with `[S1]` etc., followed by a list of source filenames and page numbers. These are **retrieval references**, not a guarantee that the answer is historically accurate. Check original records before drawing conclusions.

## Architecture

```text
National Archives PDF links / local PDFs
               |
         PDF extraction
          /         \
    embedded text  optional Tesseract OCR
          \         /
       page-aware chunks
               |
       OpenAI embeddings
               |
      local NumPy vector index
               |
       similarity retrieval
               |
     OpenAI chat completion
               |
      answer + page sources
```

**Why NumPy rather than FAISS?** The recovered prototype used LangChain + FAISS. This reconstructed version uses a transparent NumPy cosine-similarity index to simplify installation and make the pipeline easier to audit. For very large collections, a persistent FAISS or database-backed vector store would be more scalable. The index stores document excerpts in plaintext JSON; treat it as potentially sensitive and keep it private when appropriate.

## Repository structure

```text
jfk_reader/           CLI, download, extraction, indexing, and Q&A
tests/                Offline unit tests with mocked API calls
.env.example          Credential configuration template
pyproject.toml        Installable package and dependencies
```

`legacy/`               Original recovered scripts (historical reference only)

## Development

```bash
pip install -e '.[dev]'
pytest -q
python -m jfk_reader --help
```

## Original project vs reconstruction

| Earlier recovered scripts | Modernized implementation |
| --- | --- |
| `jfk_scraper_2025.py`: PDF collection and ZIP archive | Bounded downloader with file validation |
| `ocr_trocr.py`: experimental TrOCR processing | Optional Tesseract OCR for image-only pages |
| `gpt_summaries.py`: OpenAI document summaries | Grounded, on-demand Q&A (not batch summaries) |
| `qa_experiment.py`: experimental document QA | Page-aware retrieval and citations |
| `query_rag.py`: LangChain/FAISS RAG prototype | Interactive CLI and NumPy vector retrieval |

The recovered original scripts are **not** part of the supported runtime and may depend on outdated APIs, placeholder credentials, or local file paths. Do not run them without reviewing and updating them.

## Limitations and responsible use

- Retrieval can miss relevant records and language models can hallucinate.
- Historical scans and OCR may contain transcription errors.
- This is a research aid, not a definitive source for historical claims.
- Live functionality requires network access, a valid OpenAI key, and compatible upstream APIs.
- No authentication or multi-user deployment is included; the app is intended for local command-line use.

## License

No license has been selected. All rights reserved unless a license is added by the repository owner.
