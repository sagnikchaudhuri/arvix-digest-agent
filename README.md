# Autonomous arXiv Paper Digest & QA Agent
## Evaluator Quickstart

Python 3.11
Ollama + llama3.2:3b

Run:
python -m arxiv_digest

Verified:
108 tests passed
Example paper: 2601.16325v1
QA demonstrates grounded and out-of-paper refusal behavior.

## Overview

This Python 3.11 command-line agent accepts a research topic, arXiv ID, or arXiv URL. It searches the official arXiv API for topic queries, or resolves a specific paper directly, then fetches and parses its PDF.

The paper is divided into deterministic page-aware chunks, embedded locally with Sentence Transformers, and stored in a persistent Chroma collection. A LangGraph workflow carries the paper, parsed text, chunk/index information, briefing, and QA conversation through an explicit shared state.

Briefings use deterministic extractive selection and application-generated evidence. One optional local Ollama call may rewrite a short selected set of passages into a plain-English summary; failure falls back to an extractive summary. Follow-up QA retrieves from the selected paper only and returns source chunk/page citations. No hosted API key is required.

## Architecture

```text
Research topic / arXiv ID / URL
                |
       understand_query
                |
       retrieve_papers
          /          \
   direct ID/URL     topic search
        |             select_paper
         \             /
          fetch_and_parse
                |
            index_paper
                |
          summarize_paper
                |
          briefing ready
                |
        same session state
                |
          answer_question <---- next question
                |
  question embedding -> paper-filtered Chroma retrieval
                |
        grounded answer + validated provenance
```

The graph consists of `understand_query`, `retrieve_papers`, `select_paper`, `fetch_and_parse`, `index_paper`, `summarize_paper`, and `answer_question`. Conditional routes handle invalid input, no candidates, retrieval/PDF/indexing/summarization failures, and direct-ID lookup versus topic selection. Each graph node delegates to a domain service; network, parsing, indexing, summarization, and QA implementations remain outside orchestration.

## State Shape

`AgentState` carries the original query and classification, candidate and selected `PaperMetadata`, `ParsedPaper`, chunks and indexed chunk IDs, vector-store reference, `PaperBriefing`, conversation history, current question/latest QA result, status, warnings, and structured stage errors.

`ArxivDigestAgent` uses a LangGraph checkpoint thread for the lifetime of the agent instance. `ask()` resumes that thread, retains the selected paper and indexed chunks, and does not repeat ingestion. `reset()` removes the active thread. The default `InMemorySaver` is process-local: Chroma data persists on disk, but the interactive conversation checkpoint does not survive restarting the Python process.

## Setup

From the project root in PowerShell, install Python 3.11, then run:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

The example environment selects local Ollama and contains no credentials. The default local paths are `data/papers/` and `data/vector_store/`; the optional `PDF_DIR`, `VECTOR_STORE_DIR`, and `EMBEDDING_MODEL` variables in `.env` are honored by the default components. `.env`, downloaded PDFs, model caches, and vector-store data are ignored by Git.

## Ollama Setup

Install Ollama using its official installer, start its local service, then download the configured model once:

```powershell
ollama pull llama3.2:3b
ollama list
```

The defaults in `.env` are `LLM_PROVIDER=ollama`, `LLM_BASE_URL=http://127.0.0.1:11434`, and `LLM_MODEL=llama3.2:3b`. No API key is needed. The model runs on the Ollama host; CPU-only execution can be selected for a standalone server process (use the default Ollama model directory unless you have configured another location):

```powershell
$env:OLLAMA_MODELS = "$env:USERPROFILE\.ollama\models"
$env:OLLAMA_LLM_LIBRARY = "cpu_avx2"
$env:OLLAMA_HOST = "127.0.0.1:11434"
ollama serve
```

These variables must be set in the Ollama server process, not in the Python client. If the Ollama desktop service is already running, use that service or configure its environment before starting it; do not launch a second server on the same port.

Optional hosted providers are supported but are not required: set `LLM_PROVIDER=gemini`, `mistral`, or `openrouter` and configure that provider's key locally in `.env`. Never commit credentials.

## Run

Start the interactive CLI:

```powershell
.\.venv\Scripts\python.exe -m arxiv_digest
```

The installed console script is also available as `digest`. Enter a topic, arXiv ID, or arXiv URL. After the briefing, ask follow-up questions; use `reset` for another paper or `exit`/`quit` to leave.

## Example Run

Verified against arXiv paper `2601.16325v1` in one CLI session:

```text
Research topic, arXiv ID, or URL (exit to quit): 2601.16325v1
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2591.27it/s]
Preparing grounded paper briefing...
Paper briefing generated.

========================================================================
Does Gravity Care About Electric Charge? A Minimalist Model and Experimental Test
Authors: Renato Vieira dos Santos
arXiv: 2601.16325v1 | Published: 2026-01-22
https://arxiv.org/abs/2601.16325v1
========================================================================

Plain-English summary
Precision tests of the weak equivalence principle achieve remarkable sensitivity but deliberately minimize electric charge on test masses, leaving this fundamental question experimentally open. This reveals a significant gap in our experimental knowledge of gravity, a gap we propose to fill with a modified tor- sion balance experiment that treatsq/m as a controlled variable rather than a background to be eliminated.

Problem
Precision tests of the weak equivalence principle achieve remarkable sensitivity but deliberately minimize electric charge on test masses, leaving this fundamental question experimentally open.

Method / approach
This reveals a significant gap in our experimental knowledge of gravity, a gap we propose to fill with a modified tor- sion balance experiment that treatsq/m as a controlled variable rather than a background to be eliminated.

Key results
Precision tests of the weak equivalence principle achieve remarkable sensitivity but deliberately minimize electric charge on test masses, leaving this fundamental question experimentally open.

Limitations
We have de- liberately chosen a linear formulation in the complex current to maintain transparency and direct testability, leaving more complete theoretical extensions (variational actions, embedding in non-linear general relativity) for future work, should experimental evidence warrant it.

Suggested follow-up questions
  - How does the paper evaluate its proposed approach?
  - What limitations do the authors identify?

Question (reset for another paper, exit to quit): What problem does this paper address?

Answer: Crucially, for neutral matter (q = 0), the violation vanishes, explaining why existing high-precision WEP tests, which deliberately use neutral or nearly neutral test masses, are blind to this effect. The prediction occupies virgin experimental territory precisely because it targets a regime (q/m varied and non-zero) that has been systematically avoided to suppress electrostatic backgrounds. Phenomenological stance and limitations This model is presented as aminimal phenomenological parameterizationthat captures a possible charge-induced violation of the weak equivalence principle. We have de- liberately chosen a linear formulation in the complex current to maintain transparency and direct testability, leaving more complete theoretical extensions (variational actions, embedding in non-linear general relativity) for future work, should experimental evidence warrant it. Our primary aim is to highlight an experimental regime that, to the best of our knowledge,has not been system- atically investigatedin high-precision WEP tests—since such experiments deliberately suppress electrostatic back- grounds by minimizing charge-to-mass ratios—and to provide a clean, one-parameter target for a
Sources
  2601.16325v1:0022:7e0db673d3cc (page 4): “Crucially, for neutral matter (q = 0), the violation vanishes, explaining why existing high-precision WEP tests, which deliberately use neutral or nearly neutral test masses, are blind to this effect. The prediction occupies virgin experimental territory precisely because it targets a regime (q/m varied and non-zero) that has been systematically avoided to suppress electrostatic backgrounds. Phenomenological stance and limitations This model is presented as aminimal phenomenological parameterizationthat captures a possible charge-induced violation of the weak equivalence principle. We have de- liberately chosen a linear formulation in the complex current to maintain transparency and direct testability, leaving more complete theoretical extensions (variational actions, embedding in non-linear general relativity) for future work, should experimental evidence warrant it. Our primary aim is to highlight an experimental regime that, to the best of our knowledge,has not been system- atically investigatedin high-precision WEP tests—since such experiments deliberately suppress electrostatic back- grounds by minimizing charge-to-mass ratios—and to provide a clean, one-parameter target for a”

Question (reset for another paper, exit to quit): What limitations do the authors identify?

Answer: dictions while introducing minimal additional structure. Second, it identifies and proposes to test an unexplored regimeofexperimentalgravity. Beyonditsspecificpredic- tions, the framework illustrates how theoretical minimal- ism can reveal overlooked opportunities in fundamental physics. Scope and limitations.We work exclusively with clas- sical Maxwell electrodynamics and linearized gravity in the weak-field approximation (gravitoelectromagnetism [3]), in flat spacetime. This is not a full unification with general relativity, nor does it address quantum as- pects; rather, it is an effective description that isolates the core conceptual issues of EM-gravity coupling while preserving mathematical transparency and testability. The restriction to linearized gravity is not only math- ematically convenient but also physically justified: the experiments we propose (torsion balances, free-fall tests, atom interferometry) operate in the weak-field regime 1 arXiv:2601.16325v1 [gr-qc] 22 Jan 2026
Sources
  2601.16325v1:0004:7cf2d89037f1 (page 1): “dictions while introducing minimal additional structure. Second, it identifies and proposes to test an unexplored regimeofexperimentalgravity. Beyonditsspecificpredic- tions, the framework illustrates how theoretical minimal- ism can reveal overlooked opportunities in fundamental physics. Scope and limitations.We work exclusively with clas- sical Maxwell electrodynamics and linearized gravity in the weak-field approximation (gravitoelectromagnetism [3]), in flat spacetime. This is not a full unification with general relativity, nor does it address quantum as- pects; rather, it is an effective description that isolates the core conceptual issues of EM-gravity coupling while preserving mathematical transparency and testability. The restriction to linearized gravity is not only math- ematically convenient but also physically justified: the experiments we propose (torsion balances, free-fall tests, atom interferometry) operate in the weak-field regime 1 arXiv:2601.16325v1 [gr-qc] 22 Jan 2026”

Question (reset for another paper, exit to quit): exit
Goodbye.
```

The two paper questions returned citations from their retrieved chunks and also mentioned their source of information. The selected paper and QA history remained in the same graph session between questions.

## Retrieval and Parsing

`ArxivRetriever` uses the official arXiv API/library for both ID lookups and topic searches. Topic candidates are selected deterministically by title/abstract token overlap, metadata completeness, then API order. `PDFFetcher` accepts official arXiv PDF URLs, validates downloads, writes them atomically, and reuses cached PDFs under `PDF_DIR` (default `data/papers/`). `PDFTextParser` extracts page text, best-effort abstract, headings, references, and warnings with `pypdf`. Empty/corrupt downloads and insufficient text produce structured failures. Scanned/image-only PDFs may not have extractable text; OCR is not implemented.

## Chunking, Embeddings, and Vector Store

`TextChunker` deterministically packs paragraphs near a configurable character target, favors paragraph/section boundaries, adds word-safe overlap, and retains paper ID, page numbers, section, and stable chunk IDs. `SentenceTransformerEmbedder` uses `sentence-transformers/all-MiniLM-L6-v2` by default and loads lazily; indexing and question queries share the same embedder instance/model. This local model avoids paid embedding APIs and keeps embedding inputs local. Its weights are obtained from the model hub on first use and cached by the library.

`ChromaVectorStore` implements the backend-neutral vector-store interface and persists at `VECTOR_STORE_DIR` (default `data/vector_store/`). The collection stores embeddings, original text, and scalar metadata. Search supports `top_k` and a paper-ID filter. Re-indexing replaces stale chunks for that paper without deleting other papers. Chroma uses cosine distance, where lower values are closer; QA applies the configurable `QA_MAX_COSINE_DISTANCE` relevance gate before generation.

## Grounding Strategy

`PaperSummarizer` ranks extracted sentences using title/abstract terminology, section cues, and position, then selects a compact set of passages across the paper. Problem, method, results, and limitations are source sentences; if the text does not support a field, it remains `Not found in the available paper text.` The plain-English summary uses an extractive fallback and, when a provider is configured, at most one optional LLM call on the selected passages (about 3,500 characters, up to 300 output tokens, 90-second timeout). Provider errors or malformed output fall back without failing the briefing.

For both briefing and QA, the model may identify chunk IDs but cannot author provenance. The application verifies each ID against its source chunks and constructs the quote and page numbers from parsed/chunk metadata. QA embeds the current question, searches the active paper's Chroma records, and uses relevant earlier turns only to clarify referential follow-ups. Conversation history is context, never evidence. Invalid evidence is discarded; empty or irrelevant retrieval returns `The answer is not found in the available paper text.` The model is instructed not to use outside knowledge.

## Failure Handling

- Invalid input, arXiv API failures, and zero search results stop with a structured status and stage message.
- Topic searches with several candidates use the deterministic ranking rule above; direct IDs/URLs bypass ranking.
- PDF download/HTTP/corruption and insufficient extractable text are reported separately; OCR is not claimed.
- Empty chunk output and embedding/vector-store errors stop before briefing generation.
- Unsupported QA, empty retrieval, malformed model output, and invalid evidence fail closed to the explicit not-found answer or the briefing's deterministic summary fallback.

## Design Decisions & Tradeoffs

LangGraph makes the stages, conditional failure routes, shared state, and QA continuation explicit instead of hiding the workflow in one prompt. Its default in-memory checkpoint is sufficient for one interactive CLI process; durable cross-process conversation history would require a persistent LangGraph checkpointer.

Chroma provides a small local persistent vector store and paper metadata filters. Sentence Transformers keeps embeddings local and free, at the cost of a first-use model download and local compute. The deterministic extractive briefing keeps all paper-specific fields explainable and ensures unsupported fields are not invented. One optional bounded LLM pass improves only the prose summary; it is not on the critical path. QA uses semantic retrieval because questions are open-ended, while the application remains authoritative for chunk IDs, exact excerpts, and page provenance. Ollama is the no-key default, avoiding a paid-service requirement; CPU inference can be slow, particularly for the optional summary and QA generations.

With more time, useful improvements would be evaluation on a broader paper/question set, stronger semantic entailment checks for answer paraphrases, better section detection for irregular PDF layouts, and a durable checkpoint backend. These are tradeoffs, not hidden guarantees of the current implementation.

## Known Limitations

- arXiv is the only source; there is no OCR for scanned PDFs.
- PDF extraction and section detection are best-effort and can be affected by layout.
- The briefing's field selection is heuristic and may leave information as not found.
- CPU-only local LLM inference may have noticeable latency; the optional briefing call is bounded and has an extractive fallback.
- Interactive graph checkpoint history is process-local even though PDF and Chroma data persist locally.

## Testing

Run the full offline suite (network, Ollama, and hosted-provider calls are mocked):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Final local verification: **108 tests passed** under Python 3.11.9.
