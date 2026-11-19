# BasicRAG — UiPath Maestro + FastAPI + Qdrant

BasicRAG is an end-to-end retrieval-augmented generation (RAG) workflow. UiPath Maestro orchestrates document ingestion, user input, retrieval, and a grounded answer; a Python FastAPI service extracts and indexes source content; and local Qdrant provides vector search.

## Architecture

```text
Maestro manual trigger
  └─ source type + source value
       └─ POST /ingest ──> FastAPI extracts and chunks content
                            └─ BAAI/bge-small-en-v1.5 embeddings
                                 └─ Qdrant local vector collection
  └─ user question
       └─ POST /retrieve ─> query embedding + top-k similar chunks
            └─ Maestro RAG Answer Agent ─> cited answer
```

The flow is in `MyProject/`; the API and its tests are in `UiPath_RAG_Agent/agent/`. The backend uses Qdrant's local persistent mode, writing its database under `local_qdrant_db/`.

## Features

- Ingest PDF, Excel (`.xls`/`.xlsx`), and public web page sources.
- Extract text, split it into overlapping chunks, and create normalized sentence-transformer embeddings.
- Retrieve the most relevant chunks with cosine similarity and include source/page-or-row metadata.
- Ask the Maestro agent to answer only from retrieved context and cite claims as `[document_name, Page/Row X]`.
- Keep API tests isolated with a mock Qdrant client for ingestion and retrieval behavior.

Each `/ingest` call replaces the current `rag_docs` collection with the newly indexed source. This sample is intended for local development and demonstration; secure and deploy the API appropriately before connecting it to a production Maestro process.

## Run the API locally

Requirements: Python 3.10 or later.

```bash
cd UiPath_RAG_Agent/agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

The first embedding request downloads `BAAI/bge-small-en-v1.5`. Once the service is running, interactive API documentation is available at <http://localhost:8000/docs>.

For PDF or Excel ingestion, put the input file in `UiPath_RAG_Agent/agent/data/` and pass its filename as `data_source_value`. URL ingestion accepts public `http` or `https` pages.

### API examples

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"data_source_type":"pdf","data_source_value":"handbook.pdf"}'

curl -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the leave policy?","top_k":5}'
```

`data_source_type` accepts `pdf`, `excel`, or `url`. Retrieval returns ranked text chunks with a similarity score and source location.

## Connect the Maestro flow

Open `MyProject/project.uiproj` in UiPath Studio Web/Maestro (or use the included `MySolution.uipx` solution). The flow takes a source type and value, calls `/ingest`, prompts for a question, calls `/retrieve`, then passes the returned chunks and question to the RAG Answer Agent.

Maestro must be able to reach the API over HTTP. For a cloud-hosted process, deploy the API or use a secure, temporary public tunnel during development; `localhost` on your computer is not reachable from cloud Maestro. Set the base URL in both HTTP activities in `MyProject/MyProject.flow` to your reachable API host, preserving the `/ingest` and `/retrieve` paths. Configure the UiPath HTTP connection in your tenant before running the flow.

## Tests

```bash
cd UiPath_RAG_Agent/agent
python -m pip install pytest
pytest -q
```

Local environments and generated vector databases are excluded from Git. Do not commit credentials, source documents, or Qdrant data.
