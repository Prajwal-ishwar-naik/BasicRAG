from __future__ import annotations

import ipaddress
import logging
import socket
import uuid
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request as UrlRequest, build_opener

import httpx
import pandas as pd
import pymupdf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

load_dotenv(Path(__file__).with_name(".env"), override=False)

logger = logging.getLogger(__name__)
app = FastAPI(title="UiPath RAG Agent API")

COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
MAX_SOURCE_BYTES = 20 * 1024 * 1024
DATA_DIR = (Path(__file__).parent / "data").resolve()
LOCAL_QDRANT_PATH = "./local_qdrant_db"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class IngestRequest(BaseModel):
    data_source_type: Literal["pdf", "excel", "url"]
    data_source_value: str = Field(min_length=1)


class IngestResponse(BaseModel):
    status: Literal["success"]
    chunks_indexed: int


class RetrievedChunk(BaseModel):
    text: str
    score: float
    document_name: str
    page_number: int


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source_file: str
    page_or_row: int


def _chunk_text(text: str, source_file: str, page_or_row: int) -> list[DocumentChunk]:
    normalized = " ".join(text.split())
    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(normalized):
        end = min(start + MAX_CHUNK_CHARS, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(DocumentChunk(chunk, source_file, page_or_row))
        if end == len(normalized):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_CHARS)

    return chunks


def _resolve_local_file(value: str) -> Path:
    supplied = Path(value).expanduser()
    candidate = supplied if supplied.is_absolute() else DATA_DIR / supplied
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Source file was not found in the configured data directory: {value}") from exc

    if not resolved.is_relative_to(DATA_DIR):
        pass
    if not resolved.is_file():
        raise ValueError("The source path must be a file.")
    if resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("The source file exceeds the 20 MiB limit.")
    return resolved


def _extract_pdf(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            chunks.extend(_chunk_text(page.get_text(), path.name, page_index))
    return chunks


def _extract_excel(path: Path) -> list[DocumentChunk]:
    sheet = pd.read_excel(path, sheet_name=0, dtype=object).fillna("")
    chunks: list[DocumentChunk] = []
    for row_number, (_, row) in enumerate(sheet.iterrows(), start=2):
        values = [f"{column}: {value}" for column, value in row.items() if str(value).strip()]
        if values:
            chunks.extend(_chunk_text(" | ".join(values), path.name, row_number))
    return chunks


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL sources must use an http or https URL.")

    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or default_port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("The URL host could not be resolved.") from exc

    for address in addresses:
        resolved_ip = ipaddress.ip_address(address[4][0])
        if not resolved_ip.is_global:
            raise ValueError("URL sources must resolve to a public IP address.")


class _URLForbiddenError(Exception):
    pass


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_url(url: str) -> bytes:
    _validate_public_url(url)
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            with client.stream("GET", url, headers=BROWSER_HEADERS) as response:
                if 300 <= response.status_code < 400:
                    raise ValueError("URL redirects are not supported.")
                response.raise_for_status()
                content = bytearray()
                for part in response.iter_bytes():
                    content.extend(part)
                    if len(content) > MAX_SOURCE_BYTES:
                        raise ValueError("The URL response exceeds the 20 MiB limit.")
                return bytes(content)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            raise _URLForbiddenError("The website denied the initial request.") from exc
        raise ValueError("The URL could not be fetched successfully.") from exc
    except httpx.HTTPError as exc:
        raise ValueError("The URL could not be fetched successfully.") from exc


def _guard_playwright_request(route) -> None:
    try:
        _validate_public_url(route.request.url)
    except ValueError:
        route.abort()
    else:
        route.continue_()


def _fetch_url_with_playwright(url: str) -> bytes | None:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    user_agent=BROWSER_HEADERS["User-Agent"],
                    extra_http_headers={"Accept-Language": BROWSER_HEADERS["Accept-Language"]},
                )
                page = context.new_page()
                page.route("**/*", _guard_playwright_request)
                page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                page.wait_for_timeout(1_000)
                content = page.content().encode("utf-8")
                if len(content) > MAX_SOURCE_BYTES:
                    raise ValueError("The URL response exceeds the 20 MiB limit.")
                return content
            finally:
                browser.close()
    except PlaywrightError:
        logger.warning("Playwright URL fallback failed; trying urllib.", exc_info=True)
        return None


def _fetch_url_with_urllib(url: str) -> bytes:
    _validate_public_url(url)
    request = UrlRequest(url, headers=BROWSER_HEADERS)
    opener = build_opener(_PublicRedirectHandler())
    try:
        with opener.open(request, timeout=20) as response:
            content = response.read(MAX_SOURCE_BYTES + 1)
            if len(content) > MAX_SOURCE_BYTES:
                raise ValueError("The URL response exceeds the 20 MiB limit.")
            return content
    except (URLError, TimeoutError) as exc:
        raise ValueError("The URL could not be fetched successfully.") from exc


def _fetch_url_fallback(url: str) -> bytes:
    _validate_public_url(url)
    rendered_html = _fetch_url_with_playwright(url)
    if rendered_html is not None:
        return rendered_html
    return _fetch_url_with_urllib(url)


def _extract_page_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        element.decompose()
    return soup.get_text(" ", strip=True)


def _extract_url(url: str) -> list[DocumentChunk]:
    fallback_used = False
    try:
        html = _fetch_url(url)
    except _URLForbiddenError:
        logger.info("Initial URL request returned 403; using fallback fetch.")
        html = _fetch_url_fallback(url)
        fallback_used = True

    text = _extract_page_text(html)
    if len(text) < 200 and not fallback_used:
        logger.info("Initial URL response had fewer than 200 text characters; using fallback fetch.")
        fallback_text = _extract_page_text(_fetch_url_fallback(url))
        if len(fallback_text) > len(text):
            text = fallback_text
    return _chunk_text(text, url, 1)


def _parse_source(source_type: str, source_value: str) -> list[DocumentChunk]:
    if source_type == "url":
        return _extract_url(source_value)

    path = _resolve_local_file(source_value)
    expected_suffixes = {
        "pdf": {".pdf"},
        "excel": {".xls", ".xlsx"},
    }
    if path.suffix.lower() not in expected_suffixes[source_type]:
        raise ValueError(f"Source file extension does not match data_source_type '{source_type}'.")
    if source_type == "pdf":
        return _extract_pdf(path)
    return _extract_excel(path)


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _get_embedding_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()


@lru_cache(maxsize=1)
def _get_local_qdrant_client() -> QdrantClient:
    return QdrantClient(path=LOCAL_QDRANT_PATH)


def _recreate_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )


def _document_chunks(request: IngestRequest) -> list[DocumentChunk]:
    try:
        chunks = _parse_source(request.data_source_type, request.data_source_value)
    except (OSError, ValueError, pymupdf.FileDataError, pd.errors.ParserError) as exc:
        print("INGEST_ERR:", repr(exc), flush=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=400, detail="The data source contains no indexable text.")
    return chunks


def _extract_query(body: object) -> str | None:
    if isinstance(body, dict):
        for key in ("query", "Query", "user_query", "var_UserQuery"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in body.values():
            query = _extract_query(value)
            if query is not None:
                return query
    elif isinstance(body, list):
        for value in body:
            query = _extract_query(value)
            if query is not None:
                return query
    return None


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    request.data_source_value = request.data_source_value.strip().strip("\"").strip("'")
    print("INGEST_REQ:", request.data_source_type, repr(request.data_source_value), flush=True)
    chunks = _document_chunks(request)
    try:
        vectors = _embed_texts([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "page_or_row": chunk.page_or_row,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        local_client = _get_local_qdrant_client()
        _recreate_collection(local_client)
        local_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )
    except Exception as exc:
        logger.exception("Document ingestion failed.")
        raise HTTPException(status_code=502, detail="Embedding or Qdrant indexing failed.") from exc

    return IngestResponse(
        status="success",
        chunks_indexed=len(chunks),
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: Request) -> RetrieveResponse:
    raw_body = await request.body()
    logger.info("RAW_RETRIEVE_BODY: %r", raw_body)

    try:
        body = loads(raw_body)
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    query = _extract_query(body)
    if query is None:
        raise HTTPException(
            status_code=400,
            detail="Request body must include a non-empty query, Query, user_query, or var_UserQuery string.",
        )

    try:
        top_k_value = float(body.get("top_k", 5))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.") from exc
    if not top_k_value.is_integer() or top_k_value < 1:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
    top_k = int(top_k_value)

    try:
        query_vector = _embed_texts([query])[0]
        result = _get_local_qdrant_client().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        chunks = []
        for point in result.points:
            payload = point.payload or {}
            source_file = payload.get("source_file")
            page_or_row = payload.get("page_or_row")
            text = payload.get("text")
            if not isinstance(source_file, str) or not isinstance(text, str) or not isinstance(page_or_row, int):
                raise ValueError("A Qdrant result is missing required document metadata.")
            chunks.append(
                RetrievedChunk(
                    text=text,
                    score=float(point.score),
                    document_name=source_file,
                    page_number=page_or_row,
                )
            )
    except Exception as exc:
        logger.exception("Qdrant retrieval failed.")
        raise HTTPException(status_code=502, detail="Embedding or Qdrant retrieval failed.") from exc

    return RetrieveResponse(chunks=chunks)
