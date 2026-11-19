import logging
from types import SimpleNamespace

import httpx
import pandas as pd
import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import api


class FakeQdrant:
    def __init__(self):
        self.created = False
        self.points = []
        self.query_calls = []
        self.deleted_collections = []

    def collection_exists(self, collection_name):
        return self.created

    def create_collection(self, collection_name, vectors_config):
        self.created = True

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)
        self.created = False
        self.points.clear()

    def upsert(self, collection_name, wait, points):
        self.points.extend(points)

    def query_points(self, collection_name, query, limit, with_payload):
        self.query_calls.append({"query": query, "limit": limit})
        return SimpleNamespace(
            points=[
                SimpleNamespace(payload=point.payload, score=0.85)
                for point in self.points[:limit]
            ]
        )


def test_pdf_ingest_and_retrieve(monkeypatch, tmp_path):
    path = tmp_path / "guide.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Qdrant stores vector embeddings.")
        document.save(path)

    client = FakeQdrant()
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "_get_local_qdrant_client", lambda: client)
    monkeypatch.setattr(api, "_embed_texts", lambda texts: [[0.1] * api.VECTOR_SIZE for _ in texts])

    response = TestClient(api.app).post(
        "/ingest",
        json={"data_source_type": "pdf", "data_source_value": path.name},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "chunks_indexed": 1,
    }
    assert client.points[0].payload["page_or_row"] == 1
    assert client.points[0].payload["source_file"] == "guide.pdf"

    response = TestClient(api.app).post("/retrieve", json={"query": "vector embeddings"})
    assert response.status_code == 200
    assert response.json()["chunks"] == [
        {
            "text": "Qdrant stores vector embeddings.",
            "score": 0.85,
            "document_name": "guide.pdf",
            "page_number": 1,
        }
    ]


def test_ingest_recreates_collection_before_upsert(monkeypatch, tmp_path):
    path = tmp_path / "new.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "New context only.")
        document.save(path)

    client = FakeQdrant()
    client.created = True
    client.points.append(SimpleNamespace(payload={"source_file": "old.pdf"}))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "_get_local_qdrant_client", lambda: client)
    monkeypatch.setattr(api, "_embed_texts", lambda texts: [[0.1] * api.VECTOR_SIZE for _ in texts])

    response = TestClient(api.app).post(
        "/ingest",
        json={"data_source_type": "pdf", "data_source_value": path.name},
    )

    assert response.status_code == 200
    assert client.deleted_collections == [api.COLLECTION_NAME]
    assert [point.payload["source_file"] for point in client.points] == ["new.pdf"]


@pytest.mark.parametrize("query_key", ["query", "Query", "user_query", "var_UserQuery"])
def test_retrieve_accepts_query_aliases_and_defaults_top_k(monkeypatch, query_key):
    client = FakeQdrant()
    embedded_queries = []

    def embed_texts(texts):
        embedded_queries.extend(texts)
        return [[0.1] * api.VECTOR_SIZE for _ in texts]

    monkeypatch.setattr(api, "_get_local_qdrant_client", lambda: client)
    monkeypatch.setattr(api, "_embed_texts", embed_texts)

    response = TestClient(api.app).post(
        "/retrieve",
        json={query_key: "  search text  "},
    )

    assert response.status_code == 200
    assert embedded_queries == ["search text"]
    assert client.query_calls[0]["limit"] == 5


def test_retrieve_logs_raw_body_and_extracts_nested_query(monkeypatch, caplog):
    client = FakeQdrant()
    embedded_queries = []

    def embed_texts(texts):
        embedded_queries.extend(texts)
        return [[0.1] * api.VECTOR_SIZE for _ in texts]

    monkeypatch.setattr(api, "_get_local_qdrant_client", lambda: client)
    monkeypatch.setattr(api, "_embed_texts", embed_texts)
    raw_body = b'{ "payload": { "form": { "Query": "  nested search  " } } }'
    caplog.set_level(logging.INFO, logger=api.logger.name)

    response = TestClient(api.app).post(
        "/retrieve",
        content=raw_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert repr(raw_body) in caplog.text
    assert embedded_queries == ["nested search"]
    assert client.query_calls[0]["limit"] == 5


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"var_UserQuery": 123},
        {"query": "search text", "top_k": 0},
        [],
    ],
)
def test_retrieve_invalid_json_objects_return_400_not_422(body):
    response = TestClient(api.app).post("/retrieve", json=body)

    assert response.status_code == 400


def test_retrieve_malformed_json_returns_400_not_422():
    response = TestClient(api.app).post(
        "/retrieve",
        content=b"{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


def test_local_client_uses_requested_persistent_path(monkeypatch):
    created = {}

    class ClientStub:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(api, "QdrantClient", ClientStub)
    api._get_local_qdrant_client.cache_clear()

    try:
        assert isinstance(api._get_local_qdrant_client(), ClientStub)
        assert created == {"path": "./local_qdrant_db"}
    finally:
        api._get_local_qdrant_client.cache_clear()


def test_persistent_local_qdrant_ingest_and_retrieve(monkeypatch, tmp_path):
    path = tmp_path / "guide.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Local fallback keeps documents available.")
        document.save(path)

    local_path = tmp_path / "qdrant_local_data"
    local_client = QdrantClient(path=str(local_path))

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "_get_local_qdrant_client", lambda: local_client)
    monkeypatch.setattr(
        api,
        "_embed_texts",
        lambda texts: [[0.1] * api.VECTOR_SIZE for _ in texts],
    )

    try:
        client = TestClient(api.app)
        response = client.post(
            "/ingest",
            json={"data_source_type": "pdf", "data_source_value": path.name},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "chunks_indexed": 1}
        assert local_path.is_dir()

        local_client.close()
        local_client = QdrantClient(path=str(local_path))
        response = client.post("/retrieve", json={"query": "local fallback"})
        assert response.status_code == 200
        assert response.json()["chunks"][0]["text"] == "Local fallback keeps documents available."
        assert response.json()["chunks"][0]["document_name"] == "guide.pdf"
    finally:
        local_client.close()


def test_excel_rows_keep_row_numbers(monkeypatch, tmp_path):
    path = tmp_path / "records.xlsx"
    pd.DataFrame([{"title": "First", "body": "Hello"}]).to_excel(path, index=False)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)

    chunks = api._parse_source("excel", path.name)

    assert len(chunks) == 1
    assert chunks[0].source_file == "records.xlsx"
    assert chunks[0].page_or_row == 2
    assert "title: First" in chunks[0].text


def test_url_html_is_extracted_and_attributed(monkeypatch):
    url = "https://example.com/article"
    article = "Useful article content. " * 15
    html = (
        "<html><head><script>script noise</script><style>style noise</style></head>"
        "<body><nav>navigation noise</nav><header>header noise</header>"
        f"<main><p>{article}</p></main><footer>footer noise</footer></body></html>"
    ).encode()
    monkeypatch.setattr(api, "_fetch_url", lambda requested_url: html)

    chunks = api._extract_url(url)

    assert len(chunks) == 1
    assert chunks[0].text == article.strip()
    assert not any(
        noise in chunks[0].text
        for noise in ("script noise", "style noise", "navigation noise", "header noise", "footer noise")
    )
    assert chunks[0].source_file == url
    assert chunks[0].page_or_row == 1


def test_url_with_short_text_falls_back_to_rendered_content(monkeypatch):
    url = "https://example.com/article"
    rendered = "Rendered article content. " * 12
    fallback_calls = []
    monkeypatch.setattr(api, "_fetch_url", lambda requested_url: b"<p>Short</p>")
    monkeypatch.setattr(
        api,
        "_fetch_url_fallback",
        lambda requested_url: fallback_calls.append(requested_url)
        or f"<article>{rendered}</article>".encode(),
    )

    chunks = api._extract_url(url)

    assert fallback_calls == [url]
    assert chunks[0].text == rendered.strip()


def test_forbidden_url_uses_fallback(monkeypatch):
    url = "https://example.com/article"
    rendered = "Rendered article content. " * 12
    fallback_calls = []

    def forbidden_request(requested_url):
        raise api._URLForbiddenError("Forbidden")

    monkeypatch.setattr(api, "_fetch_url", forbidden_request)
    monkeypatch.setattr(
        api,
        "_fetch_url_fallback",
        lambda requested_url: fallback_calls.append(requested_url)
        or f"<article>{rendered}</article>".encode(),
    )

    chunks = api._extract_url(url)

    assert fallback_calls == [url]
    assert chunks[0].text == rendered.strip()


def test_url_fetch_uses_browser_headers(monkeypatch):
    seen = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"<html></html>"

    class FakeClient:
        def __init__(self, timeout, follow_redirects):
            assert timeout == 20
            assert follow_redirects is False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def stream(self, method, url, headers):
            seen.update(headers)
            return FakeResponse()

    monkeypatch.setattr(api, "_validate_public_url", lambda url: None)
    monkeypatch.setattr(api.httpx, "Client", FakeClient)

    api._fetch_url("https://example.com/article")

    assert "Chrome/" in seen["User-Agent"]
    assert seen["Accept-Language"] == "en-US,en;q=0.9"


def test_url_fallback_uses_urllib_when_playwright_is_unavailable(monkeypatch):
    expected = b"<html>content</html>"
    monkeypatch.setattr(api, "_validate_public_url", lambda url: None)
    monkeypatch.setattr(api, "_fetch_url_with_playwright", lambda url: None)
    monkeypatch.setattr(api, "_fetch_url_with_urllib", lambda url: expected)

    assert api._fetch_url_fallback("https://example.com/article") == expected
