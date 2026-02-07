"""Unit tests for the RAG application.

Run with: RAG_TESTING=1 pytest tests/test_app.py -v
"""

import os
import sys
from pathlib import Path
import tempfile
import json

# Ensure project root is on sys.path before importing app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enable lightweight test mode before importing app
os.environ["RAG_TESTING"] = "1"

import pytest
import httpx

import app as app_module


# =============================================================================
# Pytest fixture for async client
# =============================================================================

@pytest.fixture
def client():
    """Create async client for testing."""
    transport = httpx.ASGITransport(app=app_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# =============================================================================
# Health & Stats Endpoints
# =============================================================================

@pytest.mark.asyncio
async def test_health_check(client):
    """Test /api/health endpoint returns ok status."""
    async with client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_get_vector_stores(client):
    """Test /api/vector-stores endpoint returns available backends."""
    async with client:
        response = await client.get("/api/vector-stores")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data


# =============================================================================
# Chat Endpoints
# =============================================================================

@pytest.mark.asyncio
async def test_chat_basic(client):
    """Test basic chat endpoint with valid question."""
    async with client:
        response = await client.post("/api/chat", json={
            "question": "What is RAG?",
            "model": "flash",
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


@pytest.mark.asyncio
async def test_chat_empty_question_returns_400(client):
    """Test that empty question returns 400."""
    async with client:
        response = await client.post("/api/chat", json={"question": "   "})
        assert response.status_code == 400
        assert "empty" in response.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_chat_with_conversation_id(client):
    """Test chat with conversation_id for multi-turn context."""
    async with client:
        response = await client.post("/api/chat", json={
            "question": "Hello",
            "conversation_id": "test-conversation-123",
        })
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_stream_basic(client):
    """Test streaming chat endpoint (uses stream=True param)."""
    async with client:
        response = await client.post("/api/chat", json={
            "question": "Explain RAG",
            "model": "flash",
            "stream": True,
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_stream_empty_question(client):
    """Test that empty question in stream mode is rejected."""
    async with client:
        response = await client.post("/api/chat", json={"question": "", "stream": True})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_stream_backend_error_event(monkeypatch, client):
    """Test streaming response emits error event when backend raises."""
    async def fake_stream(*_args, **_kwargs):
        yield {"token": "partial"}
        raise RuntimeError("stream failed")

    # Force retrieval to return at least one chunk so streaming path is used
    def fake_retrieve(*_args, **_kwargs):
        return ["stub chunk"], [{}]

    monkeypatch.setattr(app_module.llm_client, "generate_answer_stream", fake_stream)
    monkeypatch.setattr(app_module.rag_engines["chroma"], "retrieve", fake_retrieve)

    async with client:
        response = await client.post("/api/chat", json={
            "question": "Trigger stream error",
            "stream": True,
        })
        assert response.status_code == 200
        # Collect streamed body and confirm error event is present
        body = b""
        async for chunk in response.aiter_bytes():
            body += chunk
        assert b"data: {\"error\":" in body


# =============================================================================
# Conversation ID Validation (Security) - Sync tests
# =============================================================================

def test_conversation_id_validation_path_traversal():
    """Test that path traversal attacks are rejected."""
    assert not app_module._validate_conversation_id("../evil")
    assert not app_module._validate_conversation_id("..")
    assert not app_module._validate_conversation_id("foo/../bar")


def test_conversation_id_validation_absolute_path():
    """Test that absolute paths are rejected."""
    assert not app_module._validate_conversation_id("/etc/passwd")
    assert not app_module._validate_conversation_id("\\windows\\system32")


def test_conversation_id_validation_hidden_files():
    """Test that hidden files are rejected."""
    assert not app_module._validate_conversation_id(".hidden")
    assert not app_module._validate_conversation_id(".env")


def test_conversation_id_validation_valid():
    """Test that valid conversation IDs are accepted."""
    assert app_module._validate_conversation_id("abc123")
    assert app_module._validate_conversation_id("conv-2024-01-01")
    assert app_module._validate_conversation_id("user_session_xyz")


def test_conversation_id_validation_empty():
    """Test that empty/None IDs are rejected."""
    assert not app_module._validate_conversation_id("")
    assert not app_module._validate_conversation_id(None)


# =============================================================================
# Document Management Endpoints
# =============================================================================

@pytest.mark.asyncio
async def test_list_documents(client):
    """Test /api/documents returns list."""
    async with client:
        response = await client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_upload_invalid_extension(client):
    """Test that unsupported file extensions are rejected."""
    async with client:
        # Use a multipart form for file upload
        files = {"file": ("malware.exe", b"fake content", "application/octet-stream")}
        response = await client.post("/api/upload", files=files)
        assert response.status_code == 400
        detail = response.json().get("detail", "").lower()
        assert "unsupported" in detail or "file type" in detail


@pytest.mark.asyncio
async def test_upload_valid_txt(client):
    """Test uploading a valid .txt file."""
    async with client:
        response = await client.post(
            "/api/upload",
            files={"file": ("test.txt", b"This is test content.", "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    """Test deleting a document that doesn't exist."""
    async with client:
        response = await client.delete("/api/documents/nonexistent_file_xyz.txt")
        assert response.status_code == 200


# =============================================================================
# Chat History Endpoints
# =============================================================================

@pytest.mark.asyncio
async def test_clear_chat_history_invalid_id(client):
    """Test clearing history with path traversal attack."""
    async with client:
        # Path with .. segments should be rejected as invalid
        response = await client.delete("/api/chat/history/foo..bar")
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_clear_chat_history_valid(client):
    """Test clearing history with valid conversation_id."""
    async with client:
        response = await client.delete("/api/chat/history/valid-conv-id")
        assert response.status_code == 200


# =============================================================================
# Feedback Endpoint
# =============================================================================

@pytest.mark.asyncio
async def test_feedback_thumbs_up(client):
    """Test submitting positive feedback."""
    async with client:
        response = await client.post("/api/feedback", json={
            "message_id": "msg-123",
            "feedback": "up",
        })
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_feedback_thumbs_down_with_details(client):
    """Test submitting negative feedback."""
    async with client:
        response = await client.post("/api/feedback", json={
            "message_id": "msg-456",
            "feedback": "down",
        })
        assert response.status_code == 200


# =============================================================================
# Model Mapping
# =============================================================================

def test_model_mapping_exists():
    """Test that MODEL_MAPPING contains expected keys."""
    assert "flash" in app_module.MODEL_MAPPING
    assert "pro" in app_module.MODEL_MAPPING
    assert "gemini-1.5-flash" in app_module.MODEL_MAPPING


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.asyncio
async def test_chat_with_special_characters(client):
    """Test chat handles special characters in question."""
    async with client:
        response = await client.post("/api/chat", json={
            "question": "What about <script>alert('xss')</script>?",
        })
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_very_long_question(client):
    """Test chat handles very long questions."""
    async with client:
        long_question = "What is " + "very " * 500 + "important?"
        response = await client.post("/api/chat", json={"question": long_question})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_unicode_question(client):
    """Test chat handles unicode characters."""
    async with client:
        response = await client.post("/api/chat", json={
            "question": "Açıklayınız: 日本語 и русский 🎉",
        })
        assert response.status_code == 200
