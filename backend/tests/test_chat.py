def test_chat_streams_response_and_creates_session(client):
    response = client.post("/api/chat", json={"message": "What is in the document?"})
    assert response.status_code == 200
    assert "event: session" in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text


def test_chat_rejects_unknown_session(client):
    response = client.post(
        "/api/chat", json={"session_id": "does-not-exist", "message": "hi"}
    )
    assert response.status_code == 404
