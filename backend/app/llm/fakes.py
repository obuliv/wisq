from collections.abc import Iterable

from app.llm.interfaces import Message


class FakeLLMClient:
    """Echoes the prompt it was given, streamed word-by-word. Lets the API/UI/chat
    orchestration be exercised end-to-end without a real provider or API key.
    Swap for a real provider-backed LLMClient when the LLM design lands."""

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        reply = f"[fake-llm] You asked: {last_user!r}. Here's what the retrieved context contained."
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
