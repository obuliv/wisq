from app.llm.interfaces import Message
from app.llm.openai_client import OpenAIClient


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Choice:
            class _Message:
                content = "ok"
                tool_calls = None

            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


def _client_with_recorder(reasoning_effort: str = "") -> tuple[OpenAIClient, _RecordingCompletions]:
    client = OpenAIClient(api_key="test", model="gpt-4o-mini", reasoning_effort=reasoning_effort)
    recorder = _RecordingCompletions()
    client._client.chat.completions = recorder
    return client, recorder


def test_reasoning_effort_omitted_by_default():
    client, recorder = _client_with_recorder()
    client.generate_with_tools([Message(role="user", content="hi")], tools=[])
    assert "reasoning_effort" not in recorder.calls[0]


def test_reasoning_effort_included_when_configured():
    # Reasoning models (e.g. gpt-5.x) reject tool calls on chat.completions
    # unless reasoning_effort is set explicitly; non-reasoning models reject
    # the param outright if it's sent at all -- so this must stay opt-in,
    # verified here by asserting it's included ONLY when configured.
    client, recorder = _client_with_recorder(reasoning_effort="none")
    client.generate_with_tools([Message(role="user", content="hi")], tools=[])
    assert recorder.calls[0]["reasoning_effort"] == "none"


def test_reasoning_effort_applies_to_plain_generate_too():
    client, recorder = _client_with_recorder(reasoning_effort="none")
    list(client.generate([Message(role="user", content="hi")], stream=False))
    assert recorder.calls[0]["reasoning_effort"] == "none"
