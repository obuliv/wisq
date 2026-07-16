import math

from openai import OpenAI

from app.rag.interfaces import Vector


class OpenAIEmbedder:
    """Real Embedder backed by OpenAI's embeddings API (default:
    text-embedding-3-small). Swap point referenced in
    dependencies.py::get_embedder() -- selected via EMBEDDING_PROVIDER=openai
    in .env, reusing the same OPENAI_API_KEY as OpenAIClient."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[Vector]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [self._normalize(item.embedding) for item in response.data]

    @staticmethod
    def _normalize(vector: list[float]) -> Vector:
        # InMemoryVectorStore scores via a raw dot product (see rag/fakes.py's
        # _cosine), which only equals cosine similarity for unit-length vectors.
        # OpenAI's embeddings are already normalized, but normalize defensively
        # here so that guarantee doesn't have to be trusted implicitly.
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
