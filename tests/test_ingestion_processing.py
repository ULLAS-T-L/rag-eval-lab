from app.ingestion.store import IngestionStore


class _FakeDialect:
    name = "postgresql"


class _FakeBind:
    dialect = _FakeDialect()


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, statement):
        self.calls.append(statement)

    def commit(self):
        return None

    def get_bind(self):
        return _FakeBind()


class _DummyEmbeddingProvider:
    model_name = "dummy"
    dimensions = 3

    def embed_documents(self, texts):
        return [[0.0, 1.0, 0.0] for _ in texts]


def test_reset_ingestion_does_not_fail() -> None:
    session = _FakeSession()
    store = IngestionStore(session=session, embedding_provider=_DummyEmbeddingProvider())

    store.reset_for_reingest()

    assert session.calls
