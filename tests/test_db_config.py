from app.core.config import Settings


def test_database_url_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db:5432/test_db")
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://user:pass@db:5432/test_db"


def test_default_database_url_uses_postgres_driver() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
