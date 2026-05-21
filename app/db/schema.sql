CREATE EXTENSION IF NOT EXISTS vector;

-- SQLAlchemy models in app/db/models.py are the source of truth for the
-- application schema. This file documents the required pgvector extension for
-- local bootstrap and migration tooling.
