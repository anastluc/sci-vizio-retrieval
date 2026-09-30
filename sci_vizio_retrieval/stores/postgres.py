"""PostgreSQL + pgvector backed vector store."""

import logging
import re
from typing import List

import numpy as np

from sci_vizio_retrieval.embeddings import TextEmbedder
from sci_vizio_retrieval.stores.base import IndexRecord, SearchResult, VectorStore

logger = logging.getLogger(__name__)

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class PostgresVectorStore(VectorStore):
    """
    Stores each analysed image as one row: the vision-LLM JSON (``JSONB``),
    the raw image (``BYTEA``) and a pgvector embedding of the description,
    searched by cosine distance through an HNSW index.

    The extension, table and indexes are created on first use. The database
    user needs permission to ``CREATE EXTENSION vector`` unless it already exists.

    Args:
        dsn: libpq connection string, e.g. ``postgresql://user:pass@host:5432/db``.
        embedder: Produces the description embeddings. Its ``dimension`` fixes
            the column size when the table is created.
        table: Table name (letters, digits and underscores only).
    """

    backend = "postgres"

    def __init__(self, dsn: str, embedder: TextEmbedder, table: str = "image_analysis"):
        import psycopg
        from psycopg import sql

        if not _TABLE_NAME_RE.match(table):
            raise ValueError(f"Invalid PostgreSQL table name: {table!r}")

        self.table = table
        self.embedder = embedder
        self._sql = sql
        self._table = sql.Identifier(table)

        self._conn = psycopg.connect(dsn, autocommit=True)
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        from pgvector.psycopg import register_vector

        sql = self._sql
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id              TEXT PRIMARY KEY,
                    pdf_file        TEXT NOT NULL,
                    image_path      TEXT NOT NULL,
                    image_type      TEXT,
                    title           TEXT,
                    analysis        JSONB NOT NULL,
                    image_data      BYTEA,
                    embedding       vector({dim}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """).format(table=self._table, dim=sql.Literal(self.embedder.dimension)))
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw (embedding vector_cosine_ops)"
            ).format(idx=sql.Identifier(f"{self.table}_embedding_idx"), table=self._table))
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {table} (pdf_file)"
            ).format(idx=sql.Identifier(f"{self.table}_pdf_file_idx"), table=self._table))

            self._check_compatibility(cur)

    def _check_compatibility(self, cur) -> None:
        """Refuse to mix embeddings from different models in one table."""
        cur.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (self.table,),
        )
        (existing_dim,) = cur.fetchone()
        if existing_dim != self.embedder.dimension:
            raise ValueError(
                f"Table {self.table!r} stores {existing_dim}-dimensional embeddings but the "
                f"embedder {self.embedder.model_name!r} produces {self.embedder.dimension}. "
                "Use a different postgres_table or drop the table and re-index."
            )

        cur.execute(self._sql.SQL(
            "SELECT DISTINCT embedding_model FROM {table} WHERE embedding_model <> %s LIMIT 1"
        ).format(table=self._table), (self.embedder.model_name,))
        other = cur.fetchone()
        if other:
            raise ValueError(
                f"Table {self.table!r} already contains embeddings from {other[0]!r}; "
                f"refusing to add embeddings from {self.embedder.model_name!r}. "
                "Use a different postgres_table or drop the table and re-index."
            )

    @property
    def name(self) -> str:
        info = self._conn.info
        return f"postgres:{info.host}:{info.port}/{info.dbname}/{self.table}"

    def upsert(self, record: IndexRecord) -> None:
        from psycopg.types.json import Jsonb

        (embedding,) = self.embedder.embed([record.document_text])
        with self._conn.cursor() as cur:
            cur.execute(self._sql.SQL("""
                INSERT INTO {table}
                    (id, pdf_file, image_path, image_type, title, analysis,
                     image_data, embedding, embedding_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    pdf_file = EXCLUDED.pdf_file,
                    image_path = EXCLUDED.image_path,
                    image_type = EXCLUDED.image_type,
                    title = EXCLUDED.title,
                    analysis = EXCLUDED.analysis,
                    image_data = EXCLUDED.image_data,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    indexed_at = now()
            """).format(table=self._table), (
                record.id,
                record.pdf_file,
                record.image_path,
                record.image_type,
                record.title,
                Jsonb(record.analysis),
                record.image_bytes,
                np.asarray(embedding, dtype=np.float32),
                self.embedder.model_name,
            ))

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        (query_vec,) = self.embedder.embed([query])
        query_vec = np.asarray(query_vec, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(self._sql.SQL("""
                SELECT id, pdf_file, image_path, analysis, image_data,
                       embedding <=> %s AS distance
                FROM {table}
                ORDER BY embedding <=> %s
                LIMIT %s
            """).format(table=self._table), (query_vec, query_vec, n_results))
            rows = cur.fetchall()

        return [
            SearchResult(
                id=row[0],
                pdf_file=row[1],
                image_path=row[2],
                analysis=row[3],
                image_bytes=bytes(row[4]) if row[4] is not None else None,
                distance=float(row[5]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(self._sql.SQL("SELECT count(*) FROM {table}").format(table=self._table))
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
