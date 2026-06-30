"""SAHA – PostgreSQL async connection pool (asyncpg)."""
from __future__ import annotations

import asyncpg
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    POSTGRES_HOST:     str = "localhost"
    POSTGRES_PORT:     int = 5432
    POSTGRES_DB:       str = "saha"
    POSTGRES_USER:     str = "saha"
    POSTGRES_PASSWORD: str = "changeme"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = DBSettings()
        _pool = await asyncpg.create_pool(
            dsn=settings.dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def run_migrations() -> None:
    """Apply all SQL migrations in order."""
    import pathlib

    pool = await get_pool()
    migrations_dir = pathlib.Path(__file__).parent / "migrations"
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        async with pool.acquire() as conn:
            await conn.execute(sql)
