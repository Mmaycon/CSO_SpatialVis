from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SQLITE = (_REPO_ROOT / "data" / "dev_annotations.sqlite").as_posix()
# Local dev: use repo ./data so manifest + samples resolve without CSO_DATA_ROOT.
# Docker sets CSO_DATA_ROOT=/app/data in the container, which overrides this default.
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CSO_", env_file=".env", extra="ignore")

    data_root: Path = _DEFAULT_DATA_ROOT
    # Local dev defaults to SQLite under ./data so ROI annotations persist without Postgres.
    database_url: str = f"sqlite:///{_DEFAULT_SQLITE}"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
    max_cells_per_viewport: int = 150_000
    max_polygons_per_viewport: int = 100_000
    gene_cache_size: int = 32


settings = Settings()
