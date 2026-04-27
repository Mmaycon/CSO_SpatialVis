from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if _sqlite else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RegionAnnotationRow(Base):
    __tablename__ = "region_annotations"

    id = Column(String(64), primary_key=True)
    sample_id = Column(String(256), index=True, nullable=False)
    label = Column(String(512), nullable=False)
    author = Column(String(256), nullable=False)
    notes = Column(Text, default="")
    geometry_json = Column(Text, nullable=False)  # GeoJSON geometry
    confidence = Column(Float, default=1.0)
    cells_assigned = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class CellRegionTagRow(Base):
    """Derived tags from ROI — do not mutate core cell table on disk."""

    __tablename__ = "cell_region_tags"
    __table_args__ = (UniqueConstraint("sample_id", "cell_id", name="uq_cell_region_cell"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(256), index=True, nullable=False)
    cell_id = Column(String(128), index=True, nullable=False)
    annotation_id = Column(String(64), index=True, nullable=False)
    pathology_region = Column(String(512), nullable=False)
    pathology_region_source = Column(String(128), default="roi_assignment")
    pathology_region_confidence = Column(Float, default=1.0)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
