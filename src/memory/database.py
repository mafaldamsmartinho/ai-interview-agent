from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATABASE_PATH = DATA_DIR / "interview_memory.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)
