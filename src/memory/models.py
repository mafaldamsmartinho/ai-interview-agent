from sqlalchemy import Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.memory.database import Base


class SkillProfile(Base):
    __tablename__ = "skill_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    topic: Mapped[str] = mapped_column(String, nullable=False)
    skill: Mapped[str] = mapped_column(String, nullable=False)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    average_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Stored as JSON text for now, e.g. ["attention", "Q/K/V"]
    weaknesses: Mapped[str] = mapped_column(Text, default="[]")

    __table_args__ = (UniqueConstraint("topic", "skill", name="uq_topic_skill"),)
