import json

from sqlalchemy import select

from schemas import VERDICT_TO_SCORE
from src.memory.database import SessionLocal
from src.memory.models import SkillProfile

no_weakness_values = {
    "",
    "none",
    "none significant.",
    "none significant",
    "n/a",
}


def get_skill_profiles(topic: str) -> list[SkillProfile]:
    """Return all persistent skill profiles for a topic."""

    with SessionLocal() as session:
        statement = select(SkillProfile).where(SkillProfile.topic == topic)

        return list(session.scalars(statement).all())


def update_skill_profile(
    topic: str,
    skill: str,
    verdict: str,
    weakness: str,
) -> None:
    score = VERDICT_TO_SCORE[verdict] / 10

    with SessionLocal() as session:
        try:
            statement = select(SkillProfile).where(
                SkillProfile.topic == topic,
                SkillProfile.skill == skill,
            )

            profile = session.scalar(statement)

            if profile is None:
                profile = SkillProfile(
                    topic=topic,
                    skill=skill,
                    attempts=1,
                    average_score=score,
                    weaknesses=json.dumps([weakness] if weakness else []),
                )

                session.add(profile)

            else:
                total_score = profile.average_score * profile.attempts

                profile.attempts += 1
                profile.average_score = (total_score + score) / profile.attempts

                weaknesses = json.loads(profile.weaknesses)

                if (
                    weakness.strip().lower() not in no_weakness_values
                    and weakness not in weaknesses
                ):
                    weaknesses.append(weakness)

                profile.weaknesses = json.dumps(weaknesses)

            session.commit()

        except Exception:
            session.rollback()
            raise
