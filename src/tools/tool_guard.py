from dataclasses import dataclass
from enum import Enum


class AuthorizationTier(Enum):
    AUTONOMOUS = 1  # execute automatically
    NOTIFY = 2  # execute, then notify user
    APPROVE = 3  # require human approval before execution


@dataclass
class ToolClassification:
    tool_name: str
    tier: AuthorizationTier
    reason: str


TOOL_TIERS = {
    "get_question_bank": ToolClassification(
        tool_name="get_question_bank",
        tier=AuthorizationTier.AUTONOMOUS,
        reason="Read-only access to interview questions.",
    ),
    "get_topic_notes": ToolClassification(
        tool_name="get_topic_notes",
        tier=AuthorizationTier.AUTONOMOUS,
        reason="Read-only access to topic reference notes.",
    ),
    "get_candidate_history": ToolClassification(
        tool_name="get_candidate_history",
        tier=AuthorizationTier.AUTONOMOUS,
        reason="Read-only access to candidate interview history.",
    ),
    "save_interview_result": ToolClassification(
        tool_name="save_interview_result",
        tier=AuthorizationTier.APPROVE,
        reason="Writes persistent candidate data and requires approval.",
    ),
}


def classify_tool(tool_name: str) -> ToolClassification:
    """
    Classify a tool according to its authorization tier.

    Unknown tools default to APPROVE.
    """

    if tool_name in TOOL_TIERS:
        return TOOL_TIERS[tool_name]

    return ToolClassification(
        tool_name=tool_name,
        tier=AuthorizationTier.APPROVE,
        reason="Unknown tool — human approval required by default.",
    )


def guard_tool_call(tool_name: str):
    classification = classify_tool(tool_name)

    if classification.tier == AuthorizationTier.AUTONOMOUS:
        return "allow"

    if classification.tier == AuthorizationTier.NOTIFY:
        return "allow_and_notify"

    if classification.tier == AuthorizationTier.APPROVE:
        return "confirm"
