PLANNER_CASES = [
    {
        "verdict": "excellent",
        "confidence": "high",
        "missing_concepts": "None significant.",
        "allowed_actions": {"harder", "new_skill"},
    },
    {
        "verdict": "good",
        "confidence": "high",
        "missing_concepts": "None significant.",
        "allowed_actions": {"harder", "new_skill"},
    },
    {
        "verdict": "partial",
        "confidence": "high",
        "missing_concepts": "Confuses precision with recall.",
        "allowed_actions": {"probe", "coach", "easier"},
    },
    {
        "verdict": "poor",
        "confidence": "high",
        "missing_concepts": "Does not understand the basic concept.",
        "allowed_actions": {"coach", "easier", "probe"},
    },
    {
        "verdict": "partial",
        "confidence": "medium",
        "missing_concepts": "Incomplete explanation of overfitting.",
        "allowed_actions": {"probe", "coach"},
    },
]
