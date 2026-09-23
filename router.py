from langchain_core.prompts import ChatPromptTemplate

from models import get_router_model
from schemas import RoutingDecision

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a model router.

Choose which model should evaluate the candidate's answer.

Choose FAST when:
- the answer is clearly correct or clearly incorrect
- the question tests a simple, well-defined concept
- evaluation requires little interpretation

Choose STRONG when:
- the answer is partially correct
- it contains mixed correct and incorrect claims
- the candidate expresses uncertainty
- the answer is ambiguous
- correctness depends on nuanced technical reasoning
- distinguishing a minor omission from a conceptual error is difficult

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
Question:
{question}

Candidate answer:
{answer}
""",
        ),
    ]
)


def route_model(question: str, answer: str) -> RoutingDecision:
    model = get_router_model().with_structured_output(RoutingDecision)

    chain = router_prompt | model

    return chain.invoke(
        {
            "question": question,
            "answer": answer,
        }
    )
