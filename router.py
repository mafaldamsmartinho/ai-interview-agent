from langchain_core.prompts import ChatPromptTemplate

from models import get_router_model
from prompts.prompts import load_prompt
from schemas import RoutingDecision

router_system_prompt = load_prompt("router_prompts/v1.1.0.txt")
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            router_system_prompt,
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
