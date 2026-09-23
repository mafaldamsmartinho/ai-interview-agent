import time

from langchain_core.exceptions import OutputParserException

from models import ModelProvider, get_model
from prompts.prompts import evaluation_prompt
from router import route_model
from schemas import Evaluation

TEST_CASES = [
    {
        "question": "What is cross-validation and why is it useful?",
        "answer": (
            "Cross-validation splits the dataset into several folds. "
            "The model trains on some folds and validates on another, "
            "repeating the process to get a more reliable estimate "
            "of model performance."
        ),
    },
    {
        "question": "What is overfitting in machine learning?",
        "answer": (
            "Overfitting happens when a model learns the training data "
            "too closely and performs poorly on unseen data."
        ),
    },
    {
        "question": "What is the purpose of attention in a transformer?",
        "answer": (
            "Attention allows the model to determine which tokens are "
            "important when processing another token."
        ),
    },
    # Partially correct answer - swapped definitions
    {
        "question": "What is the difference between precision and recall?",
        "answer": (
            "Precision measures how many actual positives were found, "
            "while recall measures how many predicted positives were correct."
        ),
    },
    # Ambiguous question
    {
        "question": "Why might a model have high validation accuracy but still fail in production?",
        "answer": (
            "Maybe because the validation set was too easy, "
            "or the real-world data changed. It could also be "
            "overfitting, but I'm not sure."
        ),
    }
]


def benchmark_router():
    print("\n--- ROUTER ---")

    for i, case in enumerate(TEST_CASES, start=1):
        start = time.perf_counter()

        decision = route_model(
            question=case["question"],
            answer=case["answer"],
        )

        latency = time.perf_counter() - start

        print(f"\nTest {i}")
        print(f"Selected model: {decision.model.value}")
        print(f"Reason: {decision.reason}")
        print(f"Routing latency: {latency:.2f}s")


def benchmark_model(provider: ModelProvider):
    llm = get_model(provider)

    structured_llm = llm.with_structured_output(Evaluation)
    evaluation_chain = evaluation_prompt | structured_llm

    print(f"\n--- {provider.value.upper()} ---")

    for i, case in enumerate(TEST_CASES, start=1):
        start = time.perf_counter()

        try:
            response = evaluation_chain.invoke(
                {
                    "question": case["question"],
                    "answer": case["answer"],
                }
            )

            latency = time.perf_counter() - start

            print(f"\nTest {i}")
            print(f"Latency: {latency:.2f}s")
            print(f"Score: {response.score}/20")
            print(f"Correctness: {response.correctness}")
            print(f"Clarity: {response.clarity}")

        except OutputParserException as error:
            print(f"\nTest {i}")
            print(f"FAILED: {error}")


def main():
    benchmark_router()

    benchmark_model(ModelProvider.FAST)
    benchmark_model(ModelProvider.STRONG)


if __name__ == "__main__":
    main()
