import time

from models import ModelProvider, get_model
from prompts.prompts import evaluation_prompt


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
]


def benchmark_model(provider: ModelProvider):
    llm = get_model(provider)

    evaluation_chain = evaluation_prompt | llm

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
            print("Response:")
            print(response.content)

        except Exception as error:
            print(f"\nTest {i}")
            print(f"FAILED: {error}")


def main():
    benchmark_model(ModelProvider.QWEN)
    benchmark_model(ModelProvider.LLAMA)


if __name__ == "__main__":
    main()