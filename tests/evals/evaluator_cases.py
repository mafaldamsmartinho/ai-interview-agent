EVALUATOR_CASES = [
    {
        "question": "What is overfitting?",
        "answer": (
            "Overfitting happens when a model learns the training data"
            "too closely "
            "and performs poorly on unseen data."
        ),
        "expected": "excellent",
    },
    {
        "question": "What is precision?",
        "answer": (
            "Precision is the proportion of predicted positives "
            "that are actually positive."
        ),
        "expected": "excellent",
    },
    {
        "question": "What is recall?",
        "answer": "Recall measures how many predicted positives are correct.",
        "expected": "poor",
    },
    {
        "question": "Why do we use cross-validation?",
        "answer": (
            "To evaluate performance across multiple data splits rather "
            "than depending on one train-validation split."
        ),
        "expected": "excellent",
    },
    {
        "question": "What is data leakage?",
        "answer": "It means the dataset contains missing values.",
        "expected": "poor",
    },
]
