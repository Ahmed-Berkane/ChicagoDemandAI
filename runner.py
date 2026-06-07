"""Train the best demand model and report test metrics."""

from src.modeling.train import run_training
from src.prepare import prepare_datasets


def main() -> None:
    print("=== Step 1: Prepare datasets (split + feature selection) ===", flush=True)
    prepare_datasets()

    print("\n=== Step 2: Train models and score ===", flush=True)
    run_training()


if __name__ == "__main__":
    main()
