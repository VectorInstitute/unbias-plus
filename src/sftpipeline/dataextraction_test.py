"""Create a 50-sample biased test split from VLDBench excluding training data."""

import random

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "vector-institute/VLDBench"
SPLIT = "train"

TRAIN_FILE = "VLDBench_5k_60_40_balanced.csv"
TEST_OUTPUT_FILE = "VLDBench_test_50_biased.csv"

NUM_TEST_SAMPLES = 50
RANDOM_SEED = 123


def load_full_dataset() -> pd.DataFrame:
    """Load the full VLDBench dataset as a pandas DataFrame."""
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    return dataset.to_pandas()


def load_training_set() -> pd.DataFrame:
    """Load the previously generated training CSV file."""
    return pd.read_csv(TRAIN_FILE)


def create_test_split(df_full: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Create a test split of biased samples not present in the training set."""
    remaining_df = df_full[~df_full["article_text"].isin(train_df["article_text"])]

    remaining_biased = remaining_df[remaining_df["text_label"] == "Likely"]

    test_sample = remaining_biased.sample(
        n=NUM_TEST_SAMPLES,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    test_sample["classification_label"] = "biased"

    return test_sample[["article_text", "classification_label"]]


def main() -> None:
    """Generate test CSV of unseen biased samples."""
    random.seed(RANDOM_SEED)
    pd.set_option("display.max_colwidth", None)

    df_full = load_full_dataset()
    print("Total dataset size:", len(df_full))

    train_df = load_training_set()
    print("Training set size:", len(train_df))

    final_test_df = create_test_split(df_full, train_df)

    print("Final test size:", len(final_test_df))
    print(final_test_df["classification_label"].value_counts())

    final_test_df.to_csv(TEST_OUTPUT_FILE, index=False)
    print(f"Saved as {TEST_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
