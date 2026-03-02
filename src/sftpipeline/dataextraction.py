"""Create a 5k balanced VLDBench dataset with a 60/40 bias ratio."""

import random

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "vector-institute/VLDBench"
SPLIT = "train"
TOTAL_SAMPLES = 5000
BIAS_RATIO = 0.6
RANDOM_SEED = 42

OUTPUT_FILE = "VLDBench_5k_60_40_balanced.csv"


def load_vldbench() -> pd.DataFrame:
    """Load the VLDBench dataset split as a pandas DataFrame."""
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    return dataset.to_pandas()


def create_balanced_split(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 60/40 biased/unbiased balanced subset of size TOTAL_SAMPLES."""
    num_biased = int(TOTAL_SAMPLES * BIAS_RATIO)
    num_unbiased = TOTAL_SAMPLES - num_biased

    likely_df = df[df["text_label"] == "Likely"]
    unlikely_df = df[df["text_label"] == "Unlikely"]

    biased_sample = likely_df.sample(n=num_biased, random_state=RANDOM_SEED)
    unbiased_sample = unlikely_df.sample(n=num_unbiased, random_state=RANDOM_SEED)

    combined_df = pd.concat([biased_sample, unbiased_sample])
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(
        drop=True
    )

    label_map = {"Likely": "biased", "Unlikely": "unbiased"}
    combined_df["classification_label"] = combined_df["text_label"].map(label_map)

    return combined_df[["article_text", "classification_label"]]


def preview_samples(df: pd.DataFrame) -> None:
    """Print a preview of the first five biased and unbiased samples."""
    print("\n========== FIRST 5 BIASED ==========\n")
    biased_samples = df[df["classification_label"] == "biased"].head(5)

    for idx, row in biased_samples.iterrows():
        print(f"\n--- Sample {idx} ---")
        print("Label:", row["classification_label"])
        print("Article Text:\n")
        print(row["article_text"])
        print("\n" + "=" * 80)

    print("\n========== FIRST 5 UNBIASED ==========\n")
    unbiased_samples = df[df["classification_label"] == "unbiased"].head(5)

    for idx, row in unbiased_samples.iterrows():
        print(f"\n--- Sample {idx} ---")
        print("Label:", row["classification_label"])
        print("Article Text:\n")
        print(row["article_text"])
        print("\n" + "=" * 80)


def main() -> None:
    """Run dataset creation pipeline and save balanced CSV."""
    random.seed(RANDOM_SEED)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    df = load_vldbench()

    print("Total rows:", len(df))
    print(df["text_label"].value_counts())

    final_df = create_balanced_split(df)

    print("Final dataset size:", len(final_df))
    print("Updated label distribution:")
    print(final_df["classification_label"].value_counts())

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved as {OUTPUT_FILE}")

    preview_samples(final_df)


if __name__ == "__main__":
    main()
