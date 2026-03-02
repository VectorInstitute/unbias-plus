from datasets import load_dataset
import pandas as pd
import random

pd.set_option("display.max_colwidth", None)

DATASET_NAME = "vector-institute/VLDBench"
SPLIT = "train"

TRAIN_FILE = "VLDBench_5k_60_40_balanced.csv"
TEST_OUTPUT_FILE = "VLDBench_test_50_biased.csv"

NUM_TEST_SAMPLES = 50
RANDOM_SEED = 123

random.seed(RANDOM_SEED)

dataset = load_dataset(DATASET_NAME, split=SPLIT)
df_full = dataset.to_pandas()

print("Total dataset size:", len(df_full))

train_df = pd.read_csv(TRAIN_FILE)

print("Training set size:", len(train_df))


remaining_df = df_full[
    ~df_full["article_text"].isin(train_df["article_text"])
]

print("Remaining after removing 5k:", len(remaining_df))


remaining_biased = remaining_df[
    remaining_df["text_label"] == "Likely"
]

print("Remaining biased samples:", len(remaining_biased))


test_sample = remaining_biased.sample(
    n=NUM_TEST_SAMPLES,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Rename label
test_sample["classification_label"] = "biased"

# Keep only required columns
final_test_df = test_sample[["article_text", "classification_label"]]

print("Final test size:", len(final_test_df))
print(final_test_df["classification_label"].value_counts())

final_test_df.to_csv(TEST_OUTPUT_FILE, index=False)

print(f"Saved as {TEST_OUTPUT_FILE}")