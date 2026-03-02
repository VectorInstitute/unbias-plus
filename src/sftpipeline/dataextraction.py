from datasets import load_dataset
import pandas as pd
import random

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)


DATASET_NAME = "vector-institute/VLDBench"
SPLIT = "train"
TOTAL_SAMPLES = 5000
BIAS_RATIO = 0.6  
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


NUM_BIASED = int(TOTAL_SAMPLES * BIAS_RATIO)      # 3000
NUM_UNBIASED = TOTAL_SAMPLES - NUM_BIASED         # 2000


dataset = load_dataset(DATASET_NAME, split=SPLIT)


df = dataset.to_pandas()

print("Total rows:", len(df))
print(df["text_label"].value_counts())


likely_df = df[df["text_label"] == "Likely"]
unlikely_df = df[df["text_label"] == "Unlikely"]

print("Likely count:", len(likely_df))
print("Unlikely count:", len(unlikely_df))


biased_sample = likely_df.sample(n=NUM_BIASED, random_state=RANDOM_SEED)
unbiased_sample = unlikely_df.sample(n=NUM_UNBIASED, random_state=RANDOM_SEED)


combined_df = pd.concat([biased_sample, unbiased_sample])


combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print("Final dataset size:", len(combined_df))
print(combined_df["text_label"].value_counts())


label_map = {
    "Likely": "biased",
    "Unlikely": "unbiased"
}

combined_df["classification_label"] = combined_df["text_label"].map(label_map)


final_df = combined_df[["article_text", "classification_label"]]

print("Updated label distribution:")
print(final_df["classification_label"].value_counts())


final_df.to_csv("VLDBench_5k_60_40_balanced.csv", index=False)

print("Saved as VLDBench_5k_60_40_balanced.csv")



print("\n========== FIRST 5 BIASED ==========\n")

biased_samples = final_df[final_df["classification_label"] == "biased"].head(5)

for i, row in biased_samples.iterrows():
    print(f"\n--- Sample {i} ---")
    print("Label:", row["classification_label"])
    print("Article Text:\n")
    print(row["article_text"])
    print("\n" + "="*80)

print("\n========== FIRST 5 UNBIASED ==========\n")

unbiased_samples = final_df[final_df["classification_label"] == "unbiased"].head(5)

for i, row in unbiased_samples.iterrows():
    print(f"\n--- Sample {i} ---")
    print("Label:", row["classification_label"])
    print("Article Text:\n")
    print(row["article_text"])
    print("\n" + "="*80)