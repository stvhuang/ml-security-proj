import pandas as pd
from datasets import Dataset


def load_dataset(dataset: str = "ChnSentiCorp", split: str = "train"):
    df = pd.read_csv(f"/data/{dataset}_{split}.tsv", sep="\t")
    return Dataset.from_pandas(df)
