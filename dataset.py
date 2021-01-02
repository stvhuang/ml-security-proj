import pandas as pd
from datasets import Dataset, DatasetInfo


def load_dataset(dataset: str = "ChnSentiCorp", split: str = "train"):
    df = pd.read_csv(f"/data/{dataset}_{split}.tsv", sep="\t")
    ds = Dataset.from_pandas(df)
    ds.features["label"].num_classes = 2
    ds.features["label"].names = ["pos", "neg"]

    return ds
