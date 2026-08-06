"""Loading grasp datasets, wherever they came from."""

import glob
import os

from datasets import Dataset, load_dataset, load_from_disk


def load_grasp_dataset(path, split="train"):
    """Load a grasp dataset from any of the layouts this repo produces or consumes.

    Accepts:
      * a `.parquet` file
      * a `save_to_disk` directory (what every script here writes)
      * a directory holding one or more `.parquet` files (older `grasp_rrt_expand.py`
        output wrote a bare parquet into --output_dir)
      * a Hugging Face Hub dataset id
    """
    path = str(path)

    if path.endswith('.parquet'):
        return load_dataset("parquet", data_files=path, split=split)

    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "dataset_info.json")) or \
           os.path.exists(os.path.join(path, "state.json")):
            dataset = load_from_disk(path)
            return dataset[split] if not isinstance(dataset, Dataset) else dataset

        parquets = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if parquets:
            return load_dataset("parquet", data_files=parquets, split=split)

        raise FileNotFoundError(
            f"{path} is a directory but holds neither a save_to_disk dataset nor any "
            f".parquet files."
        )

    return load_dataset(path, split=split)
