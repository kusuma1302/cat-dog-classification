"""
dataset.py
----------
Loads the cat-vs-dog dataset described by data/dataset_info.csv.

The CSV has columns: filename, split, class, label, source
We use `split` (train / validation / test) and `label` (0=cat, 1=dog).

Image resolution strategy
--------------------------
Your images might live in any of a few common layouts. This dataset class
checks, for each row, the following candidate locations (in order) under
`images_root` (default: data/images) and uses the first one that exists:

  1. images_root/<filename>                      e.g. data/images/cat_train_00000.jpg
  2. images_root/<split>/<class>/<filename>       e.g. data/images/train/cats/cat_train_00000.jpg
  3. images_root/<class>/<filename>               e.g. data/images/cats/cat_train_00000.jpg
  4. images_root/<split>/<filename>               e.g. data/images/train/cat_train_00000.jpg
  5. images_root/<source>                          e.g. data/images/5459.jpg  (original filename)
  6. images_root/<class>/<source>                 e.g. data/images/cats/5459.jpg

If none of these exist, the row is skipped and reported at load time so you
know exactly which layout to fix.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CANDIDATE_TEMPLATES = [
    "{filename}",
    "{split}/{class_}/{filename}",
    "{class_}/{filename}",
    "{split}/{filename}",
    "{source}",
    "{class_}/{source}",
]


def resolve_image_path(images_root, row):
    """Try known folder layouts and return the first path that exists."""
    has_source = "source" in row.index and isinstance(row.get("source"), str)
    for template in CANDIDATE_TEMPLATES:
        if "{source}" in template and not has_source:
            continue
        rel = template.format(
            filename=row["filename"],
            split=row["split"],
            class_=row["class"],
            source=row["source"] if has_source else "",
        )
        candidate = os.path.join(images_root, rel)
        if os.path.isfile(candidate):
            return candidate
    return None


class CatDogDataset(Dataset):
    def __init__(self, csv_path, images_root, split, transform=None, verbose=True):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        resolved_paths = []
        missing = 0
        for _, row in df.iterrows():
            path = resolve_image_path(images_root, row)
            if path is None:
                missing += 1
            resolved_paths.append(path)

        df["resolved_path"] = resolved_paths
        found_df = df[df["resolved_path"].notna()].reset_index(drop=True)

        if verbose:
            print(f"[{split}] {len(found_df)}/{len(df)} images found under '{images_root}'.")
            if missing:
                print(
                    f"[{split}] WARNING: {missing} image(s) could not be located. "
                    f"See README.md 'Getting the images' section for expected folder layouts."
                )

        if len(found_df) == 0:
            raise FileNotFoundError(
                f"No images found for split='{split}' under '{images_root}'. "
                f"Check README.md for the expected folder structure."
            )

        self.df = found_df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["resolved_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label
