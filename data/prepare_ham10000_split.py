import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def copy_images(df: pd.DataFrame, src_dirs: list[Path], split_dir: Path) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["dx"]
        filename = f"{image_id}.jpg"

        src_path = None
        for src_dir in src_dirs:
            candidate = src_dir / filename
            if candidate.exists():
                src_path = candidate
                break

        if src_path is None:
            print(f"[WARN] Missing image file: {filename}")
            continue

        class_dir = split_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, class_dir / filename)
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--images_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    src_dirs = [Path(p) for p in args.images_dirs]
    output_root = Path(args.output_root)

    temp_size = args.val_size + args.test_size
    if temp_size <= 0 or temp_size >= 1:
        raise ValueError("val_size + test_size must be > 0 and < 1")

    df = pd.read_csv(metadata_csv)
    df = df[df["dx"].isin(EXPECTED_CLASSES)].copy()

    train_df, temp_df = train_test_split(
        df, test_size=temp_size, random_state=args.seed, stratify=df["dx"]
    )
    test_ratio_in_temp = args.test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_in_temp,
        random_state=args.seed,
        stratify=temp_df["dx"],
    )

    train_count = copy_images(train_df, src_dirs, output_root / "train")
    val_count = copy_images(val_df, src_dirs, output_root / "val")
    test_count = copy_images(test_df, src_dirs, output_root / "test")

    print(
        f"Prepared dataset at: {output_root}\n"
        f"Total rows: {len(df)}\n"
        f"Train: {len(train_df)} (copied {train_count})\n"
        f"Val:   {len(val_df)} (copied {val_count})\n"
        f"Test:  {len(test_df)} (copied {test_count})"
    )


if __name__ == "__main__":
    main()
