import argparse
import os
from pathlib import Path

from config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Link a BraTS2021 training data directory into this project.")
    parser.add_argument("source", help="Path containing BraTS2021 patient directories, or its parent.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if (source / "BraTS2021_Training_Data").is_dir():
        source = source / "BraTS2021_Training_Data"
    if not source.is_dir():
        raise SystemExit(f"Data directory not found: {source}")
    patient_dirs = [p for p in source.iterdir() if p.is_dir() and p.name.startswith("BraTS2021")]
    if not patient_dirs:
        raise SystemExit(f"No BraTS2021 patient directories found in: {source}")

    target_parent = PROJECT_ROOT / "data"
    target_parent.mkdir(exist_ok=True)
    target = target_parent / "BraTS2021_Training_Data"
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            print(f"Already linked: {target} -> {source}")
            return
        raise SystemExit(f"Target already exists: {target}")
    os.symlink(source, target)
    print(f"Linked: {target} -> {source}")
    print(f"Detected patients: {len(patient_dirs)}")


if __name__ == "__main__":
    main()
