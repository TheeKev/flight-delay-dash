from pathlib import Path
import pandas as pd

# Folder containing the full merged yearly CSV files
DATA_DIR = Path("/Volumes/LaCie/CSE 6242/Project merged data_2015-2024")

# Desired sample size per file
SAMPLE_SIZE = 50000

# Random seed for reproducibility
RANDOM_STATE = 42


def create_50k_samples(data_dir: Path, sample_size: int = 50000, random_state: int = 42):
    """
    For each MERGED_*.csv file in the folder:
    - read the CSV
    - randomly sample 50,000 rows (or all rows if file has fewer than 50,000)
    - save a new CSV in the same folder with '_50k_sample' appended
    """
    csv_files = sorted(data_dir.glob("MERGED_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No files matching MERGED_*.csv found in: {data_dir}")

    print("Found files:")
    for f in csv_files:
        print(f" - {f.name}")

    for f in csv_files:
        print(f"\nReading {f.name} ...")

    if not csv_files:
        raise FileNotFoundError(f"No files matching MERGED_*.csv found in: {data_dir}")

    print("Found files:")
    for f in csv_files:
        print(f" - {f.name}")

    for f in csv_files:
        print(f"\nReading {f.name} ...")

        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines="skip")
            n_rows = len(df)

            if n_rows == 0:
                print(f"Skipping {f.name}: file is empty.")
                continue

            actual_sample_size = min(sample_size, n_rows)

            print(f"Total rows: {n_rows:,}")
            print(f"Sampling : {actual_sample_size:,}")

            sampled_df = df.sample(
                n=actual_sample_size,
                random_state=random_state,
                replace=False
            )

            output_name = f"{f.stem}_50k_sample.csv"
            output_path = data_dir / output_name

            sampled_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path.name}")

        except Exception as e:
            print(f"Failed on {f.name}: {e}")


if __name__ == "__main__":
    create_50k_samples(
        data_dir=DATA_DIR,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )