"""
run_custom_pipeline.py -- Run the custom Vietnamese dataset pipeline safely.

Default mode does NOT re-embed the dataset. It only runs the analysis steps
that consume existing files in embeddings/:

    python run_custom_pipeline.py

To intentionally rebuild embeddings from custom_dataset/ first:

    python run_custom_pipeline.py --with-embed

This file exists to avoid accidentally running run_pipeline.py, which is for
the LFW auxiliary pipeline and can overwrite the current custom embeddings.
"""

import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
EMBED_DIR = os.path.join(ROOT, "embeddings")

EMBED_FILES = [
    os.path.join(EMBED_DIR, "embeddings.npy"),
    os.path.join(EMBED_DIR, "labels.npy"),
    os.path.join(EMBED_DIR, "images.npy"),
]

EMBED_STEP = (
    "Step 2B - Embed custom Vietnamese dataset",
    os.path.join(SRC, "02b_embed_custom.py"),
)

ANALYSIS_STEPS = [
    ("Step 3A - Top-K retrieval, similarity distribution, ROC/AUC/EER",
     os.path.join(SRC, "03_retrieval.py")),
    ("Step 3B - KMeans clustering, Elbow, Silhouette",
     os.path.join(SRC, "04_cluster.py")),
    ("Step 4 - PCA, t-SNE, similarity heatmap",
     os.path.join(SRC, "05_visualize.py")),
]


def print_box(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def missing_embedding_files():
    return [path for path in EMBED_FILES if not os.path.exists(path)]


def run_step(name: str, script: str):
    print_box(name)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}: {script}")
        sys.exit(result.returncode)
    print(f"\n[OK] Finished in {elapsed:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the custom dataset pipeline. By default this skips embedding "
            "and only runs 03_retrieval.py, 04_cluster.py, 05_visualize.py."
        )
    )
    parser.add_argument(
        "--with-embed",
        action="store_true",
        help=(
            "Also run src/02b_embed_custom.py first. This can take a long time "
            "and overwrites embeddings/*.npy."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps that would run, without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    steps = []

    if args.with_embed:
        steps.append(EMBED_STEP)
    else:
        missing = missing_embedding_files()
        if missing:
            print_box("Custom embeddings are missing")
            for path in missing:
                print(f"[MISSING] {os.path.relpath(path, ROOT)}")
            print("\nRun this only if you really want to rebuild embeddings:")
            print("  python run_custom_pipeline.py --with-embed")
            sys.exit(1)

    steps.extend(ANALYSIS_STEPS)

    print_box("CUSTOM DATASET PIPELINE")
    if args.with_embed:
        print("[MODE] with embed: will rebuild custom embeddings first.")
        print("[WARN] This can take a long time and overwrites embeddings/*.npy.")
    else:
        print("[MODE] skip embed: using existing custom embeddings.")
        print("[SAFE] This mode does not touch src/02b_embed_custom.py.")
    print()
    for _, script in steps:
        print(f" - {os.path.relpath(script, ROOT)}")

    if args.dry_run:
        print("\n[DRY RUN] No step executed.")
        return

    total_t0 = time.time()
    for name, script in steps:
        run_step(name, script)

    print_box("CUSTOM PIPELINE FINISHED")
    print(f"Total time: {time.time() - total_t0:.1f}s")
    print("Outputs: results/ and embeddings/cluster_labels.npy")


if __name__ == "__main__":
    main()
