"""
run_pipeline.py — Chạy toàn bộ pipeline LFW từ đầu đến cuối (1 lệnh)

CANH BAO:
    File nay chi danh cho LFW. Neu chay file nay, src/02_embed.py se ghi de
    embeddings/*.npy hien tai. Voi custom dataset, dung run_custom_pipeline.py.

Sử dụng:
    python run_pipeline.py                  # Chạy pipeline LFW đầy đủ

Nếu dùng custom dataset (Vietnamese Celebrity...):
    python run_custom_pipeline.py           # Da co embeddings, chi chay 03/04/05
    python run_custom_pipeline.py --with-embed  # Co y embed lai custom_dataset/

Đưa ảnh từ ngoài vào query:
    python query_external.py "anh.jpg"      # Tìm người giống nhất

Thứ tự pipeline LFW:
    1. Preprocess  → xem dataset, lưu biểu đồ
    2. Embed       → resize/normalize LFW crop + FaceNet → embeddings.npy
    3. Retrieval   → Top-K search + đánh giá threshold
    4. Cluster     → Elbow + KMeans + Silhouette
    5. Visualize   → PCA + t-SNE + Heatmap

Tất cả hình ảnh được lưu vào thư mục: results/
"""

import subprocess
import sys
import os
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

STEPS = [
    ("Bước 1 — Preprocess & Dataset Stats", os.path.join(SRC, "01_preprocess.py")),
    ("Bước 2 — Face Embedding Extraction (AI Core)", os.path.join(SRC, "02_embed.py")),
    ("Bước 3A — Top-K Face Retrieval", os.path.join(SRC, "03_retrieval.py")),
    ("Bước 3B — KMeans Clustering", os.path.join(SRC, "04_cluster.py")),
    ("Bước 4 — Visualization (PCA + t-SNE + Heatmap)", os.path.join(SRC, "05_visualize.py")),
]


def run_step(name: str, script: str):
    print("\n" + "═" * 60)
    print(f"  🚀  {name}")
    print("═" * 60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        check=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  Bước thất bại! (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n✅  Hoàn thành trong {elapsed:.1f}s")


if __name__ == "__main__":
    print("=" * 60)
    print("  FACE SIMILARITY RETRIEVAL SYSTEM — LFW AUXILIARY PIPELINE")
    print("  WARNING: This can overwrite current custom embeddings.")
    print("  For custom dataset, use: python run_custom_pipeline.py")
    print("=" * 60)

    total_t0 = time.time()
    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 60)
    print(f"  🎉  TOÀN BỘ PIPELINE HOÀN TẤT")
    print(f"  ⏱   Tổng thời gian: {time.time() - total_t0:.1f}s")
    print(f"  📁  Kết quả → results/")
    print("=" * 60)
