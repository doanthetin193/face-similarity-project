"""
utils.py -- Cac ham tien ich dung chung cho toan bo pipeline
"""

import sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend khong can GUI (safe khi import)
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Đường dẫn chuẩn
# ──────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(ROOT_DIR, "embeddings")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

for d in [EMBED_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

EMBED_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(EMBED_DIR, "labels.npy")
IMAGES_PATH = os.path.join(EMBED_DIR, "images.npy")


# ──────────────────────────────────────────────
# 2. Lưu / load embeddings
# ──────────────────────────────────────────────
def save_embeddings(embeddings: np.ndarray,
                    labels: np.ndarray,
                    images: np.ndarray = None):
    """Lưu embedding + nhãn (+ ảnh tuỳ chọn) ra file .npy"""
    np.save(EMBED_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    if images is not None:
        np.save(IMAGES_PATH, images)
    print(f"[OK] Da luu {len(embeddings)} embeddings -> {EMBED_DIR}")


def load_embeddings(with_images: bool = False):
    """
    Load embeddings đã trích sẵn.
    Returns: (embeddings, labels) hoặc (embeddings, labels, images)
    """
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(
            "Chưa có file embeddings. Hãy chạy 02_embed.py trước."
        )
    embeddings = np.load(EMBED_PATH, allow_pickle=False)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    print(f"[OK] Da load {len(embeddings)} embeddings - {embeddings.shape[1]}D")
    if with_images and os.path.exists(IMAGES_PATH):
        images = np.load(IMAGES_PATH, allow_pickle=False)
        return embeddings, labels, images
    return embeddings, labels


# ──────────────────────────────────────────────
# 3. Lưu figure
# ──────────────────────────────────────────────
def save_figure(fig: plt.Figure, filename: str):
    """Lưu matplotlib figure vào thư mục results/"""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Da luu hinh -> {path}")
    plt.close(fig)
