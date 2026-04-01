"""
01_preprocess.py — Bước 1: Load LFW dataset và tiền xử lý ảnh

Chức năng:
- Tải LFW dataset tự động qua scikit-learn
- Chuẩn hoá ảnh về dải [0, 1]
- Lưu thông tin cơ bản về dataset
- In thống kê dataset để kiểm tra
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Thêm thư mục src vào path để import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RESULTS_DIR, save_figure

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
MIN_FACES_PER_PERSON = 20   # Chỉ lấy người có ≥ 20 ảnh
RESIZE_FACTOR = 0.5          # Resize ảnh để giảm dung lượng


def load_lfw(min_faces: int = MIN_FACES_PER_PERSON,
             resize: float = RESIZE_FACTOR,
             color: bool = True):
    """
    Load LFW dataset từ scikit-learn (tự tải nếu chưa có).

    Returns:
        images  : np.ndarray shape (N, H, W, 3), dtype float32, range [0,1]
        labels  : np.ndarray shape (N,), dtype str  — tên người
        target_names: list of unique person names
    """
    print("=" * 55)
    print("  BƯỚC 1 — LOAD & PREPROCESS LFW DATASET")
    print("=" * 55)
    print(f"[*] Tải LFW (min_faces_per_person={min_faces}, color={color}) …")
    print("[*] Lần đầu sẽ tải ~168MB — vui lòng chờ …\n")

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=resize,
        color=color,         # Ảnh màu RGB
        download_if_missing=True,
    )

    # Ảnh gốc từ sklearn: shape (N, H, W, 3), color=True → float32 [0, 1] luôn
    images = lfw.images.astype("float32")

    # Label dạng integer → chuyển về tên người
    labels = np.array([lfw.target_names[t] for t in lfw.target])

    return images, labels, list(lfw.target_names)


def print_stats(images: np.ndarray,
                labels: np.ndarray,
                target_names: list):
    """In thống kê dataset."""
    n_samples, h, w, c = images.shape
    n_classes = len(target_names)

    print(f"  Tổng số ảnh    : {n_samples:,}")
    print(f"  Số người       : {n_classes}")
    print(f"  Kích thước ảnh : {h} × {w} × {c}  (H × W × C)")
    print(f"  Dtype / Range  : {images.dtype}  [{images.min():.2f}, {images.max():.2f}]")
    print()

    # Top-10 người có nhiều ảnh nhất
    unique, counts = np.unique(labels, return_counts=True)
    idx = np.argsort(counts)[::-1]
    print("  Top-10 người có nhiều ảnh nhất:")
    for i in idx[:10]:
        print(f"    {unique[i]:<30s}: {counts[i]:>4d} ảnh")
    print()


def plot_sample_grid(images: np.ndarray,
                     labels: np.ndarray,
                     n: int = 16):
    """Hiển thị lưới n ảnh mẫu và lưu kết quả."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("LFW — Sample Faces", fontsize=14, y=1.01)

    indices = np.random.choice(len(images), n, replace=False)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(images[idx])
        ax.set_title(labels[idx].split()[-1], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, "01_sample_faces.png")
    print("[✓] Đã lưu ảnh mẫu → results/01_sample_faces.png")


def plot_class_distribution(labels: np.ndarray):
    """Vẽ biểu đồ phân phối số ảnh theo người."""
    unique, counts = np.unique(labels, return_counts=True)
    idx = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(unique)), counts[idx], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(
        [n.split()[-1] for n in unique[idx]],
        rotation=90, fontsize=7
    )
    ax.set_xlabel("Người")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân phối số ảnh theo người — LFW dataset")
    plt.tight_layout()
    save_figure(fig, "01_class_distribution.png")
    print("[✓] Đã lưu biểu đồ phân phối → results/01_class_distribution.png")


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    images, labels, target_names = load_lfw()

    print("\n📊 THỐNG KÊ DATASET:")
    print_stats(images, labels, target_names)

    plot_sample_grid(images, labels)
    plot_class_distribution(labels)

    print("\n[✓] Preprocess hoàn tất. Chạy tiếp: python src/02_embed.py")
