"""
04_cluster.py — Bước 3B: Clustering trong không gian Embedding

Chức năng:
- KMeans clustering trên 512-dim embeddings
- Elbow Method để tìm số cụm tối ưu
- Silhouette Score để đánh giá chất lượng clustering
- Visualize ảnh mẫu của mỗi cụm

Ý nghĩa học thuật:
  Phân cụm trong không gian embedding cho thấy FaceNet đã học
  được biểu diễn có cấu trúc thực sự trong không gian 512D.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_embeddings, save_figure

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
K_RANGE = range(3, 16)    # Thử các giá trị k từ 3 đến 15
K_BEST = 8                # K mặc định để demo (hoặc từ elbow)
RANDOM_STATE = 42


def elbow_method(embeddings: np.ndarray,
                 k_range=K_RANGE) -> int:
    """
    Tìm số cụm tối ưu bằng Elbow Method.

    Vẽ đồ thị inertia (within-cluster sum of squares) theo k.
    Điểm "khuỷu" (elbow) là k tối ưu.

    Returns: k_optimal (gợi ý — người dùng tự quan sát)
    """
    print("[*] Elbow Method — thử k từ", k_range.start, "đến", k_range.stop - 1)
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        cluster_labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        sil = silhouette_score(embeddings, cluster_labels, sample_size=1000,
                               random_state=RANDOM_STATE)
        sil_scores.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    # Vẽ Elbow Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=6)
    ax1.set_xlabel("Số cụm (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method")
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), sil_scores, "rs-", linewidth=2, markersize=6)
    ax2.set_xlabel("Số cụm (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score theo k")
    ax2.grid(alpha=0.3)

    plt.suptitle("Chọn Số Cụm Tối Ưu", fontsize=13)
    plt.tight_layout()
    save_figure(fig, "04_elbow_silhouette.png")
    print("[✓] Đã lưu → results/04_elbow_silhouette.png\n")

    # Gợi ý k tốt nhất theo silhouette
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"[✓] Silhouette gợi ý k = {best_k}  (sil={max(sil_scores):.4f})")
    return best_k


def run_kmeans(embeddings: np.ndarray, k: int):
    """
    Chạy KMeans với k cụm.
    Returns: cluster_labels (N,), kmeans model
    """
    print(f"\n[*] KMeans clustering — k = {k} …")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    cluster_labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, cluster_labels,
                           sample_size=1000, random_state=RANDOM_STATE)
    print(f"[✓] KMeans (k={k}): Silhouette Score = {sil:.4f}")
    return cluster_labels, km


def plot_cluster_samples(cluster_labels: np.ndarray,
                          images: np.ndarray,
                          labels: np.ndarray,
                          k: int,
                          n_per_cluster: int = 5):
    """Vẽ lưới ảnh mẫu cho mỗi cụm."""
    fig, axes = plt.subplots(k, n_per_cluster,
                              figsize=(n_per_cluster * 2, k * 2.2))
    fig.suptitle(f"KMeans Clusters (k={k}) — Ảnh Mẫu Mỗi Cụm",
                 fontsize=12, y=1.01)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for c in range(k):
        idxs = np.where(cluster_labels == c)[0]
        sample = np.random.choice(idxs, min(n_per_cluster, len(idxs)),
                                   replace=False)
        for j, idx in enumerate(sample):
            ax = axes[c][j]
            ax.imshow(images[idx])
            ax.set_title(labels[idx].split()[-1], fontsize=6)
            ax.axis("off")
            # Viền màu theo cụm
            for sp in ax.spines.values():
                sp.set_edgecolor(colors[c])
                sp.set_linewidth(2)

        # Dán nhãn cụm bên trái
        axes[c][0].set_ylabel(f"Cụm {c}", rotation=90, fontsize=8,
                               color=colors[c], fontweight="bold")

    plt.tight_layout()
    save_figure(fig, f"04_cluster_samples_k{k}.png")
    print(f"[✓] Đã lưu → results/04_cluster_samples_k{k}.png")


def analyse_clusters(cluster_labels: np.ndarray,
                      labels: np.ndarray,
                      k: int):
    """In top-3 người có nhiều ảnh nhất trong mỗi cụm."""
    print(f"\n📊 PHÂN TÍCH NỘI DUNG CỤM (k={k}):")
    for c in range(k):
        idxs = np.where(cluster_labels == c)[0]
        people = labels[idxs]
        unique, counts = np.unique(people, return_counts=True)
        top = np.argsort(counts)[::-1][:3]
        top_str = ", ".join(
            f"{unique[i].split()[-1]}({counts[i]})" for i in top
        )
        print(f"  Cụm {c:2d} [{len(idxs):4d} ảnh]: {top_str}")


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BƯỚC 3B — CLUSTERING TRONG EMBEDDING SPACE")
    print("=" * 55 + "\n")

    embeddings, labels, images = load_embeddings(with_images=True)

    # 1. Tìm k tối ưu
    best_k = elbow_method(embeddings)
    k_use = best_k  # hoặc thay bằng K_BEST nếu muốn cố định

    # 2. Cluster cuối cùng
    cluster_labels, km_model = run_kmeans(embeddings, k=k_use)

    # 3. Phân tích & visualize
    analyse_clusters(cluster_labels, labels, k=k_use)
    plot_cluster_samples(cluster_labels, images, labels, k=k_use)

    # Lưu cluster labels để dùng ở bước visualize
    import os
    from utils import EMBED_DIR
    np.save(os.path.join(EMBED_DIR, "cluster_labels.npy"), cluster_labels)
    print(f"\n[✓] Đã lưu cluster labels → embeddings/cluster_labels.npy")
    print("[✓] Chạy tiếp: python src/05_visualize.py")
