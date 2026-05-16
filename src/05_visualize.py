"""
05_visualize.py — Bước 4: Giảm chiều & Visualize Embedding Space

Chức năng:
- PCA: giảm 512D → 2D (nhanh, tuyến tính)
- t-SNE: giảm 512D → 2D (chậm hơn, phi tuyến, đẹp hơn)
- Scatter plot màu theo: người / cluster
- Cosine similarity heatmap

Ý nghĩa: Kiểm tra xem embedding space có cấu trúc không —
  nếu ảnh cùng người tụ thành cụm → model hoạt động tốt.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_embeddings, save_figure, EMBED_DIR

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
RANDOM_STATE = 42
# Số người hiển thị màu (quá nhiều → khó phân biệt)
N_DISPLAY_CLASSES = 15


def reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Giảm chiều bằng PCA (nhanh, tuyến tính)."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[✓] PCA: {embeddings.shape[1]}D → {n_components}D  "
          f"(giải thích {explained:.1f}% phương sai)")
    return reduced


def reduce_tsne(embeddings: np.ndarray,
                perplexity: int = TSNE_PERPLEXITY,
                n_iter: int = TSNE_N_ITER) -> np.ndarray:
    """
    Giảm chiều bằng t-SNE (phi tuyến, trực quan hơn PCA).

    Lưu ý: t-SNE chậm trên toàn bộ dataset lớn.
    Với LFW ~3K ảnh, mất ~1-2 phút trên CPU.
    """
    print(f"[*] t-SNE đang chạy … (perplexity={perplexity}) — có thể mất 1-2 phút")
    # PCA xuống 50D trước để t-SNE nhanh hơn
    if embeddings.shape[1] > 50:
        embeddings = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(embeddings)
    # n_iter đổi tên thành max_iter từ sklearn 1.5+
    import sklearn
    tsne_kwargs = dict(n_components=2, perplexity=perplexity,
                       random_state=RANDOM_STATE, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter
    tsne = TSNE(**tsne_kwargs)
    reduced = tsne.fit_transform(embeddings)
    print(f"[✓] t-SNE hoàn tất → shape {reduced.shape}")
    return reduced


def plot_scatter(reduced_2d: np.ndarray,
                 labels: np.ndarray,
                 title: str,
                 save_name: str,
                 n_classes: int = N_DISPLAY_CLASSES):
    """
    Scatter plot 2D — màu theo từng người (hoặc cụm).
    Lọc N_DISPLAY_CLASSES người có nhiều ảnh nhất để dễ nhìn.
    """
    # Chọn N người có nhiều ảnh nhất
    unique, counts = np.unique(labels, return_counts=True)
    top_people = unique[np.argsort(counts)[::-1][:n_classes]]
    mask = np.isin(labels, top_people)

    colors = plt.cm.tab20(np.linspace(0, 1, len(top_people)))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        reduced_2d[~mask, 0], reduced_2d[~mask, 1],
        c="lightgray", s=5, alpha=0.3, label="Khác"
    )
    for i, person in enumerate(top_people):
        idx = np.where(labels == person)[0]
        ax.scatter(
            reduced_2d[idx, 0], reduced_2d[idx, 1],
            c=[colors[i]], s=20, alpha=0.75,
            label=person.split()[-1]
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Chiều 1")
    ax.set_ylabel("Chiều 2")
    ax.legend(loc="upper right", fontsize=6, ncol=2,
              markerscale=1.5, framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_figure(fig, save_name)
    print(f"[✓] Đã lưu → results/{save_name}")


def plot_cluster_scatter(reduced_2d: np.ndarray,
                          cluster_labels: np.ndarray,
                          title: str,
                          save_name: str):
    """Scatter plot 2D — màu theo cluster (KMeans)."""
    k = len(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(11, 8))
    for c in range(k):
        idx = cluster_labels == c
        ax.scatter(reduced_2d[idx, 0], reduced_2d[idx, 1],
                   c=[colors[c]], s=15, alpha=0.7, label=f"Cụm {c}")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Chiều 1")
    ax.set_ylabel("Chiều 2")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_figure(fig, save_name)
    print(f"[✓] Đã lưu → results/{save_name}")


def plot_similarity_heatmap(embeddings: np.ndarray,
                             labels: np.ndarray,
                             n_people: int = 10):
    """
    Heatmap cosine similarity cho N người đầu tiên.
    Trục x và y là ảnh, sắp xếp theo người.
    """
    # Chọn n_people người có nhiều ảnh nhất
    unique, counts = np.unique(labels, return_counts=True)
    top_people = unique[np.argsort(counts)[::-1][:n_people]]

    # Lọc và sắp xếp
    mask = np.isin(labels, top_people)
    emb_sub = embeddings[mask]
    lbl_sub = labels[mask]

    # Sắp xếp theo người
    order = np.argsort(lbl_sub)
    emb_sub = emb_sub[order]
    lbl_sub = lbl_sub[order]

    sim_mat = cosine_similarity(emb_sub)   # (M, M)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_mat, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_title(f"Cosine Similarity Heatmap\n(Top {n_people} người)", fontsize=13)
    ax.set_xlabel("Ảnh (sắp theo người)")
    ax.set_ylabel("Ảnh (sắp theo người)")

    # Đánh dấu ranh giới giữa các người
    boundaries = np.where(lbl_sub[:-1] != lbl_sub[1:])[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8)

    plt.tight_layout()
    save_figure(fig, "05_similarity_heatmap.png")
    print("[✓] Đã lưu → results/05_similarity_heatmap.png")


def plot_pca_variance(embeddings: np.ndarray):
    """Vẽ đồ thị tích luỹ phương sai giải thích bởi PCA."""
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(embeddings)
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, linewidth=1.5, color="steelblue")
    ax.axhline(90, color="tomato", linestyle="--", linewidth=1, label="90% variance")
    ax.axhline(95, color="darkorange", linestyle="--", linewidth=1, label="95% variance")
    ax.set_xlabel("Số thành phần PCA")
    ax.set_ylabel("Phương sai giải thích tích luỹ (%)")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.legend()
    ax.grid(alpha=0.3)
    n90 = np.searchsorted(cum_var, 90) + 1
    n95 = np.searchsorted(cum_var, 95) + 1
    print(f"[✓] PCA: cần {n90} thành phần để giải thích 90% phương sai")
    print(f"[✓] PCA: cần {n95} thành phần để giải thích 95% phương sai")
    plt.tight_layout()
    save_figure(fig, "05_pca_variance.png")
    print("[✓] Đã lưu → results/05_pca_variance.png")


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BƯỚC 4 — VISUALIZATION (PCA + t-SNE + HEATMAP)")
    print("=" * 55 + "\n")

    embeddings, labels = load_embeddings(with_images=False)

    # Load cluster labels nếu có
    cluster_path = os.path.join(EMBED_DIR, "cluster_labels.npy")
    cluster_labels = None
    if os.path.exists(cluster_path):
        loaded_clusters = np.load(cluster_path)
        if len(loaded_clusters) == len(embeddings):
            cluster_labels = loaded_clusters
            print(f"[✓] Đã load cluster labels — {len(np.unique(cluster_labels))} cụm")
        else:
            print(
                "[!] Bỏ qua cluster_labels.npy vì không khớp embeddings "
                f"({len(loaded_clusters)} labels vs {len(embeddings)} embeddings). "
                "Hãy chạy lại: python src/04_cluster.py"
            )

    # 1. PCA variance
    print("\n[1] PCA Explained Variance …")
    plot_pca_variance(embeddings)

    # 2. PCA 2D scatter — màu theo người
    print("\n[2] PCA 2D scatter …")
    pca_2d = reduce_pca(embeddings, n_components=2)
    plot_scatter(pca_2d, labels,
                 title="PCA — Embedding Space (màu theo người)",
                 save_name="05_pca_by_person.png")

    # 3. PCA 2D scatter — màu theo cụm
    if cluster_labels is not None:
        plot_cluster_scatter(pca_2d, cluster_labels,
                             title="PCA — Embedding Space (màu theo cụm KMeans)",
                             save_name="05_pca_by_cluster.png")

    # 4. t-SNE scatter — màu theo người
    print("\n[3] t-SNE 2D scatter …")
    tsne_2d = reduce_tsne(embeddings)
    plot_scatter(tsne_2d, labels,
                 title="t-SNE — Embedding Space (màu theo người)",
                 save_name="05_tsne_by_person.png")

    # 5. t-SNE scatter — màu theo cụm
    if cluster_labels is not None:
        plot_cluster_scatter(tsne_2d, cluster_labels,
                             title="t-SNE — Embedding Space (màu theo cụm KMeans)",
                             save_name="05_tsne_by_cluster.png")

    # 6. Similarity heatmap
    print("\n[4] Similarity heatmap …")
    plot_similarity_heatmap(embeddings, labels, n_people=10)

    print("\n" + "=" * 55)
    print("  ✅ TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT!")
    print("  Xem kết quả trong thư mục: results/")
    print("=" * 55)
