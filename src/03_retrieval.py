"""
03_retrieval.py — Bước 3A: Top-K Face Retrieval dựa trên Cosine Similarity

Chức năng:
- Load embeddings đã trích sẵn
- Tính cosine similarity giữa query và toàn bộ dataset
- Trả về Top-K khuôn mặt giống nhất
- Vẽ kết quả trực quan

Metric: Cosine Similarity
  sim(A, B) = (A · B) / (‖A‖ × ‖B‖)
  Giá trị trong [-1, 1]; càng gần 1 = càng giống nhau.
  Ưu điểm so với Euclidean: không bị ảnh hưởng bởi magnitude của vector.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_embeddings, save_figure

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
TOP_K = 5                 # Số kết quả trả về
SAME_PERSON_THRESH = 0.7  # Ngưỡng cosine để coi là cùng người


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Tính ma trận cosine similarity (N × N) cho toàn bộ dataset.
    Lưu ý: sklearn.cosine_similarity đã xử lý L2-norm tự động.
    """
    return cosine_similarity(embeddings)   # (N, N)


def query_top_k(query_idx: int,
                embeddings: np.ndarray,
                labels: np.ndarray,
                k: int = TOP_K):
    """
    Tìm Top-K ảnh giống nhất với ảnh query.

    Args:
        query_idx: index của ảnh query trong dataset
        embeddings: (N, 512) — toàn bộ embedding
        labels: (N,) — nhãn tên người
        k: số kết quả trả về

    Returns:
        top_indices : (K,) — index trong dataset
        top_scores  : (K,) — cosine similarity score
    """
    query_emb = embeddings[query_idx].reshape(1, -1)       # (1, 512)
    scores = cosine_similarity(query_emb, embeddings)[0]   # (N,)

    # Không lấy chính query ảnh
    scores[query_idx] = -1.0

    top_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_indices]
    return top_indices, top_scores


def plot_retrieval_result(query_idx: int,
                          top_indices: np.ndarray,
                          top_scores: np.ndarray,
                          images: np.ndarray,
                          labels: np.ndarray,
                          save_name: str = "03_retrieval_result.png"):
    """Vẽ ảnh query và K kết quả giống nhất."""
    k = len(top_indices)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5))
    fig.suptitle("Face Similarity Retrieval — Top-K Results", fontsize=13)

    # Query
    axes[0].imshow(images[query_idx])
    axes[0].set_title("QUERY\n" + labels[query_idx].split()[-1],
                       fontsize=9, color="royalblue", fontweight="bold")
    axes[0].axis("off")
    # Viền xanh cho query
    for spine in axes[0].spines.values():
        spine.set_edgecolor("royalblue")
        spine.set_linewidth(3)

    # Kết quả
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        axes[i + 1].imshow(images[idx])
        is_same = labels[idx] == labels[query_idx]
        color = "green" if is_same else "tomato"
        marker = "✓" if is_same else "✗"
        axes[i + 1].set_title(
            f"{marker} {labels[idx].split()[-1]}\nsim={score:.3f}",
            fontsize=8, color=color
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    save_figure(fig, save_name)


def evaluate_threshold(sim_matrix: np.ndarray,
                        labels: np.ndarray,
                        threshold: float = SAME_PERSON_THRESH):
    """
    Đánh giá: với ngưỡng cosine, cặp ảnh cùng người (positive) có
    similarity > threshold không?

    Returns dict với precision, recall (simplified).
    """
    n = len(labels)
    tp = fp = fn = tn = 0

    # Chỉ xét upper triangle (tránh đếm 2 lần và tránh diagonal)
    for i in range(n):
        for j in range(i + 1, n):
            pred_same = sim_matrix[i, j] >= threshold
            actually_same = labels[i] == labels[j]
            if pred_same and actually_same:
                tp += 1
            elif pred_same and not actually_same:
                fp += 1
            elif not pred_same and actually_same:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

    print(f"\n📊 ĐÁNH GIÁ THRESHOLD = {threshold}")
    print(f"   TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   Accuracy  : {accuracy:.4f}")
    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def plot_similarity_distribution(sim_matrix: np.ndarray,
                                  labels: np.ndarray):
    """Vẽ phân phối cosine similarity cho cặp cùng người vs khác người."""
    n = len(labels)
    same_scores, diff_scores = [], []

    # Lấy mẫu ngẫu nhiên để tránh O(N²) quá lớn
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, n, size=(50_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    for i, j in pairs[:20_000]:
        score = sim_matrix[i, j]
        if labels[i] == labels[j]:
            same_scores.append(score)
        else:
            diff_scores.append(score)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diff_scores, bins=60, alpha=0.6, color="tomato",
            label="Khác người (Negative pairs)")
    ax.hist(same_scores, bins=60, alpha=0.7, color="steelblue",
            label="Cùng người (Positive pairs)")
    ax.axvline(SAME_PERSON_THRESH, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold = {SAME_PERSON_THRESH}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Số lượng cặp ảnh")
    ax.set_title("Phân phối Cosine Similarity — Same vs Different Person")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, "03_similarity_distribution.png")
    print("[✓] Đã lưu → results/03_similarity_distribution.png")


def plot_roc_curve(sim_matrix: np.ndarray,
                   labels: np.ndarray,
                   n_pairs: int = 50_000):
    """
    Vẽ ROC Curve và tính AUC cho bài toán Face Verification.

    Cách hoạt động:
      - Lấy ngẫu nhiên n_pairs cặp ảnh từ dataset
      - y_true  = 1 nếu cùng người, 0 nếu khác người
      - y_score = cosine similarity giữa 2 ảnh đó
      - Dùng sklearn roc_curve() để tính TPR / FPR tại mọi threshold
      - Tính AUC (Area Under Curve) — càng gần 1.0 càng tốt

    Args:
        sim_matrix : (N, N) cosine similarity matrix
        labels     : (N,) tên người
        n_pairs    : số cặp ảnh lấy mẫu (mặc định 50.000)
    """
    n = len(labels)
    rng = np.random.default_rng(42)

    # Lấy ngẫu nhiên n_pairs cặp (i, j) với i ≠ j
    pairs = rng.integers(0, n, size=(n_pairs * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:n_pairs]

    y_true  = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).astype(int)
    y_score = sim_matrix[pairs[:, 0], pairs[:, 1]]

    # Tính ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Tìm điểm EER (Equal Error Rate) — nơi FPR ≈ FNR
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_thresh = thresholds[eer_idx]
    eer_val    = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Vẽ
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1,
            linestyle="--", label="Random baseline (AUC = 0.50)")
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color="tomato", zorder=5, s=80,
               label=f"EER = {eer_val:.4f}  (threshold = {eer_thresh:.3f})")

    ax.axvline(fpr[eer_idx], color="tomato", linestyle=":", lw=1, alpha=0.6)
    ax.axhline(tpr[eer_idx], color="tomato", linestyle=":", lw=1, alpha=0.6)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("ROC Curve — Face Verification (FaceNet VGGFace2)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "03_roc_curve.png")
    print("[✓] Đã lưu → results/03_roc_curve.png")

    # In tóm tắt
    print(f"\n📈 KẾT QUẢ ROC / AUC")
    print(f"   AUC      : {roc_auc:.4f}  {'(Xuất sắc)' if roc_auc > 0.95 else '(Tốt)' if roc_auc > 0.85 else '(Trung bình)'}")
    print(f"   EER      : {eer_val:.4f}  (càng thấp càng tốt)")
    print(f"   Threshold EER: {eer_thresh:.4f}")
    print(f"   Số cặp mẫu  : {len(pairs):,}  ({y_true.sum():,} positive / {(1-y_true).sum():,} negative)")

    return roc_auc


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BƯỚC 3A — TOP-K FACE RETRIEVAL")
    print("=" * 55 + "\n")

    # Load embeddings + ảnh
    embeddings, labels, images = load_embeddings(with_images=True)

    # Demo: chọn ngẫu nhiên 3 query
    rng = np.random.default_rng(0)
    query_indices = rng.integers(0, len(embeddings), size=3)

    print(f"[*] Demo retrieval với {len(query_indices)} ảnh query …\n")
    for i, qidx in enumerate(query_indices):
        print(f"  Query [{i+1}]: {labels[qidx]} (idx={qidx})")
        top_idx, top_scores = query_top_k(qidx, embeddings, labels, k=TOP_K)
        for rank, (tidx, sc) in enumerate(zip(top_idx, top_scores), 1):
            match = "✓" if labels[tidx] == labels[qidx] else "✗"
            print(f"    #{rank} {match} {labels[tidx]:<30s} sim={sc:.4f}")
        plot_retrieval_result(
            qidx, top_idx, top_scores, images, labels,
            save_name=f"03_retrieval_query{i+1}.png"
        )
        print()

    # Phân phối similarity
    print("[*] Tính ma trận similarity (có thể mất vài giây) …")
    sim_mat = cosine_sim_matrix(embeddings)
    plot_similarity_distribution(sim_mat, labels)

    # Đánh giá threshold
    evaluate_threshold(sim_mat, labels, threshold=SAME_PERSON_THRESH)

    # ROC Curve + AUC
    print("\n[*] Vẽ ROC Curve + tính AUC …")
    plot_roc_curve(sim_mat, labels)

    print("\n[✓] Retrieval hoàn tất. Chạy tiếp: python src/04_cluster.py")
