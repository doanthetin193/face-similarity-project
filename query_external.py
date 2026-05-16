"""
query_external.py — Nhận ảnh từ bên ngoài, tìm Top-K khuôn mặt giống nhất trong dataset đã embed

Cách dùng:
    python query_external.py "đường_dẫn_ảnh.jpg"
    python query_external.py "C:/Users/ban/Pictures/selfie.jpg" --topk 5
    python query_external.py "anh.png" --topk 8

Yêu cầu:
    - Đã chạy 02_embed.py để có embeddings.npy
    - Ảnh đầu vào có khuôn mặt rõ ràng (nhìn thẳng, đủ sáng)

Đầu ra:
    - In tên + similarity score của Top-K người giống nhất
    - Lưu ảnh kết quả vào results/query_external_result.png
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

# Import utils từ thư mục src
from src.utils import load_embeddings, save_figure, RESULTS_DIR, align_and_crop_face

# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 160


def load_models(device: str):
    """Khởi tạo MTCNN (detect face) + FaceNet (embedding)."""
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,           # Lấy thêm vùng xung quanh mặt
        keep_all=False,      # Chỉ lấy mặt có confidence cao nhất
        device=device,
        post_process=True,   # Normalize [-1, 1] luôn
    )
    resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    return mtcnn, resnet


def embed_external_image(image_path: str, mtcnn: MTCNN,
                          resnet: InceptionResnetV1, device: str):
    """
    Nhận ảnh ngoài → detect mặt bằng MTCNN → trích embedding 512D.

    Returns:
        embedding  : np.ndarray (512,) hoặc None nếu không detect được mặt
        face_crop  : PIL.Image — ảnh mặt đã crop (để hiển thị)
    """
    img = Image.open(image_path).convert("RGB")

    # Detect, align và crop → tensor (3, 160, 160) range [-1, 1]
    face_tensor = align_and_crop_face(img, mtcnn)

    if face_tensor is None:
        return None, None

    # Chuyển tensor [-1,1] → PIL để hiển thị
    face_display = face_tensor.permute(1, 2, 0).numpy()
    face_display = (face_display + 1) / 2.0  # [-1,1] → [0,1]
    face_display = np.clip(face_display, 0, 1)
    face_pil = Image.fromarray((face_display * 255).astype(np.uint8))

    # Trích embedding
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device))  # (1, 512)

    return emb.cpu().numpy()[0], face_pil  # (512,), PIL


def find_top_k(query_emb: np.ndarray,
               embeddings: np.ndarray,
               labels: np.ndarray,
               k: int = 5):
    """
    So sánh query embedding với toàn bộ dataset.
    Returns: top_indices, top_scores, top_labels
    """
    query = query_emb.reshape(1, -1)
    scores = cosine_similarity(query, embeddings)[0]  # (N,)
    top_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_indices]
    top_labels = labels[top_indices]
    return top_indices, top_scores, top_labels


def plot_result(face_pil: Image.Image,
                top_indices: np.ndarray,
                top_scores: np.ndarray,
                top_labels: np.ndarray,
                dataset_images: np.ndarray,
                image_path: str,
                k: int):
    """Vẽ ảnh query + Top-K kết quả, lưu vào results/."""
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    fig.suptitle("External Query — Top-K Similar Faces in Dataset", fontsize=13)

    # Query image
    axes[0].imshow(face_pil)
    axes[0].set_title("YOUR PHOTO\n(face crop)", fontsize=9,
                       color="royalblue", fontweight="bold")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("royalblue")
        spine.set_linewidth(3)

    # Gradient màu xanh → đỏ theo rank
    cmap = plt.cm.RdYlGn
    for i, (idx, score, label) in enumerate(zip(top_indices, top_scores, top_labels)):
        ax = axes[i + 1]
        ax.imshow(dataset_images[idx])
        color = cmap(score)  # màu theo similarity score
        ax.set_title(
            f"#{i+1} {label.split()[-1]}\nsim={score:.3f}",
            fontsize=8, color="darkgreen" if score > 0.7 else "darkorange"
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    plt.tight_layout()
    out_name = "query_external_result.png"
    save_figure(fig, out_name)
    print(f"\n[✓] Kết quả đã lưu → results/{out_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Tìm Top-K khuôn mặt trong dataset đã embed giống nhất với ảnh của bạn"
    )
    parser.add_argument("image", help="Đường dẫn tới ảnh đầu vào (jpg/png/...)")
    parser.add_argument("--topk", type=int, default=5,
                        help="Số kết quả trả về (mặc định: 5)")
    args = parser.parse_args()

    image_path = args.image
    k = args.topk

    # Kiểm tra file tồn tại
    if not os.path.isfile(image_path):
        print(f"[!] Không tìm thấy file: {image_path}")
        sys.exit(1)

    print("=" * 55)
    print("  EXTERNAL FACE QUERY")
    print("=" * 55)
    print(f"[*] Ảnh đầu vào : {image_path}")
    print(f"[*] Top-K       : {k}")
    print(f"[*] Device      : {DEVICE.upper()}\n")

    # 1. Load embeddings dataset
    print("[*] Load embeddings dataset …")
    embeddings, labels, dataset_images = load_embeddings(with_images=True)

    # 2. Load models
    print("[*] Khởi tạo MTCNN + FaceNet …")
    mtcnn, resnet = load_models(DEVICE)
    print(f"[✓] Models sẵn sàng\n")

    # 3. Embed ảnh query
    print("[*] Detect khuôn mặt & trích embedding …")
    query_emb, face_pil = embed_external_image(image_path, mtcnn, resnet, DEVICE)

    if query_emb is None:
        print("\n[✗] Không phát hiện được khuôn mặt trong ảnh!")
        print("    Gợi ý:")
        print("    - Ảnh cần có khuôn mặt rõ ràng, nhìn thẳng")
        print("    - Đủ ánh sáng, không bị che khuất")
        print("    - Khuôn mặt chiếm ít nhất 1/4 diện tích ảnh")
        sys.exit(1)

    print(f"[✓] Embedding shape: {query_emb.shape}  "
          f"range=[{query_emb.min():.3f}, {query_emb.max():.3f}]")

    # 4. Tìm Top-K
    print(f"\n[*] Tìm Top-{k} khuôn mặt giống nhất trong {len(embeddings):,} ảnh dataset …")
    top_indices, top_scores, top_labels = find_top_k(query_emb, embeddings, labels, k)

    # 5. In kết quả
    print(f"\n{'─'*50}")
    print(f"  KẾT QUẢ TOP-{k}")
    print(f"{'─'*50}")
    for i, (score, label) in enumerate(zip(top_scores, top_labels), 1):
        bar = "█" * int(score * 20)
        tag = "✓ Khá giống" if score > 0.75 else ("~ Tương đồng" if score > 0.6 else "✗ Ít giống")
        print(f"  #{i}  {label:<32s}  sim={score:.4f}  {bar}  {tag}")
    print(f"{'─'*50}")

    # Giải thích score
    top1_score = top_scores[0]
    if top1_score > 0.8:
        verdict = "Kết quả rất tốt — khuôn mặt có cấu trúc tương đồng cao"
    elif top1_score > 0.7:
        verdict = "Kết quả tốt — tìm thấy người có nét tương đồng"
    elif top1_score > 0.6:
        verdict = "Tương đồng vừa — một số đặc trưng khuôn mặt gần nhau"
    else:
        verdict = "Ít tương đồng — khuôn mặt khá độc đáo so với dataset hiện tại"
    print(f"\n  → {verdict}")

    # 6. Lưu ảnh kết quả & mở lên xem
    plot_result(face_pil, top_indices, top_scores, top_labels, dataset_images, image_path, k)

    # Tự động mở ảnh kết quả bằng trình xem mặc định
    result_path = os.path.join(RESULTS_DIR, "query_external_result.png")
    print("[*] Đang mở ảnh kết quả …")
    os.startfile(result_path)


if __name__ == "__main__":
    main()
