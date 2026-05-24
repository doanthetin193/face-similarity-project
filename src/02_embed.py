"""
02_embed.py — Bước 2: Trích xuất Face Embedding bằng FaceNet (PHẦN AI CỐT LÕI)

Quy trình:
  1. Load ảnh từ LFW dataset
  2. LFW đã là face crop nên resize/normalize trực tiếp về 160x160
  3. Dùng InceptionResnetV1 (pretrained VGGFace2) để trích 512-dim embedding
  4. Lưu embeddings.npy + labels.npy + images.npy

Ghi chú:
  - File này KHÔNG dùng MTCNN; MTCNN nằm ở luồng custom/query ngoài.
  - InceptionResnetV1 = FaceNet backbone, pretrained on VGGFace2 (3.31M images, 9,131 identities)
  - Embedding 512D: mặt giống nhau → vector gần nhau trong không gian 512D
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_embeddings

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32          # Số ảnh xử lý cùng lúc
IMAGE_SIZE = 160         # FaceNet yêu cầu 160×160
MIN_FACES = 20


def build_resnet(device: str) -> InceptionResnetV1:
    """Khoi tao FaceNet (InceptionResnetV1 pretrained VGGFace2)."""
    resnet = InceptionResnetV1(
        pretrained="vggface2",   # Pretrained tren VGGFace2 (3.31M images, 9,131 identities)
        classify=False,          # Lay embedding, khong classify
    ).eval().to(device)
    print(f"[OK] FaceNet san sang — Device: {device.upper()}")
    print(f"     Model : InceptionResnetV1 (VGGFace2) -> 512-dim embedding\n")
    return resnet


def preprocess_face_direct(img_np: np.ndarray,
                            size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Xử lý ảnh mặt đã crop (LFW) mà không qua MTCNN detect.
    LFW images đã là face crop, chỉ cần resize → normalize → tensor.
    FaceNet yêu cầu input: tensor (3, 160, 160), range [-1, 1].
    """
    pil = Image.fromarray((img_np * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BILINEAR)
    arr = np.array(pil).astype("float32") / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype="float32")
    std  = np.array([0.5, 0.5, 0.5], dtype="float32")
    arr  = (arr - mean) / std                                        # [-1, 1]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()        # (3, H, W), float32
    return tensor


def extract_embeddings(images_np: np.ndarray,
                       labels: np.ndarray,
                       resnet: InceptionResnetV1,
                       device: str,
                       batch_size: int = BATCH_SIZE):
    """
    Trích xuất 512-dim embedding cho mỗi ảnh LFW.

    LFW images đã là face crop (62x47) → bypass MTCNN detect,
    resize trực tiếp 160x160 rồi đưa vào InceptionResnetV1.
    (Với ảnh thực tế ngoài LFW → dùng MTCNN detect trước.)

    Returns:
        embeddings   : np.ndarray (N, 512)
        final_labels : np.ndarray (N,)
        final_images : np.ndarray (N, H, W, 3)
    """
    n = len(images_np)
    all_embeddings = []

    print(f"[*] Bắt đầu trích embedding cho {n:,} ảnh ...")
    print(f"    Batch size = {batch_size}, Device = {device.upper()}\n")

    for start in tqdm(range(0, n, batch_size), desc="Embedding"):
        batch_np = images_np[start: start + batch_size]
        tensors = [preprocess_face_direct(img) for img in batch_np]
        batch_tensor = torch.stack(tensors).to(device)   # (B, 3, 160, 160)

        with torch.no_grad():
            embs = resnet(batch_tensor)   # (B, 512)

        all_embeddings.append(embs.cpu().numpy())

    embeddings = np.vstack(all_embeddings)   # (N, 512)

    print(f"\n[OK] Trích embedding hoan tat!")
    print(f"    So anh     : {n:,}")
    print(f"    Shape      : {embeddings.shape}  (N x 512)")
    print(f"    Range emb  : [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    return embeddings, labels, images_np



# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BƯỚC 2 — FACE EMBEDDING EXTRACTION (AI CORE)")
    print("=" * 55 + "\n")

    # 1. Load data
    from sklearn.datasets import fetch_lfw_people
    print("[*] Loading LFW dataset …")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES, resize=0.5, color=True)
    images = lfw.images.astype("float32")  # color=True → đã là [0, 1]
    labels = np.array([lfw.target_names[t] for t in lfw.target])
    print(f"[✓] Loaded {len(images):,} ảnh — {len(np.unique(labels))} người\n")

    # 2. Build FaceNet model
    resnet = build_resnet(DEVICE)

    # 3. Extract embeddings
    embeddings, final_labels, final_images = extract_embeddings(
        images, labels, resnet, DEVICE
    )

    # 4. Save
    save_embeddings(embeddings, final_labels, final_images)
    print("\n[OK] Chay tiep: python src/03_retrieval.py")
