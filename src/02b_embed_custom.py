"""
02b_embed_custom.py — Trích embedding từ custom dataset (hỗ trợ nhiều cấu trúc thư mục)

Hỗ trợ 2 kiểu cấu trúc — tự động nhận dạng:

  Kiểu 2 cấp (Vietnamese Celebrity Faces):
    custom_dataset/
    ├── Ca sĩ/
    │   ├── ca sĩ Bảo Anh/
    │   │   ├── anh1.jpg  ← ảnh nằm ở đây
    │   └── ...
    └── ...

  Kiểu 1 cấp (VN-Celeb — ID số, không có tên):
    custom_dataset/
    └── VN-celeb/
        ├── 1/
        │   ├── 0.png  ← ảnh nằm ở đây
        └── ...

Cả 2 kiểu được gộp chung vào 1 embeddings.npy duy nhất.
Output: embeddings/ (ghi đè embeddings.npy, labels.npy, images.npy)
Sau đó dùng 03_retrieval.py, query_external.py bình thường.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from src.utils import save_embeddings, align_and_crop_face

from facenet_pytorch import MTCNN, InceptionResnetV1

# ──────────────────────────────────────────────────────
# Tham số
# ──────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 160
BATCH_SIZE = 32
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "custom_dataset"
)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in SUPPORTED_EXT


def _scan_one_root(root: str) -> list:
    """
    Quét 1 thư mục gốc, tự nhận dạng cấu trúc:
      - 1 cấp: root/Person/image.*
      - 2 cấp: root/Category/Person/image.*
    Trả về list of (image_path, label)
    """
    samples = []

    for entry in sorted(os.listdir(root)):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue

        # Kiểm tra xem entry_path chứa ảnh trực tiếp không (1 cấp)
        sub_entries = os.listdir(entry_path)
        has_images  = any(_is_image(f) for f in sub_entries)
        has_subdirs = any(os.path.isdir(os.path.join(entry_path, f)) for f in sub_entries)

        if has_images and not has_subdirs:
            # Cấu trúc 1 cấp: entry_path là thư mục người, ảnh nằm ngay trong này
            label = entry
            for fname in sub_entries:
                if _is_image(fname):
                    samples.append((os.path.join(entry_path, fname), label))

        elif has_subdirs:
            # Cấu trúc 2 cấp: entry_path là category, bên trong là thư mục người
            for person in sorted(os.listdir(entry_path)):
                person_path = os.path.join(entry_path, person)
                if not os.path.isdir(person_path):
                    continue
                label = person
                for fname in os.listdir(person_path):
                    if _is_image(fname):
                        samples.append((os.path.join(person_path, fname), label))

    return samples


def scan_dataset(root: str) -> list:
    """
    Quét toàn bộ custom_dataset/, tự nhận dạng & gộp kết quả từ tất cả thư mục con.
    Trả về list of (image_path, label)
    """
    return _scan_one_root(root)


def build_models(device: str):
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,
        keep_all=False,
        device=device,
        post_process=True,   # output [-1, 1]
    )
    resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    return mtcnn, resnet


def embed_dataset(samples, mtcnn, resnet, device, batch_size=BATCH_SIZE):
    """
    Trích embedding cho toàn bộ dataset.
    Dùng MTCNN detect mặt trước (ảnh ngoài không đảm bảo đã crop).
    Ảnh nào không detect được mặt → bỏ qua.
    """
    all_embeddings = []
    all_labels     = []
    all_images     = []
    skipped        = 0

    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    print(f"[*] Tổng số file ảnh   : {len(samples):,}")
    print(f"[*] Device             : {device.upper()}")
    print(f"[*] Batch size         : {batch_size}\n")

    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
        batch_paths  = paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        pil_imgs = []
        valid_labels = []
        valid_pils   = []

        for path, lbl in zip(batch_paths, batch_labels):
            try:
                img = Image.open(path).convert("RGB")
                pil_imgs.append(img)
                valid_labels.append(lbl)
                valid_pils.append(img)
            except Exception:
                skipped += 1
                continue

        if not pil_imgs:
            continue

        # MTCNN detect từng ảnh (tránh lỗi batch inhomogeneous shape)
        tensors_ok = []
        labels_ok  = []
        imgs_ok    = []

        for pil, lbl in zip(valid_pils, valid_labels):
            try:
                tensor = align_and_crop_face(pil, mtcnn)
            except Exception:
                tensor = None
            if tensor is None:
                skipped += 1
                continue
            tensors_ok.append(tensor)
            labels_ok.append(lbl)
            # Lưu ảnh resize nhỏ để hiển thị (không normalize, range [0,1])
            arr = np.array(pil.resize((62, 62))).astype("float32") / 255.0
            imgs_ok.append(arr)

        if not tensors_ok:
            continue

        batch_tensor = torch.stack(tensors_ok).to(device)   # (B, 3, 160, 160)
        with torch.no_grad():
            embs = resnet(batch_tensor).cpu().numpy()        # (B, 512)

        all_embeddings.append(embs)
        all_labels.extend(labels_ok)
        all_images.extend(imgs_ok)

    if not all_embeddings:
        raise RuntimeError(
            "Không embed được ảnh nào. Hãy kiểm tra custom_dataset/ và chất lượng ảnh."
        )

    embeddings = np.vstack(all_embeddings)        # (N, 512)
    labels_arr = np.array(all_labels)
    images_arr = np.stack(all_images)             # (N, 62, 62, 3)

    return embeddings, labels_arr, images_arr, skipped


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BƯỚC 2B — EMBED VIETNAMESE CELEBRITY DATASET")
    print("=" * 60 + "\n")

    if not os.path.isdir(DATASET_DIR):
        print(f"[!] Không tìm thấy thư mục: {DATASET_DIR}")
        sys.exit(1)

    # 1. Scan
    print(f"[*] Quét dataset: {DATASET_DIR}")
    samples = scan_dataset(DATASET_DIR)

    # Thống kê
    from collections import Counter
    person_counts = Counter(lbl for _, lbl in samples)
    print(f"[✓] Tổng ảnh  : {len(samples):,}")
    print(f"[✓] Số người  : {len(person_counts)}")
    print(f"\n  Top-10 người nhiều ảnh nhất:")
    for name, cnt in person_counts.most_common(10):
        print(f"    {name:<35s}: {cnt:>3d} ảnh")
    print()

    # 2. Build models
    print("[*] Khởi tạo MTCNN + FaceNet …")
    mtcnn, resnet = build_models(DEVICE)
    print(f"[✓] Models sẵn sàng — Device: {DEVICE.upper()}\n")

    # 3. Embed
    embeddings, labels, images, skipped = embed_dataset(
        samples, mtcnn, resnet, DEVICE
    )

    print(f"\n[✓] Hoàn tất embedding!")
    print(f"    Ảnh đã embed : {len(embeddings):,}")
    print(f"    Ảnh bỏ qua  : {skipped:,}  (không detect được mặt)")
    print(f"    Shape        : {embeddings.shape}")
    print(f"    Range        : [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    # 4. Lưu (ghi đè embeddings cũ)
    save_embeddings(embeddings, labels, images)
    print("\n[✓] Chạy tiếp các bước khác bình thường:")
    print("    python src/03_retrieval.py")
    print("    python query_external.py <anh.jpg>")
