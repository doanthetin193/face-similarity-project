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
import math
from PIL import Image
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
CLUSTER_LABELS_PATH = os.path.join(EMBED_DIR, "cluster_labels.npy")


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
    if os.path.exists(CLUSTER_LABELS_PATH):
        os.remove(CLUSTER_LABELS_PATH)
        print("[OK] Da xoa cluster_labels.npy cu. Hay chay lai 04_cluster.py.")
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
    if len(labels) != len(embeddings):
        raise ValueError(
            f"Embeddings/labels không khớp: {len(embeddings)} embeddings "
            f"nhưng có {len(labels)} labels. Hãy embed lại dataset."
        )
    print(f"[OK] Da load {len(embeddings)} embeddings - {embeddings.shape[1]}D")
    if with_images and os.path.exists(IMAGES_PATH):
        images = np.load(IMAGES_PATH, allow_pickle=False)
        if len(images) != len(embeddings):
            raise ValueError(
                f"Embeddings/images không khớp: {len(embeddings)} embeddings "
                f"nhưng có {len(images)} images. Hãy embed lại dataset."
            )
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

# ──────────────────────────────────────────────
# 4. Face Alignment
# ──────────────────────────────────────────────
def align_and_crop_face(img: Image.Image, mtcnn_model, points=None):
    """
    Phát hiện (nếu chưa có points), xoay thẳng mặt (align) và trả về tensor.
    points: Mảng numpy shape (5, 2) chứa 5 điểm mốc (landmarks) của khuôn mặt.
    """
    if points is None:
        try:
            boxes, probs, landmarks = mtcnn_model.detect(img, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return None
            best_idx = int(np.argmax(probs)) if probs is not None else 0
            points = landmarks[best_idx]
        except Exception:
            return None
            
    left_eye = points[0]
    right_eye = points[1]
    
    # Tính góc nghiêng (đơn vị độ)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    # Nếu góc nghiêng quá nhỏ (< 2 độ), coi như đã thẳng
    if abs(angle) < 2.0:
        return mtcnn_model(img)
        
    # Tính tâm xoay (giữa 2 mắt)
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )
    
    # Xoay ảnh gốc (ngược lại góc nghiêng)
    rotated_img = img.rotate(angle, center=eye_center, resample=Image.BICUBIC)
    
    # Truyền ảnh đã xoay vào MTCNN để lấy tensor đã crop
    face_tensor = mtcnn_model(rotated_img)
    
    # Fallback nếu xoay xong mtcnn không nhận ra mặt
    if face_tensor is None:
        face_tensor = mtcnn_model(img)
        
    return face_tensor
