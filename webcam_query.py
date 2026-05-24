# -*- coding: utf-8 -*-
"""
webcam_query.py -- Nhan dien khuon mat real-time qua webcam

Cach dung:
    python webcam_query.py              # Top-3 (mac dinh)
    python webcam_query.py --topk 5    # Top-5
    python webcam_query.py --camera 1  # Camera thu 2

Phim tat:
    Q / ESC  : Thoat
    S        : Chup anh va luu vao results/webcam_snapshot.png
    SPACE    : Tam dung / Tiep tuc

Yeu cau:
    - Da chay 02_embed.py hoac 02b_embed_custom.py (co embeddings.npy)
    - pip install opencv-python facenet-pytorch torch
"""

import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_embeddings, RESULTS_DIR, align_and_crop_face

# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 160
FRAME_SKIP  = 4        # Chi embed moi N frame (tang FPS)
CONF_THRESH = 0.85     # Chi xu ly mat co confidence >= nguong nay
MATCH_THRESH = 0.70    # Dong bo voi SAME_PERSON_THRESH trong src/03_retrieval.py
SIMILAR_THRESH = 0.60  # Muc dien giai mem cho webcam/UI
BOX_COLOR_MATCH   = (0, 220, 80)    # Xanh la -- tim thay (BGR)
BOX_COLOR_SEARCH  = (0, 165, 255)   # Cam -- dang xu ly
BOX_COLOR_NONE    = (80, 80, 80)    # Xam -- khong co mat
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Font he thong ho tro Unicode / tieng Viet
_PIL_FONT_CACHE = {}

def _get_pil_font(size=16):
    """Load font he thong ho tro Unicode, cache lai de tai su dung."""
    if size in _PIL_FONT_CACHE:
        return _PIL_FONT_CACHE[size]
    for path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _PIL_FONT_CACHE[size] = font
                return font
            except Exception:
                pass
    return None


def put_vn_text(img_bgr, text, pos, font_size=16,
                fg=(255, 255, 255), bg=None, padding=4):
    """
    Ve text Unicode (tieng Viet) len OpenCV BGR image dung PIL.
    Tra ve img_bgr da chinh sua.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font    = _get_pil_font(font_size)
    x, y   = pos
    # Ve nen neu co
    if bg is not None:
        if font:
            bbox = draw.textbbox((x, y), text, font=font)
        else:
            bbox = (x, y, x + len(text) * font_size // 2, y + font_size)
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding,
             bbox[2] + padding, bbox[3] + padding],
            fill=(bg[2], bg[1], bg[0])   # BGR -> RGB
        )
    # Ve chu
    rgb_fg = (fg[2], fg[1], fg[0])       # BGR -> RGB
    if font:
        draw.text((x, y), text, fill=rgb_fg, font=font)
    else:
        draw.text((x, y), text, fill=rgb_fg)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────
# Tien ich ve len frame
# ─────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    """Ve hinh chu nhat voi goc bo (fallback neu khong ho tro)."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)


def draw_text_bg(img, text, pos, font_scale=0.55, color=(255,255,255),
                 bg=(30,30,30), thickness=1, padding=5):
    """Ve text co nen mo (de doc hon tren nen bat ky)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg, -1)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_sidebar(frame, top_labels, top_scores, top_k, top_imgs=None):
    """Ve panel ben phai hien thi ket qua Top-K (co anh thumbnail)."""
    h, w = frame.shape[:2]
    panel_w  = 330
    THUMB    = 62   # kich thuoc thumbnail
    ROW_H    = 82   # chieu cao 1 dong ket qua
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 30)

    # Tieu de
    cv2.putText(panel, f"TOP-{top_k} SIMILAR", (10, 35),
                FONT, 0.65, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (10, 45), (panel_w - 10, 45), (80, 80, 120), 1)

    if top_labels is None:
        cv2.putText(panel, "No face detected", (10, 90),
                    FONT, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        for i, (lbl, sc) in enumerate(zip(top_labels, top_scores)):
            y0 = 52 + i * ROW_H

            rank_color = (100, 220, 100) if sc >= MATCH_THRESH else \
                         (100, 180, 255) if sc >= SIMILAR_THRESH else (150, 150, 150)

            # ── Thumbnail ─────────────────────────────────
            if top_imgs is not None:
                img = top_imgs[i]
                img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 \
                         else img.astype(np.uint8)
                thumb = cv2.resize(img_u8, (THUMB, THUMB))
                panel[y0: y0 + THUMB, 6: 6 + THUMB] = thumb
                # Vien mau theo score
                cv2.rectangle(panel, (5, y0 - 1),
                              (6 + THUMB, y0 + THUMB), rank_color, 2)

            # ── Text (ben phai thumbnail) ──────────────────
            tx = 6 + THUMB + 8   # = 76

            # So thu tu
            cv2.putText(panel, f"#{i+1}", (tx, y0 + 14),
                        FONT, 0.55, rank_color, 1, cv2.LINE_AA)

            # Ten nguoi (tieng Viet, dung PIL)
            name = lbl if len(lbl) <= 20 else lbl[:18] + ".."
            panel = put_vn_text(panel, name, (tx, y0 + 18),
                                font_size=13, fg=(230, 230, 230))

            # Score text
            score_txt = f"sim = {sc:.3f}"
            tag = "Match" if sc >= MATCH_THRESH else ("Similar" if sc >= SIMILAR_THRESH else "Low")
            cv2.putText(panel, f"{score_txt}  [{tag}]", (tx, y0 + 50),
                        FONT, 0.38, rank_color, 1, cv2.LINE_AA)

            # Progress bar
            bar_x1, bar_y = tx, y0 + 56
            bar_w_max = panel_w - tx - 12
            bar_w = int(np.clip(sc, 0, 1) * bar_w_max)
            cv2.rectangle(panel, (bar_x1, bar_y),
                          (bar_x1 + bar_w_max, bar_y + 7), (50, 50, 60), -1)
            cv2.rectangle(panel, (bar_x1, bar_y),
                          (bar_x1 + bar_w, bar_y + 7), rank_color, -1)

    # Huong dan phim
    cv2.line(panel, (10, h - 80), (panel_w - 10, h - 80), (60, 60, 80), 1)
    for i, hint in enumerate(["[Q/ESC] Thoat", "[S] Chup anh", "[SPACE] Tam dung"]):
        cv2.putText(panel, hint, (10, h - 60 + i * 18),
                    FONT, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    return np.hstack([frame, panel])


# ─────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────
def embed_frame(pil_img, mtcnn, resnet):
    """
    Tu 1 frame PIL RGB:
      1. MTCNN detect mat → tra ve box, prob, tensor
      2. FaceNet trich embedding 512D
    Returns: (box, prob, embedding) hoac (None, None, None) neu khong detect duoc
    """
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)

    if boxes is None or probs[0] is None or probs[0] < CONF_THRESH:
        return None, None, None

    # Lay mat co confidence cao nhat
    best_idx = int(np.argmax(probs))
    box  = boxes[best_idx].astype(int)
    prob = probs[best_idx]
    points = landmarks[best_idx]

    # Lay tensor da align va crop
    tensor = align_and_crop_face(pil_img, mtcnn, points)   # (3, 160, 160) hoac None
    if tensor is None:
        return None, None, None

    with torch.no_grad():
        emb = resnet(tensor.unsqueeze(0).to(DEVICE))   # (1, 512)

    return box, prob, emb.cpu().numpy()[0]   # (512,)


def find_top_k(query_emb, db_embeddings, db_labels, db_imgs, k):
    scores   = cosine_similarity(query_emb.reshape(1, -1), db_embeddings)[0]
    top_idx  = np.argsort(scores)[::-1][:k]
    return db_labels[top_idx], scores[top_idx], db_imgs[top_idx]


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def run(camera_idx: int = 0, top_k: int = 3):
    print("=" * 55)
    print("  WEBCAM REAL-TIME FACE SIMILARITY QUERY")
    print("=" * 55)
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Top-K   : {top_k}")
    print(f"  Camera  : {camera_idx}")
    print()

    # 1. Load embeddings + images
    print("[*] Load embeddings ...")
    db_embs, db_labels, db_imgs = load_embeddings(with_images=True)
    print(f"[OK] {len(db_embs):,} mat trong dataset\n")

    # 2. Build models
    print("[*] Khoi tao MTCNN + FaceNet ...")
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, margin=20,
        keep_all=False, device=DEVICE,
        post_process=True,
    )
    resnet = InceptionResnetV1(pretrained="vggface2",
                               classify=False).eval().to(DEVICE)
    print(f"[OK] Models san sang\n")

    # 3. Mo webcam
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc camera {camera_idx}!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Webcam san sang. Nhan Q hoac ESC de thoat.\n")

    frame_count = 0
    paused      = False
    last_box    = None
    last_labels = None
    last_scores = None
    last_imgs   = None
    fps_list      = []
    t_prev        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Khong doc duoc frame tu camera!")
            break

        frame_count += 1

        # Tinh FPS
        t_now = time.time()
        fps_list.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_list) > 20:
            fps_list.pop(0)
        fps = np.mean(fps_list)

        if not paused and frame_count % FRAME_SKIP == 0:
            # BGR (cv2) -> RGB (PIL)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            box, prob, emb = embed_frame(pil_img, mtcnn, resnet)

            if emb is not None:
                last_box = box
                last_labels, last_scores, last_imgs = find_top_k(
                    emb, db_embs, db_labels, db_imgs, top_k
                )
            else:
                last_box    = None
                last_labels = None
                last_scores = None
                last_imgs   = None

        # ─── Ve len frame ───────────────────────────────
        display = frame.copy()

        # Bounding box
        if last_box is not None:
            x1, y1, x2, y2 = last_box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(display.shape[1] - 1, x2)
            y2 = min(display.shape[0] - 1, y2)
            draw_rounded_rect(display, (x1, y1), (x2, y2), BOX_COLOR_MATCH, 2)
            if last_labels is not None:
                name_short = last_labels[0]
                if len(name_short) > 18:
                    name_short = name_short[:16] + ".."
                label_txt = f"{name_short} ({last_scores[0]:.2f})"
                display = put_vn_text(
                    display, label_txt,
                    (x1, max(y1 - 24, 4)),
                    font_size=16,
                    fg=(200, 255, 200),
                    bg=(0, 100, 40)
                )
        else:
            draw_text_bg(display, "Dang tim mat...", (10, 30),
                         font_scale=0.5, color=(180, 180, 180),
                         bg=(30, 30, 30))

        # FPS + trang thai
        status = "PAUSED" if paused else "LIVE"
        draw_text_bg(display, f"FPS: {fps:.1f}  |  {status}",
                     (10, display.shape[0] - 12),
                     font_scale=0.45, color=(200, 200, 255), bg=(20, 20, 40))

        # Ghep sidebar
        output = draw_sidebar(display, last_labels, last_scores, top_k,
                               top_imgs=last_imgs)

        cv2.imshow("Face Similarity - Webcam (Q de thoat)", output)

        # ─── Xu ly phim ─────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q hoac ESC
            break
        elif key == ord('s') or key == ord('S'):
            snap_path = os.path.join(RESULTS_DIR, "webcam_snapshot.png")
            cv2.imwrite(snap_path, output)
            print(f"[OK] Da chup -> {snap_path}")
        elif key == ord(' '):
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Da thoat webcam.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Webcam real-time face similarity query"
    )
    parser.add_argument("--topk",   type=int, default=3,
                        help="So ket qua hien thi (mac dinh: 3)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index camera (mac dinh: 0)")
    args = parser.parse_args()

    run(camera_idx=args.camera, top_k=args.topk)
