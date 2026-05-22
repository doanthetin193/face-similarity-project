# -*- coding: utf-8 -*-
# app.py -- Streamlit Web UI cho Face Similarity Retrieval System
# Chay: streamlit run app.py

import sys, io
# Fix UnicodeEncodeError tren Windows (stdout mac dinh dung cp1252)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# ─── Path setup ──────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from utils import load_embeddings, RESULTS_DIR, EMBED_DIR, align_and_crop_face

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Similarity System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS -- giao dien toi, hien dai
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Nen tong the */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2d3e; }

/* Header chinh */
.main-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-title { color: #8b92a5; font-size: 1rem; margin-top: 4px; margin-bottom: 28px; }

/* Card ket qua */
.result-card {
    background: #1a1f2e; border: 1px solid #2a2d3e; border-radius: 12px;
    padding: 14px; text-align: center; transition: transform .2s;
}
.result-card:hover { transform: translateY(-3px); }
.sim-badge {
    display: inline-block; padding: 4px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-top: 6px;
}
.sim-high   { background: #1a3a2a; color: #4ade80; border: 1px solid #22c55e; }
.sim-medium { background: #1a2a3a; color: #60a5fa; border: 1px solid #3b82f6; }
.sim-uncertain { background: #2a2414; color: #fbbf24; border: 1px solid #f59e0b; }
.sim-low    { background: #2a2a1a; color: #facc15; border: 1px solid #eab308; }
.explain-box {
    background: #151a26; border: 1px solid #2a2d3e; border-radius: 10px;
    padding: 14px 16px; margin: 8px 0 14px 0; color: #d6d9e6;
}
.explain-box b { color: #ffffff; }
.step-box {
    background: #151a26; border-left: 3px solid #667eea; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 10px; color: #d6d9e6;
}

/* Metric box */
.metric-box {
    background: #1a1f2e; border: 1px solid #2a2d3e; border-radius: 10px;
    padding: 16px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-lbl { font-size: 0.78rem; color: #8b92a5; margin-top: 2px; }

/* Tab style */
[data-testid="stTab"] { font-size: 0.95rem; }

/* Divider */
hr { border-color: #2a2d3e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cache: Load embeddings + models (chi load 1 lan)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Dang load embeddings...")
def get_embeddings():
    embs, labels, imgs = load_embeddings(with_images=True)
    return embs, labels, imgs


@st.cache_resource(show_spinner="Dang load model FaceNet...")
def get_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn  = MTCNN(image_size=160, margin=20, keep_all=False,
                   device=device, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2",
                               classify=False).eval().to(device)
    return mtcnn, resnet, device


def tensor_to_display_image(tensor):
    """Chuyen tensor khuon mat cua MTCNN ve anh uint8 de hien thi."""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 128.0 + 127.5, 0, 255).astype(np.uint8)
    return arr


def embed_uploaded(pil_img, mtcnn, resnet, device):
    """Upload anh -> detect/align/crop -> FaceNet embed -> metadata hien thi."""
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None, None, 0, None

    valid_probs = np.array([p if p is not None else -1 for p in probs])
    best_idx = int(np.argmax(valid_probs))
    best_prob = float(valid_probs[best_idx])
    if best_prob < 0:
        return None, None, len(boxes), None

    tensor = align_and_crop_face(pil_img, mtcnn, landmarks[best_idx])
    if tensor is None:
        return None, None, len(boxes), best_prob
    with torch.no_grad():
        emb = resnet(tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy()[0], tensor, len(boxes), best_prob


def find_top_k(query_emb, db_embs, db_labels, db_imgs, k):
    scores     = cosine_similarity(query_emb.reshape(1, -1), db_embs)[0]
    top_idx    = np.argsort(scores)[::-1][:k]
    return db_labels[top_idx], scores[top_idx], db_imgs[top_idx], top_idx


EER_THRESHOLD = 0.4968


def sim_color_class(score):
    if score >= 0.75:
        return "sim-high", "Rất giống"
    if score >= 0.60:
        return "sim-medium", "Khá giống"
    if score >= EER_THRESHOLD:
        return "sim-uncertain", "Có tương đồng"
    return "sim-low", "Không chắc chắn"


def sim_explanation(score):
    css_cls, label = sim_color_class(score)
    if score >= 0.75:
        detail = "Mức tương đồng cao, có thể xem là ứng viên rất mạnh trong Top-K."
    elif score >= 0.60:
        detail = "Mức tương đồng khá tốt, nên đối chiếu thêm ảnh gốc khi báo cáo/demo."
    elif score >= EER_THRESHOLD:
        detail = "Điểm vượt ngưỡng EER nhưng chưa cao, hệ thống xem là có dấu hiệu tương đồng."
    else:
        detail = "Điểm thấp hơn ngưỡng EER, kết quả chỉ nên dùng để tham khảo."
    return css_cls, label, detail


def result_image_path(name):
    p = os.path.join(RESULTS_DIR, name)
    return p if os.path.exists(p) else None


def launch_webcam(top_k=3, camera=0):
    """Mo webcam_query.py tu Streamlit bang mot process rieng."""
    script_path = os.path.join(ROOT, "webcam_query.py")
    cmd = [sys.executable, script_path, "--topk", str(top_k), "--camera", str(camera)]
    kwargs = {"cwd": ROOT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(cmd, **kwargs)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Face Similarity")
    st.markdown("---")

    # Trang thai dataset
    embed_ok = os.path.exists(os.path.join(EMBED_DIR, "embeddings.npy"))
    if embed_ok:
        st.success("✅ Embeddings sẵn sàng")
    else:
        st.error("❌ Chưa có embeddings.npy")
        st.info("Chạy:\n```\npython src/02b_embed_custom.py\n```")

    st.markdown("---")
    st.markdown("**Mở Webcam từ web**")
    webcam_top_k = st.slider("Top-K webcam", 1, 10, 3)
    webcam_camera = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    if st.button("Mở webcam", disabled=not embed_ok, use_container_width=True):
        try:
            launch_webcam(webcam_top_k, webcam_camera)
            st.success("Đã mở webcam. Xem cửa sổ OpenCV trên máy.")
        except Exception as exc:
            st.error(f"Không mở được webcam: {exc}")
    st.caption("Chức năng này chạy local và mở cửa sổ OpenCV riêng, không nhúng trực tiếp vào trang web.")

    st.markdown("---")
    st.markdown("**Phím tắt Webcam**")
    st.markdown("`Q/ESC` Thoát  \n`S` Chụp ảnh  \n`SPACE` Tạm dừng")
    st.markdown("---")
    st.markdown(
        "<div style='color:#555;font-size:0.75rem'>"
        "FaceNet VGGFace2 · MTCNN · KMeans<br>"
        "Lập trình Trí tuệ Nhân tạo"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Face Similarity Retrieval System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">FaceNet (VGGFace2) · Cosine Similarity · KMeans · PCA · t-SNE</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Query Upload",
    "📊  Dataset Info",
    "📈  Kết quả & Biểu đồ",
    "🗂️  Clustering",
    "Pipeline & Giải thích",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Query Upload
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Tìm khuôn mặt giống nhất trong dataset")
    st.markdown("Upload ảnh bất kỳ có khuôn mặt → hệ thống tìm Top-K người giống nhất.")

    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        uploaded = st.file_uploader(
            "Chọn ảnh (jpg / png / webp)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="uploader"
        )
        top_k = st.slider("Số kết quả Top-K", 1, 10, 5)

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Ảnh bạn tải lên", use_container_width=True)

    with col_right:
        if uploaded and embed_ok:
            with st.spinner("Đang detect khuôn mặt & trích embedding..."):
                db_embs, db_labels, db_imgs = get_embeddings()
                mtcnn, resnet, device = get_models()
                pil_img = Image.open(uploaded).convert("RGB")
                query_emb, aligned_tensor, face_count, face_prob = embed_uploaded(
                    pil_img, mtcnn, resnet, device
                )

            if query_emb is None:
                st.error("❌ Không phát hiện được khuôn mặt trong ảnh!")
                st.info(
                    "💡 Gợi ý: hãy dùng ảnh có khuôn mặt rõ ràng, đủ sáng, không bị che quá nhiều "
                    "và khuôn mặt nên chiếm một phần đáng kể trong ảnh."
                )
            else:
                if face_count > 1:
                    st.warning(
                        f"Ảnh có {face_count} khuôn mặt. Hệ thống chọn khuôn mặt có xác suất phát hiện cao nhất "
                        f"để crop-align và truy vấn."
                    )
                else:
                    st.success("✅ Đã phát hiện 1 khuôn mặt và xử lý thành công!")

                preview_cols = st.columns([1, 2])
                with preview_cols[0]:
                    st.image(
                        tensor_to_display_image(aligned_tensor),
                        caption="Ảnh sau crop-align đưa vào FaceNet",
                        use_container_width=True
                    )
                with preview_cols[1]:
                    prob_text = f"{face_prob:.3f}" if face_prob is not None else "không xác định"
                    st.markdown(
                        f"<div class='explain-box'>"
                        f"<b>Kiểm tra tiền xử lý ảnh truy vấn</b><br>"
                        f"Số khuôn mặt phát hiện được: <b>{face_count}</b><br>"
                        f"Xác suất phát hiện của khuôn mặt được chọn: <b>{prob_text}</b><br>"
                        f"Ảnh bên trái là khuôn mặt sau khi MTCNN phát hiện landmark, căn chỉnh/crop và resize về 160x160. "
                        f"Đây chính là ảnh được đưa vào FaceNet để sinh embedding 512 chiều."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                top_labels, top_scores, top_imgs, top_idx = find_top_k(
                    query_emb, db_embs, db_labels, db_imgs, top_k
                )

                best_css, best_label, best_detail = sim_explanation(top_scores[0])
                st.markdown(
                    f"<div class='explain-box'>"
                    f"<b>Kết quả gần nhất:</b> {top_labels[0]} "
                    f"<span class='sim-badge {best_css}'>{top_scores[0]:.3f} · {best_label}</span><br>"
                    f"<b>Ngưỡng tham chiếu EER:</b> {EER_THRESHOLD:.4f}. "
                    f"{best_detail}<br>"
                    f"<span style='color:#8b92a5'>Cosine similarity càng cao thì vector embedding của hai khuôn mặt càng gần nhau, "
                    f"vì vậy mức tương đồng càng lớn.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown(f"#### Top-{top_k} kết quả")
                cols = st.columns(top_k)

                for i, (col, lbl, sc, img) in enumerate(
                    zip(cols, top_labels, top_scores, top_imgs)
                ):
                    css_cls, tag = sim_color_class(sc)
                    # Hien thi anh
                    if img.max() <= 1.0:
                        display_img = (img * 255).astype(np.uint8)
                    else:
                        display_img = img.astype(np.uint8)

                    with col:
                        st.image(display_img, use_container_width=True)
                        rank_icon = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
                        st.markdown(
                            f"<div style='text-align:center'>"
                            f"<b>{rank_icon} {lbl}</b><br>"
                            f"<span class='sim-badge {css_cls}'>{sc:.3f} · {tag}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Bar chart similarity
                st.markdown("---")
                st.markdown("**Biểu đồ Cosine Similarity**")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                fig.patch.set_facecolor("#1a1f2e")
                ax.set_facecolor("#1a1f2e")
                colors = [
                    "#4ade80" if s >= 0.75 else "#60a5fa" if s >= 0.60 else "#fbbf24" if s >= EER_THRESHOLD else "#facc15"
                    for s in top_scores
                ]
                short_labels = [l[:18] + ".." if len(l) > 18 else l for l in top_labels]
                bars = ax.barh(range(len(top_scores)), top_scores, color=colors, height=0.5)
                ax.set_yticks(range(len(top_scores)))
                ax.set_yticklabels([f"#{i+1} {l}" for i, l in enumerate(short_labels)],
                                   color="white", fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Cosine Similarity", color="#8b92a5")
                ax.tick_params(colors="#8b92a5")
                ax.axvline(0.75, color="#4ade80", lw=1, ls="--", alpha=0.5, label="Rất giống >=0.75")
                ax.axvline(0.60, color="#60a5fa", lw=1, ls="--", alpha=0.5, label="Khá giống >=0.60")
                ax.axvline(EER_THRESHOLD, color="#fbbf24", lw=1, ls="--", alpha=0.5, label=f"EER {EER_THRESHOLD:.4f}")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                ax.legend(fontsize=7, labelcolor="white", facecolor="#1a1f2e",
                          edgecolor="#2a2d3e")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        elif not embed_ok:
            st.warning("⚠️ Chưa có embeddings. Hãy chạy bước embed trước.")
        else:
            st.info("👆 Upload ảnh ở cột bên trái để bắt đầu.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Dataset Info
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Thống kê Dataset")

    if not embed_ok:
        st.warning("Chưa có embeddings. Hãy chạy bước embed.")
    else:
        db_embs, db_labels, db_imgs = get_embeddings()
        unique, counts = np.unique(db_labels, return_counts=True)
        idx_sort = np.argsort(counts)[::-1]

        # Metric boxes
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (f"{len(db_embs):,}", "Tổng số ảnh"),
            (f"{len(unique):,}", "Số người"),
            (f"{db_embs.shape[1]}D", "Embedding size"),
            (f"{counts.max()}", "Ảnh nhiều nhất"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
            col.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-val'>{val}</div>"
                f"<div class='metric-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Bieu do phan phoi
        col_chart, col_top = st.columns([2, 1])
        with col_chart:
            st.markdown("**Phân phối số ảnh theo người (Top 50)**")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor("#1a1f2e")
            ax.set_facecolor("#1a1f2e")
            n_show = min(50, len(unique))
            ax.bar(range(n_show), counts[idx_sort][:n_show],
                   color="#667eea", edgecolor="#0f1117", linewidth=0.3)
            short = [u[:14] + ".." if len(u) > 14 else u for u in unique[idx_sort][:n_show]]
            ax.set_xticks(range(n_show))
            ax.set_xticklabels(short, rotation=75, ha="right", fontsize=6.5, color="white")
            ax.set_ylabel("Số ảnh", color="#8b92a5")
            ax.tick_params(colors="#8b92a5")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


        with col_top:
            st.markdown("**Top 15 người nhiều ảnh nhất**")
            for i in range(min(15, len(unique))):
                name = unique[idx_sort[i]]
                cnt  = counts[idx_sort[i]]
                short_name = name[:22] + ".." if len(name) > 22 else name
                bar_pct = int(cnt / counts.max() * 100)
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#ccc; margin:3px 0;'>"
                    f"<b>#{i+1}</b> {short_name}<br>"
                    f"<div style='background:#2a2d3e;border-radius:4px;height:6px;'>"
                    f"<div style='background:#667eea;width:{bar_pct}%;height:6px;"
                    f"border-radius:4px;'></div></div>"
                    f"<span style='color:#8b92a5;font-size:0.7rem'>{cnt} ảnh</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        # Luoi anh mau
        st.markdown("**Lưới ảnh mẫu ngẫu nhiên**")
        n_sample = 20
        rng = np.random.default_rng(99)
        sample_idx = rng.choice(len(db_imgs), n_sample, replace=False)

        cols_grid = st.columns(10)
        for j, sidx in enumerate(sample_idx):
            img_arr = db_imgs[sidx]
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            with cols_grid[j % 10]:
                st.image(img_arr, use_container_width=True)
                name = db_labels[sidx]
                short = name.split()[-1] if " " in name else name[:10]
                st.markdown(f"<div style='font-size:0.6rem;color:#8b92a5;text-align:center'>{short}</div>",
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Kết quả & Biểu đồ
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Kết quả đã tạo từ Pipeline")
    st.markdown("Các biểu đồ được tạo từ các bước Retrieval, Visualization.")

    # Nhom cac plot theo loai
    plot_groups = {
        "🔍 Retrieval": [
            ("03_retrieval_query1.png", "Query 1 — Top-K Retrieval"),
            ("03_retrieval_query2.png", "Query 2 — Top-K Retrieval"),
            ("03_retrieval_query3.png", "Query 3 — Top-K Retrieval"),
            ("03_similarity_distribution.png", "Phân phối Cosine Similarity"),
        ],
        "📈 ROC Curve": [
            ("03_roc_curve.png", "ROC Curve + AUC (Face Verification)"),
        ],
        "🔬 PCA": [
            ("05_pca_by_person.png",   "PCA 2D — Màu theo người"),
            ("05_pca_by_cluster.png",  "PCA 2D — Màu theo cụm KMeans"),
            ("05_pca_variance.png",    "PCA Explained Variance"),
        ],
        "🌀 t-SNE": [
            ("05_tsne_by_person.png",  "t-SNE 2D — Màu theo người"),
            ("05_tsne_by_cluster.png", "t-SNE 2D — Màu theo cụm"),
        ],
        "🔥 Heatmap": [
            ("05_similarity_heatmap.png", "Cosine Similarity Heatmap"),
        ],
    }

    for group_name, plots in plot_groups.items():
        available = [(f, t) for f, t in plots if result_image_path(f)]
        if not available:
            continue
        st.markdown(f"#### {group_name}")
        cols_p = st.columns(min(len(available), 3))
        for col, (fname, title) in zip(cols_p, available):
            p = result_image_path(fname)
            with col:
                st.image(p, caption=title, use_container_width=True)
        st.markdown("---")

    # Neu chua co gi
    all_plots = [f for grp in plot_groups.values() for f, _ in grp]
    if not any(result_image_path(f) for f in all_plots):
        st.info("Chưa có biểu đồ nào. Hãy chạy pipeline:")
        st.code("python src/03_retrieval.py\npython src/05_visualize.py")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Clustering
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Phân nhóm KMeans")
    st.markdown("Kết quả phân cụm tự động không cần nhãn.")

    # Hien thi cac anh cluster da co
    cluster_plots = [f for f in os.listdir(RESULTS_DIR)
                     if f.startswith("04_") and f.endswith(".png")] \
        if os.path.exists(RESULTS_DIR) else []
    cluster_plots.sort()

    if cluster_plots:
        # Elbow + Silhouette
        elbow = [f for f in cluster_plots if "elbow" in f]
        samples = [f for f in cluster_plots if "cluster_samples" in f]

        if elbow:
            st.markdown("#### Elbow Method + Silhouette Score")
            for f in elbow:
                st.image(os.path.join(RESULTS_DIR, f), use_container_width=True)

        if samples:
            st.markdown("---")
            st.markdown("#### Ảnh mẫu từng cụm")
            num_cols = min(len(samples), 3)
            cols_cl = st.columns(num_cols)
            for col, f in zip(cols_cl * 10, samples):
                with col:
                    k_val = f.replace("04_cluster_samples_k", "").replace(".png", "")
                    st.image(os.path.join(RESULTS_DIR, f),
                             caption=f"K = {k_val}",
                             use_container_width=True)
    else:
        st.info("Chưa có kết quả clustering. Hãy chạy:")
        st.code("python src/04_cluster.py")

    # Thong ke cluster hien tai
    cluster_labels_path = os.path.join(EMBED_DIR, "cluster_labels.npy")
    labels_path         = os.path.join(EMBED_DIR, "labels.npy")

    if os.path.exists(cluster_labels_path) and os.path.exists(labels_path):
        st.markdown("---")
        cluster_labels = np.load(cluster_labels_path, allow_pickle=True)
        true_labels    = np.load(labels_path, allow_pickle=True)

        cluster_ok = len(cluster_labels) == len(true_labels)
        if not cluster_ok:
            st.warning(
                "cluster_labels.npy không khớp với labels.npy. "
                "Hãy chạy lại `python src/04_cluster.py` sau khi embed custom dataset."
            )
        else:
            st.markdown("#### Thống kê cụm")
            unique_clusters = np.unique(cluster_labels)
            k_actual = len(unique_clusters)

            c1, c2 = st.columns(2)
            c1.metric("Số cụm K", k_actual)
            c2.metric("Tổng ảnh", len(cluster_labels))

            # Phan phoi kich thuoc cum
            sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
            fig, ax = plt.subplots(figsize=(8, 2.5))
            fig.patch.set_facecolor("#1a1f2e")
            ax.set_facecolor("#1a1f2e")
            colors = plt.cm.plasma(np.linspace(0.2, 0.85, k_actual))
            ax.bar(unique_clusters, sizes, color=colors)
            ax.set_xlabel("Cluster ID", color="#8b92a5")
            ax.set_ylabel("Số ảnh", color="#8b92a5")
            ax.set_title("Phân phối kích thước cụm", color="white")
            ax.tick_params(colors="#8b92a5")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# =====================================================================
# TAB 5 - Pipeline & Explanation
# =====================================================================
with tab5:
    st.subheader("Báo cáo thuyết trình: Face Similarity Retrieval")
    st.markdown(
        "Nội dung dưới đây được thiết kế như một bản thuyết trình trực tiếp trên web: "
        "đi từ bài toán, kiến thức nền tảng, phương pháp xử lý, kỹ thuật áp dụng, kết quả thực nghiệm "
        "đến các ứng dụng thực tế của hệ thống."
    )

    c1, c2, c3, c4 = st.columns(4)
    summary_metrics = [
        ("31,480", "Ảnh đã embed"),
        ("512D", "Kích thước embedding"),
        ("0.9914", "AUC"),
        ("0.0518", "EER"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], summary_metrics):
        col.markdown(
            f"<div class='metric-box'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("#### 1. Bài toán hệ thống giải quyết")
    st.markdown(
        "<div class='explain-box'>"
        "<b>Mục tiêu của đề tài</b> là xây dựng hệ thống tìm kiếm khuôn mặt tương đồng. "
        "Khi người dùng đưa vào một ảnh bất kỳ có khuôn mặt, hệ thống sẽ phát hiện khuôn mặt, "
        "căn chỉnh ảnh, trích xuất đặc trưng bằng mô hình học sâu, sau đó so sánh với cơ sở dữ liệu "
        "đã được nhúng sẵn để trả về Top-K khuôn mặt giống nhất. "
        "<br><br>"
        "Bài toán này thuộc nhóm <b>Face Similarity Retrieval</b>: hệ thống không nhất thiết phải kết luận "
        "người trong ảnh là ai, mà tập trung tìm những khuôn mặt có đặc trưng gần nhất với ảnh truy vấn. "
        "Điểm quan trọng là hệ thống so sánh trên đặc trưng khuôn mặt đã học được, không so sánh trực tiếp "
        "từng pixel của ảnh."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### 2. Kiến thức nền tảng và thuật ngữ")
    concept_cols = st.columns(2)
    concepts = [
        (
            "Embedding là gì?",
            "Embedding là vector số biểu diễn đặc trưng của một đối tượng. Trong đề tài này, mỗi khuôn mặt "
            "được chuyển thành một vector 512 chiều. Hai khuôn mặt càng giống nhau thì hai vector embedding "
            "càng gần nhau trong không gian đặc trưng."
        ),
        (
            "FaceNet / InceptionResnetV1 là gì?",
            "FaceNet là hướng tiếp cận dùng mạng học sâu để biến ảnh khuôn mặt thành embedding. "
            "Project sử dụng InceptionResnetV1 pretrained trên VGGFace2, tức là mô hình đã được học trước "
            "trên tập khuôn mặt lớn và được dùng để trích xuất đặc trưng thay vì huấn luyện lại từ đầu."
        ),
        (
            "MTCNN là gì?",
            "MTCNN là mô hình phát hiện khuôn mặt nhiều tầng. Nó tìm vị trí khuôn mặt và landmark như mắt, "
            "mũi, miệng. Landmark giúp hệ thống xoay và căn chỉnh mặt trước khi đưa vào FaceNet."
        ),
        (
            "Cosine similarity là gì?",
            "Cosine similarity đo độ giống nhau về hướng giữa hai vector. Với embedding khuôn mặt, điểm càng cao "
            "thì hai khuôn mặt càng có đặc trưng gần nhau. Vì vậy hệ thống dùng cosine similarity để xếp hạng Top-K."
        ),
        (
            "Top-K Retrieval là gì?",
            "Top-K retrieval là quá trình lấy ra K kết quả có điểm tương đồng cao nhất. Ví dụ Top-5 nghĩa là "
            "hệ thống trả về 5 khuôn mặt gần nhất với ảnh truy vấn."
        ),
        (
            "AUC, EER, threshold là gì?",
            "AUC đo khả năng phân biệt cặp cùng người và khác người trên nhiều ngưỡng. EER là điểm mà tỷ lệ nhận sai "
            "và bỏ sót cân bằng nhau. Threshold là ngưỡng quyết định một điểm similarity có đủ cao để xem là tương đồng hay không."
        ),
    ]
    for idx, (title, desc) in enumerate(concepts):
        with concept_cols[idx % 2]:
            st.markdown(
                f"<div class='step-box'><b>{title}</b><br>{desc}</div>",
                unsafe_allow_html=True
            )

    st.markdown("#### 3. Luồng xử lý chính của hệ thống")
    steps = [
        (
            "Bước 1. Chuẩn bị dữ liệu",
            "Ảnh khuôn mặt được tổ chức theo từng người trong dataset. Mỗi thư mục hoặc nhãn đại diện cho một danh tính. "
            "Dữ liệu được ưu tiên theo bối cảnh người Việt để phù hợp hơn với mục tiêu thực tế của đề tài."
        ),
        (
            "Bước 2. Phát hiện khuôn mặt bằng MTCNN",
            "Mỗi ảnh được đưa qua MTCNN để tìm bounding box khuôn mặt và các điểm landmark. Nếu ảnh không có khuôn mặt rõ ràng, "
            "hệ thống bỏ qua ảnh đó để tránh tạo embedding sai."
        ),
        (
            "Bước 3. Căn chỉnh và crop khuôn mặt",
            "Dựa vào landmark hai mắt, hệ thống xoay ảnh về tư thế chuẩn hơn rồi crop khuôn mặt về kích thước phù hợp. "
            "Bước này rất quan trọng vì FaceNet hoạt động ổn định hơn khi khuôn mặt đã được chuẩn hóa góc nhìn."
        ),
        (
            "Bước 4. Trích xuất embedding bằng FaceNet",
            "Ảnh khuôn mặt đã căn chỉnh được đưa vào InceptionResnetV1 để sinh ra vector embedding 512 chiều. "
            "Vector này nén thông tin nhận dạng quan trọng của khuôn mặt thành dạng số."
        ),
        (
            "Bước 5. Lưu embedding để truy vấn nhanh",
            "Thay vì mỗi lần truy vấn lại xử lý toàn bộ dataset, hệ thống lưu sẵn embeddings, labels và ảnh đã crop. "
            "Khi có ảnh mới, chỉ cần embed ảnh truy vấn rồi so sánh với ma trận embedding đã lưu."
        ),
        (
            "Bước 6. So sánh cosine similarity",
            "Embedding của ảnh truy vấn được so sánh với mọi embedding trong dataset. Hệ thống sắp xếp điểm similarity giảm dần "
            "và lấy ra Top-K khuôn mặt giống nhất."
        ),
        (
            "Bước 7. Đánh giá và trực quan hóa",
            "Project đánh giá bằng ROC/AUC, EER, precision, recall, F1; đồng thời dùng PCA và t-SNE để trực quan hóa phân bố embedding. "
            "KMeans được dùng để thử phân cụm không giám sát trên không gian đặc trưng."
        ),
    ]
    for title, desc in steps:
        st.markdown(
            f"<div class='step-box'><b>{title}</b><br>{desc}</div>",
            unsafe_allow_html=True
        )

    st.markdown("#### 4. Cách đọc điểm similarity trong demo")
    st.markdown(
        f"<div class='explain-box'>"
        f"<b>Cosine similarity</b> nằm trong khoảng so sánh độ gần giữa hai vector embedding. "
        f"Trong demo này, hệ thống dùng ngưỡng tham chiếu EER = <b>{EER_THRESHOLD:.4f}</b>. "
        f"Các mức diễn giải trên giao diện gồm: <b>rất giống</b> khi >= 0.75, "
        f"<b>khá giống</b> khi >= 0.60, <b>có tương đồng</b> khi vượt ngưỡng EER, "
        f"và <b>không chắc chắn</b> khi thấp hơn ngưỡng EER."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### 5. Kết quả thực nghiệm và ý nghĩa")
    result_cols = st.columns(2)
    with result_cols[0]:
        st.markdown(
            "<div class='step-box'>"
            "<b>AUC = 0.9914</b><br>"
            "AUC cao cho thấy hệ thống có khả năng phân biệt tốt giữa cặp ảnh cùng người và khác người. "
            "Điều này chứng minh embedding sinh ra bởi FaceNet mang nhiều thông tin nhận dạng hữu ích."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='step-box'>"
            "<b>EER = 0.0518</b><br>"
            "EER càng thấp thì điểm cân bằng giữa nhận sai và bỏ sót càng tốt. Với EER khoảng 5.18%, "
            "hệ thống có chất lượng tương đối ổn cho một bài toán retrieval/demonstration trên dataset thực tế."
            "</div>",
            unsafe_allow_html=True
        )
    with result_cols[1]:
        st.markdown(
            "<div class='step-box'>"
            "<b>Threshold EER = 0.4968</b><br>"
            "Ngưỡng này được dùng làm mốc tham chiếu khi diễn giải kết quả. Tuy nhiên trong demo người dùng vẫn nên xem Top-K "
            "và ảnh trực quan, vì ảnh thật có thể bị ảnh hưởng bởi ánh sáng, góc mặt, biểu cảm, tóc, kính hoặc chất lượng camera."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='step-box'>"
            "<b>PCA, t-SNE, KMeans</b><br>"
            "Các kỹ thuật này không phải bước nhận dạng chính, mà dùng để phân tích không gian embedding: "
            "các ảnh cùng người có xu hướng gần nhau hơn, còn các nhóm khác nhau sẽ phân tách tương đối trong không gian đặc trưng."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("#### 6. Ứng dụng thực tế của tìm kiếm khuôn mặt tương đồng")
    applications = [
        (
            "Ứng dụng hẹn hò và gợi ý gu thẩm mỹ",
            "Người dùng có thể cung cấp một ảnh mẫu thể hiện gu khuôn mặt họ yêu thích. Hệ thống tìm các hồ sơ có đặc trưng khuôn mặt tương đồng, "
            "kết hợp thêm sở thích, vị trí, độ tuổi và hành vi tương tác để gợi ý người phù hợp hơn. Trong sản phẩm thật cần xử lý quyền riêng tư, "
            "đồng ý của người dùng và tránh dùng khuôn mặt như tiêu chí duy nhất."
        ),
        (
            "Tìm kiếm ảnh trong thư viện cá nhân",
            "Người dùng tải một ảnh chân dung lên và hệ thống tìm các ảnh có khuôn mặt tương tự trong album lớn. Ứng dụng phù hợp cho quản lý ảnh gia đình, "
            "ảnh sự kiện, ảnh lớp, ảnh công ty hoặc kho media nội bộ."
        ),
        (
            "Hỗ trợ quản lý sự kiện và điểm danh",
            "Trong hội nghị, lớp học hoặc sự kiện đông người, hệ thống có thể hỗ trợ tìm các ảnh check-in hoặc ảnh camera có khuôn mặt gần giống ảnh đăng ký. "
            "Vai trò phù hợp nhất là hỗ trợ gợi ý để con người xác nhận, không nên thay thế hoàn toàn quyết định cuối."
        ),
        (
            "Tìm kiếm nhân vật trong video hoặc kho ảnh báo chí",
            "Tòa soạn, đội truyền thông hoặc đơn vị sản xuất video có thể dùng retrieval để tìm nhanh các khung hình chứa khuôn mặt tương đồng với ảnh mẫu, "
            "giảm thời gian dò thủ công trong kho dữ liệu lớn."
        ),
        (
            "Gợi ý ảnh đại diện hoặc lọc ảnh trùng gần giống",
            "Các nền tảng mạng xã hội có thể phát hiện nhiều ảnh chân dung gần giống nhau, gợi ý ảnh rõ mặt nhất hoặc gom nhóm ảnh theo cùng một người."
        ),
        (
            "Hỗ trợ an ninh ở mức truy vấn nội bộ",
            "Trong môi trường được cấp phép như doanh nghiệp hoặc khuôn viên riêng, hệ thống có thể tìm nhanh các ảnh tương đồng từ camera hoặc ảnh đăng ký. "
            "Ứng dụng này cần quy trình pháp lý, bảo mật dữ liệu và kiểm soát sai số rất chặt chẽ."
        ),
        (
            "Tìm người trong dữ liệu thất lạc hoặc dữ liệu nhân đạo",
            "Trong các bài toán tìm kiếm ảnh người thân, ảnh hồ sơ hoặc dữ liệu cần đối chiếu, hệ thống có thể trả về danh sách ứng viên gần giống để chuyên viên kiểm tra."
        ),
        (
            "Kiểm tra trùng lặp hồ sơ",
            "Các hệ thống đăng ký thành viên, thẻ ra vào hoặc hồ sơ nội bộ có thể dùng face similarity để phát hiện một người tạo nhiều hồ sơ bằng ảnh khác nhau."
        ),
    ]
    for title, desc in applications:
        st.markdown(
            f"<div class='step-box'><b>{title}</b><br>{desc}</div>",
            unsafe_allow_html=True
        )

    st.markdown("#### 7. Điểm mạnh, giới hạn và hướng phát triển")
    st.markdown(
        "<div class='explain-box'>"
        "<b>Điểm mạnh:</b> pipeline rõ ràng, dùng mô hình pretrained mạnh, có bước căn chỉnh khuôn mặt, "
        "có đánh giá định lượng và có demo trực quan bằng Streamlit/webcam. "
        "<br><br>"
        "<b>Giới hạn:</b> chất lượng phụ thuộc vào dữ liệu, ánh sáng, góc chụp, độ phân giải ảnh và độ đa dạng của khuôn mặt trong dataset. "
        "Hệ thống hiện tập trung vào tìm kiếm tương đồng, chưa phải hệ thống định danh tuyệt đối trong môi trường sản xuất. "
        "<br><br>"
        "<b>Hướng phát triển:</b> mở rộng dataset người Việt, hiệu chỉnh threshold theo từng kịch bản, thêm cơ chế chọn nhiều khuôn mặt trong ảnh, "
        "tối ưu tốc độ truy vấn bằng FAISS/ANN khi dữ liệu lớn, bổ sung kiểm thử bias/fairness và tăng cường bảo vệ quyền riêng tư."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### 8. Gợi ý lời trình bày khi demo")
    st.markdown(
        "- Đầu tiên giới thiệu bài toán: từ một ảnh truy vấn, hệ thống tìm các khuôn mặt tương đồng nhất trong dataset.\n"
        "- Sau đó giải thích pipeline: MTCNN phát hiện và căn chỉnh mặt, FaceNet sinh embedding, cosine similarity xếp hạng Top-K.\n"
        "- Khi demo upload ảnh, chỉ vào điểm similarity và nói rõ điểm càng cao thì vector khuôn mặt càng gần nhau.\n"
        "- Mở tab **Kết quả & Biểu đồ** để trình bày AUC/EER, ROC, PCA/t-SNE và ý nghĩa của các biểu đồ.\n"
        "- Kết thúc bằng ứng dụng thực tế: tìm kiếm ảnh, gợi ý hồ sơ trong app hẹn hò, quản lý sự kiện, kho media, kiểm tra trùng lặp hồ sơ."
    )

