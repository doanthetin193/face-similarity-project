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
.sim-low    { background: #2a2a1a; color: #facc15; border: 1px solid #eab308; }

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


def embed_uploaded(pil_img, mtcnn, resnet, device):
    """Upload anh -> detect/align/crop -> FaceNet embed -> (512,)"""
    tensor = align_and_crop_face(pil_img, mtcnn)
    if tensor is None:
        return None
    with torch.no_grad():
        emb = resnet(tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy()[0]


def find_top_k(query_emb, db_embs, db_labels, db_imgs, k):
    scores     = cosine_similarity(query_emb.reshape(1, -1), db_embs)[0]
    top_idx    = np.argsort(scores)[::-1][:k]
    return db_labels[top_idx], scores[top_idx], db_imgs[top_idx], top_idx


def sim_color_class(score):
    if score >= 0.65:  return "sim-high",   "Match"
    if score >= 0.50:  return "sim-medium", "Similar"
    return "sim-low", "Low"


def result_image_path(name):
    p = os.path.join(RESULTS_DIR, name)
    return p if os.path.exists(p) else None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Face Similarity")
    st.markdown("---")

    # Trang thai dataset
    embed_ok = os.path.exists(os.path.join(EMBED_DIR, "embeddings.npy"))
    if embed_ok:
        st.success("✅ Embeddings san sang")
    else:
        st.error("❌ Chua co embeddings.npy")
        st.info("Chay:\n```\npython src/02b_embed_custom.py\n```")

    st.markdown("---")
    st.markdown("**Phim tat Webcam**")
    st.markdown("`Q/ESC` Thoat  \n`S` Chup anh  \n`SPACE` Tam dung")
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Query Upload",
    "📊  Dataset Info",
    "📈  Kết quả & Biểu đồ",
    "🗂️  Clustering",
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
                query_emb = embed_uploaded(pil_img, mtcnn, resnet, device)

            if query_emb is None:
                st.error("❌ Không phát hiện được khuôn mặt trong ảnh!")
                st.info("💡 Gợi ý: Ảnh cần có khuôn mặt rõ ràng, nhìn thẳng, đủ sáng.")
            else:
                st.success("✅ Đã detect khuôn mặt thành công!")

                top_labels, top_scores, top_imgs, top_idx = find_top_k(
                    query_emb, db_embs, db_labels, db_imgs, top_k
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
                colors = ["#4ade80" if s >= 0.65 else "#60a5fa" if s >= 0.50 else "#facc15"
                          for s in top_scores]
                short_labels = [l[:18] + ".." if len(l) > 18 else l for l in top_labels]
                bars = ax.barh(range(len(top_scores)), top_scores, color=colors, height=0.5)
                ax.set_yticks(range(len(top_scores)))
                ax.set_yticklabels([f"#{i+1} {l}" for i, l in enumerate(short_labels)],
                                   color="white", fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Cosine Similarity", color="#8b92a5")
                ax.tick_params(colors="#8b92a5")
                ax.axvline(0.65, color="#4ade80", lw=1, ls="--", alpha=0.5, label="Match ≥0.65")
                ax.axvline(0.50, color="#60a5fa", lw=1, ls="--", alpha=0.5, label="Similar ≥0.50")
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
