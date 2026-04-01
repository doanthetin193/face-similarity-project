# 🧠 Face Similarity Retrieval System

> **Môn học:** Lập trình Trí tuệ Nhân tạo  
> **Mức độ AI:** Pretrained model local — Representation Learning + Metric Learning + Unsupervised Clustering

---

## Giới thiệu

Hệ thống truy hồi khuôn mặt tương đồng sử dụng mô hình học sâu FaceNet (InceptionResnetV1 pretrained VGGFace2) để biến mỗi khuôn mặt thành một vector 512 chiều (embedding), sau đó xây dựng hệ thống:

- **Top-K Retrieval** — Nhập 1 ảnh → trả về K ảnh giống nhất (cosine similarity)
- **External Query** — Đưa ảnh bất kỳ từ ngoài vào, tìm người giống nhất trong dataset
- **Clustering** — Phân nhóm tự động không cần nhãn (KMeans)
- **Visualization** — PCA + t-SNE để khám phá cấu trúc không gian embedding

---

## Cài đặt

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# 2. Cài thư viện
pip install -r requirements.txt
```

---

## Dataset

Project hỗ trợ **2 chế độ dataset**:

### Chế độ A — LFW (mặc định, tự động tải)

| | |
|---|---|
| **Tên** | LFW (Labeled Faces in the Wild) |
| **Ảnh** | ~3.023 ảnh (người có ≥ 20 ảnh) |
| **Người** | 62 người nổi tiếng phương Tây |
| **Tải** | Tự động qua `scikit-learn` — không cần chuẩn bị gì |

### Chế độ B — Custom Dataset (ảnh tự chuẩn bị)

Cấu trúc thư mục:
```
custom_dataset/
├── Category_1/
│   ├── Nguyen_Van_A/
│   │   ├── anh1.jpg
│   │   └── ...
│   └── Tran_Thi_B/
└── Category_2/
    └── ...
```
> **Dataset đã tích hợp sẵn:**  
> - **Vietnamese Celebrity Faces** (Kaggle) — 8.557 ảnh, 224 người (ca sĩ/diễn viên/hoa hậu VN)  
> - **VN-Celeb** (Kaggle) — 23.105 ảnh, 1.020 người (ID số, không có tên)  
> - **Tổng cộng:** ~31.480 ảnh, 1.244 người sau khi embed (bỏ qua 182 ảnh không detect được mặt)

### Link tải dataset (Kaggle)

- VN-Celeb: https://www.kaggle.com/datasets/dnguyenhoang/vn-celeb
- Vietnamese Celebrity Faces: tìm theo từ khóa `Vietnamese Celebrity Faces` trên Kaggle (link có thể thay đổi theo tài khoản đăng tải)

---

## Chạy Pipeline

### Chế độ A — LFW Dataset

```bash
# Bước 1 — Xem thống kê & ảnh mẫu
python src/01_preprocess.py

# Bước 2 — Trích embedding LFW (AI core) — ~2 phút
python src/02_embed.py
```

### Chế độ B — Custom Dataset

```bash
# Bước 2B — Trích embedding từ custom_dataset/ — ~8-10 phút
python src/02b_embed_custom.py
```

### Phần còn lại — dùng chung cho cả 2 chế độ

```bash
# Bước 3A — Top-K Retrieval (so sánh nội bộ dataset)
python src/03_retrieval.py

# Bước 3B — Clustering (Elbow + KMeans + Silhouette)
python src/04_cluster.py

# Bước 4 — Visualization (PCA + t-SNE + Heatmap)
python src/05_visualize.py

# Hoặc chạy toàn bộ 1 lệnh (chế độ LFW)
python run_pipeline.py
```

### Đưa ảnh từ ngoài vào query

```bash
# Tìm người giống nhất trong dataset với ảnh bất kỳ
python query_external.py "đường/dẫn/ảnh.jpg"
python query_external.py "anh.jpg" --topk 8
```

> Kết quả tự động mở lên và lưu vào `results/query_external_result.png`

---

## Kiến trúc Pipeline

```
Ảnh (LFW hoặc custom_dataset/)
    ↓ MTCNN — detect & crop face (160×160)
    ↓ InceptionResnetV1 (FaceNet pretrained VGGFace2)
    → Embedding 512-dim  →  lưu embeddings.npy
         ├─ Cosine Similarity → Top-K Retrieval (nội bộ)
         ├─ Cosine Similarity → External Query (ảnh ngoài)
         ├─ KMeans → Clustering
         └─ PCA / t-SNE → 2D Visualization
```

---

## Kết quả đầu ra

| File | Mô tả |
|---|---|
| `results/01_sample_faces.png` | Lưới ảnh mẫu dataset |
| `results/01_class_distribution.png` | Biểu đồ phân phối số ảnh |
| `results/03_retrieval_query*.png` | Kết quả Top-K retrieval nội bộ |
| `results/03_similarity_distribution.png` | Phân phối cosine similarity |
| `results/04_elbow_silhouette.png` | Elbow Method + Silhouette |
| `results/04_cluster_samples_k*.png` | Ảnh mẫu mỗi cụm KMeans |
| `results/05_pca_*.png` | PCA 2D scatter |
| `results/05_tsne_*.png` | t-SNE 2D scatter |
| `results/05_similarity_heatmap.png` | Heatmap cosine similarity |
| `results/05_pca_variance.png` | PCA explained variance |
| `results/query_external_result.png` | Kết quả query ảnh từ ngoài |

---

## Thư viện sử dụng

| Thư viện | Vai trò |
|---|---|
| `facenet-pytorch` | MTCNN (face detection) + FaceNet (embedding) |
| `torch` / `torchvision` | Deep Learning backend |
| `scikit-learn` | KMeans, PCA, t-SNE, LFW loader |
| `numpy` | Xử lý vector / matrix |
| `matplotlib` / `seaborn` | Visualize |
| `Pillow` | Xử lý ảnh đầu vào |

---

## Cấu trúc thư mục

```
face_similarity_project/
├── src/
│   ├── utils.py               # Tiện ích chung (path, save/load)
│   ├── 01_preprocess.py       # Load & thống kê LFW dataset
│   ├── 02_embed.py            # Embed LFW (AI core)
│   ├── 02b_embed_custom.py    # Embed custom dataset (Vietnamese Celebrity...)
│   ├── 03_retrieval.py        # Top-K similarity search nội bộ
│   ├── 04_cluster.py          # KMeans clustering
│   └── 05_visualize.py        # PCA + t-SNE + heatmap
├── query_external.py          # Query ảnh từ ngoài vào dataset
├── run_pipeline.py            # Chạy toàn bộ pipeline LFW 1 lệnh
├── custom_dataset/            # Dataset tự chuẩn bị (không push lên git)
├── embeddings/                # embeddings.npy, labels.npy (auto-generated)
├── results/                   # Hình ảnh kết quả (auto-generated)
├── requirements.txt
└── README.md
```
