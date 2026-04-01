# 🧠 Kế Hoạch Dự Án AI: Face Similarity Retrieval System

> **Môn học:** Lập trình Trí tuệ Nhân tạo  
> **Nhóm:** 2 người  
> **Mục tiêu học thuật:** Xây dựng hệ thống truy hồi khuôn mặt tương đồng dựa trên Representation Learning  
> **Mục tiêu dài hạn:** Nền tảng nghiên cứu để phát triển thành ứng dụng hẹn hò (Dating App)

---

## 1. Yêu Cầu Môn Học & Cách Ta Đáp Ứng

| Yêu cầu GV | Cách ta đáp ứng |
|---|---|
| Có yếu tố AI rõ ràng | Dùng Deep Learning model (FaceNet) để học biểu diễn khuôn mặt |
| Dùng thư viện có sẵn | `facenet-pytorch`, `scikit-learn`, `torch`, `streamlit`, `opencv` |
| Dữ liệu đủ lớn, có ý nghĩa | **31.480 ảnh, 1.244 người** (Vietnamese Celebrity + VN-Celeb) |
| Không trùng project cũ | Lĩnh vực hoàn toàn mới: Computer Vision + Embedding |
| Dễ bảo vệ | Pipeline rõ ràng, có Web UI, Webcam demo, ROC/AUC metric |

> [!IMPORTANT]
> **Cấp độ AI:** Cấp 2 — Dùng **pretrained model local** (không gọi API ngoài, không train từ đầu).  
> Đây là mức chuẩn cho môn "Lập trình AI" và cũng là cách 99% production system thực tế hoạt động.

---

## 2. Lựa Chọn Dataset

### ✅ Dataset chính: Vietnamese Celebrity Faces + VN-Celeb

| Dataset | Số ảnh | Số người | Nguồn |
|---|---|---|---|
| Vietnamese Celebrity Faces | 8.557 | 224 | Kaggle |
| VN-Celeb | 23.105 | 1.020 | Kaggle |
| **Tổng (sau embed)** | **31.480** | **1.244** | — |

> [!TIP]
> Sau khi trích embedding xong → lưu `.npy` file → không cần load ảnh gốc lại.  
> **Máy yếu vẫn chạy được hoàn toàn.**

### Dataset phụ: LFW (tự động tải)

```python
from sklearn.datasets import fetch_lfw_people
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
# → Tự tải ~168MB, 3.023 ảnh, 62 người nổi tiếng phương Tây
```

---

## 3. Kiến Trúc Pipeline AI

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Ảnh khuôn mặt                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 1: FACE DETECTION                                  │
│  Thư viện: MTCNN (facenet-pytorch)                        │
│  Input:  Ảnh bất kỳ kích thước                           │
│  Output: Face crop 160×160px, normalized                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 2: EMBEDDING EXTRACTION  ← PHẦN AI CỐT LÕI        │
│  Model:   InceptionResnetV1 pretrained on VGGFace2        │
│  Input:  Face tensor 160×160×3                           │
│  Output: Vector 512 chiều (embedding)                    │
│  Ý nghĩa: Mỗi khuôn mặt = 1 điểm trong không gian 512D  │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────────┐
│  RETRIEVAL       │ │  CLUSTERING  │ │  ROC CURVE + AUC    │
│  Cosine Sim      │ │  KMeans      │ │  Đánh giá hiệu năng │
│  → Top-K faces   │ │  → Phân cụm  │ │  AUC = 0.9880       │
│  → Webcam RT    │ │  tự động     │ │  EER = 0.0681       │
└─────────┬───────┘ └──────┬──────┘ └─────────────────────┘
          │                │
          └────────┬───────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│  VISUALIZATION + WEB UI                                  │
│  PCA / t-SNE: 512D → 2D (scatter plot)                  │
│  Streamlit: Web UI upload ảnh, xem kết quả              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Giải Thích "Phần AI Nằm Ở Đâu"

> [!NOTE]
> Khi GV hỏi *"Phần AI của em là gì khi không train model?"* — đây là câu trả lời chuẩn:

**AI nằm ở 4 tầng:**

1. **Representation Learning** — FaceNet đã học cách biểu diễn khuôn mặt thành vector sao cho mặt giống nhau → gần nhau trong không gian 512D
2. **Metric Learning** — Cosine similarity đo khoảng cách trong embedding space, không phải pixel space
3. **Unsupervised Clustering** — KMeans tự động phát hiện nhóm trong không gian embedding mà không cần nhãn
4. **Dimensionality Reduction** — PCA/t-SNE để phân tích cấu trúc của không gian đặc trưng

**Câu nói khi bảo vệ:**
> *"Chúng em khai thác không gian embedding học bởi mô hình học sâu để xây dựng hệ thống truy hồi tương đồng dựa trên khoảng cách cosine, kết hợp phân cụm không giám sát và giảm chiều dữ liệu để phân tích cấu trúc của không gian đặc trưng khuôn mặt."*

---

## 5. Cấu Trúc Project (Hoàn Chỉnh)

```
face_similarity_project/
│
├── src/
│   ├── utils.py               # Tiện ích chung (path, save/load)
│   ├── 01_preprocess.py       # Load LFW, chuẩn hóa ảnh
│   ├── 02_embed.py            # MTCNN + FaceNet → embeddings (LFW)
│   ├── 02b_embed_custom.py    # MTCNN + FaceNet → embeddings (custom)
│   ├── 03_retrieval.py        # Cosine similarity, Top-K, ROC Curve + AUC
│   ├── 04_cluster.py          # KMeans, Elbow method, Silhouette
│   └── 05_visualize.py        # PCA, t-SNE, heatmap
│
├── app.py                     # ★ Streamlit Web UI (4 tab)
├── webcam_query.py            # ★ Webcam real-time query
├── query_external.py          # Query ảnh từ ngoài (CLI)
├── run_pipeline.py            # Chạy toàn bộ pipeline 1 lệnh
│
├── custom_dataset/            # Dataset ảnh tự chuẩn bị
│   ├── Ca sĩ/
│   ├── Diễn viên/
│   ├── Hoa hậu/
│   └── VN-celeb/
│
├── embeddings/                # Output sau bước embed
│   ├── embeddings.npy         # Shape: (31480, 512)
│   ├── labels.npy             # Shape: (31480,) — tên người
│   ├── images.npy             # Shape: (31480, 62, 62, 3)
│   └── cluster_labels.npy    # Shape: (31480,) — nhãn cụm
│
├── images/                    # Ảnh test cho query_external
├── results/                   # Hình ảnh kết quả (auto-generated)
├── requirements.txt
├── README.md
└── KE_HOACH_DU_AN.md
```

---

## 6. Tech Stack & Thư Viện

| Thư viện | Vai trò | Cài đặt |
|---|---|---|
| `facenet-pytorch` | MTCNN + FaceNet model | `pip install facenet-pytorch` |
| `torch` + `torchvision` | Backend Deep Learning | `pip install torch torchvision` |
| `scikit-learn` | KMeans, PCA, t-SNE, ROC/AUC | `pip install scikit-learn` |
| `numpy` | Xử lý vector/matrix | `pip install numpy` |
| `matplotlib` | Vẽ biểu đồ | `pip install matplotlib` |
| `Pillow` | Xử lý ảnh | `pip install Pillow` |
| `opencv-python` | Webcam real-time | `pip install opencv-python` |
| `streamlit` | Web UI | `pip install streamlit` |

> [!TIP]
> **Không cần GPU.** CPU là đủ. Sau khi có `embeddings.npy`, mọi bước còn lại chạy dưới 1 giây.

---

## 7. Kết Quả Đạt Được

| Kết quả | Giá trị / Mô tả | Đánh giá |
|---|---|---|
| **AUC (ROC Curve)** | **0.9880** | Xuất sắc (gần 1.0) |
| **EER** | 0.0681 | Chỉ nhầm ~6.8% |
| **Threshold tối ưu** | 0.482 | Tự phát hiện qua EER |
| **Dataset** | 31.480 ảnh · 1.244 người VN | Đủ lớn, có ý nghĩa |
| **Top-K Retrieval** | Nhập 1 ảnh → 5 ảnh giống nhất | Cosine similarity |
| **Clustering** | Phân nhóm tự động | Silhouette Score |
| **PCA / t-SNE** | Scatter plot 2D trực quan | Cấu trúc space rõ |
| **Webcam demo** | Real-time, ~15 FPS | Demo trực tiếp khi bảo vệ |
| **Web UI** | 4 tab đầy đủ | Streamlit localhost:8501 |

---

## 8. Cách Chạy Theo Kịch Bản

### Kịch bản A — Chạy pipeline đầy đủ (LFW, 1 lệnh)
```bash
python run_pipeline.py
```

### Kịch bản B — Chạy với custom dataset (Vietnamese Celebrity)
```bash
python src/02b_embed_custom.py   # Embed — chỉ cần làm 1 lần
python src/03_retrieval.py
python src/04_cluster.py
python src/05_visualize.py
```

### Kịch bản C — Đã có embeddings, chỉ xem kết quả
```bash
python src/03_retrieval.py       # Top-K + ROC Curve
python src/04_cluster.py         # Clustering
python src/05_visualize.py       # PCA + t-SNE
```

### Kịch bản D — Demo ấn tượng khi bảo vệ
```bash
streamlit run app.py             # Web UI
python webcam_query.py           # Webcam real-time
python query_external.py "anh.jpg" --topk 5  # Query ảnh bất kỳ
```

---

## 9. Phân Công Nhóm (Gợi Ý)

| Thành viên | Phần việc |
|---|---|
| Thành viên 1 | Bước 1+2: Preprocessing + Embedding extraction |
| Thành viên 2 | Bước 3+4+5: Retrieval + Clustering + Visualization |
| **Cả nhóm** | Web UI + Webcam + ROC + README + Báo cáo |

---

## 10. Câu Hỏi GV Hay Hỏi & Cách Trả Lời

| Câu hỏi | Trả lời chuẩn |
|---|---|
| "AI của em ở đâu khi không train?" | "AI nằm trong representation learning của FaceNet. Chúng em xây dựng hệ thống trên embedding space đó." |
| "Tại sao dùng cosine mà không dùng Euclidean?" | "Cosine đo góc giữa vector, không bị ảnh hưởng bởi magnitude — phù hợp hơn cho embedding space." |
| "Độ chính xác bao nhiêu?" | "AUC = 0.988, EER = 6.8% — đánh giá ở tất cả các ngưỡng threshold bằng ROC Curve." |
| "Tại sao chọn FaceNet không phải DeepFace?" | "FaceNet cho phép kiểm soát trực tiếp embedding, phù hợp nghiên cứu hơn." |
| "Dataset có đủ lớn không?" | "31.480 ảnh thực tế của người Việt Nam — 1.244 người, có ý nghĩa và phù hợp với bài toán." |
| "ROC Curve là gì?" | "Đường cong thể hiện hiệu năng ở mọi threshold. AUC = 0.988 nghĩa là mô hình phân biệt đúng 98.8% trường hợp." |

---

## 11. Hướng Phát Triển Thành Dating App

> [!NOTE]
> Đây là lý do tại sao đề tài này có giá trị thực tế dài hạn.

```
[Project học thuật hiện tại]           [Dating App tương lai]
─────────────────────────────          ───────────────────────────
Custom dataset VN Celebrity   →        Ảnh upload từ user thật
Cosine similarity              →        "Match Score" giữa 2 người
KMeans cluster                 →        Nhóm sở thích ngoại hình
Streamlit Web UI               →        Flutter Mobile App
Webcam real-time               →        Camera on-device
facenet-pytorch                →        TFLite (on-device inference)
```
