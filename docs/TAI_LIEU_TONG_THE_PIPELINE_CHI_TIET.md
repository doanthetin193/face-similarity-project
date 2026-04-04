# Tài liệu tổng thể — Face Similarity Retrieval System
## Giải thích từng dòng logic, từng bước pipeline, cho người mới hoàn toàn

> **Mục tiêu của tài liệu này:**  
> Đọc xong tài liệu này, bạn có thể giải thích được toàn bộ project trước giảng viên,  
> hiểu tại sao mỗi bước làm vậy, và không bị "tắt điện" khi bị hỏi bất kỳ chi tiết nào.


---

## MỤC LỤC

1. [Project này làm gì — giải thích trong 2 phút](#1-project-này-làm-gì)
2. [Các khái niệm cốt lõi — hiểu trước khi đọc code](#2-các-khái-niệm-cốt-lõi)
3. [Sơ đồ pipeline tổng quát](#3-sơ-đồ-pipeline-tổng-quát)
4. [Hai luồng dữ liệu — LFW và Custom](#4-hai-luồng-dữ-liệu)
5. [src/utils.py — Nền tảng hạ tầng](#5-srcutilspy)
6. [src/01_preprocess.py — Khám phá dữ liệu LFW](#6-src01_preprocesspy)
7. [src/02_embed.py — Nhúng embedding từ LFW](#7-src02_embedpy)
8. [src/02b_embed_custom.py — Nhúng embedding từ dataset VN](#8-src02b_embed_custompy)
9. [src/03_retrieval.py — Tìm ảnh giống nhất + đánh giá](#9-src03_retrievalpy)
10. [src/04_cluster.py — Phân cụm tự động](#10-src04_clusterpy)
11. [src/05_visualize.py — Trực quan hóa không gian embedding](#11-src05_visualizepy)
12. [query_external.py — Tra cứu từ ảnh ngoài (CLI)](#12-query_externalpy)
13. [webcam_query.py — Nhận diện realtime qua webcam](#13-webcam_querypy)
14. [app.py — Giao diện Web Streamlit](#14-apppy)
15. [run_pipeline.py — Chạy tự động toàn bộ pipeline](#15-run_pipelinepy)
16. [Hiểu các số liệu kết quả](#16-hiểu-các-số-liệu-kết-quả)
17. [Hạn chế hiện tại và hướng phát triển](#17-hạn-chế-và-hướng-phát-triển)
18. [Câu hỏi bảo vệ và trả lời mẫu](#18-câu-hỏi-bảo-vệ)
19. [Checklist tự kiểm tra](#19-checklist-tự-kiểm-tra)

---

## 1. Project này làm gì

### Tưởng tượng bài toán thực tế

Bạn chụp một tấm ảnh selfie. Project sẽ hỏi cơ sở dữ liệu của mình:
**"Trong kho 31.480 ảnh người nổi tiếng VN này, ai có khuôn mặt giống với ảnh này nhất?"**
Rồi trả về Top-5 khuôn mặt giống nhất kèm điểm tương đồng.

Đây gọi là **Face Similarity Retrieval** — tìm kiếm khuôn mặt theo độ tương đồng.

### Tại sao KHÔNG dùng cách truyền thống

Cách cũ (classifier): Dạy máy nhận ra từng người cụ thể.
- Vấn đề: Dataset có 1.244 người → phải dạy 1.244 nhãn riêng. Thêm người mới → phải train lại từ đầu. Rất tốn thời gian và không thực tế.

Cách project này (embedding + similarity):
- Thay vì hỏi "đây là ai?", hỏi "ảnh này giống ảnh kia bao nhiêu?"
- Thêm người mới vào database → chỉ cần nhúng embedding một lần, không cần train lại gì cả
- Đây là cách mà hầu hết hệ thống thực tế (Face ID, Facebook tag, v.v.) hoạt động

### Dataset của project

- **Vietnamese Celebrity Faces** (Kaggle): ~8.557 ảnh, 224 người VN (ca sĩ, diễn viên, hoa hậu)
- **VN-Celeb** (Kaggle): ~23.105 ảnh, 1.020 người VN (ID dạng số)
- **Sau khi embed**: 31.480 ảnh từ 1.244 người (bỏ qua 182 ảnh MTCNN không detect được mặt)
- **LFW** (bộ phụ, tự tải): ~3.023 ảnh, 62 người Tây nổi tiếng — dùng để test nhanh

---

## 2. Các khái niệm cốt lõi

Đọc phần này kỹ — sau khi hiểu, toàn bộ code còn lại sẽ rất dễ.

### 2.1. Embedding là gì?

**Embedding** = biến ảnh khuôn mặt thành một danh sách 512 con số.

Ví dụ dễ hiểu:
Giả sử ta mô tả khuôn mặt bằng 3 đặc trưng: [độ rộng mắt, chiều cao mũi, độ dài hàm].
- Bảo Anh: `[0.8, 0.5, 0.7]`
- Ảnh khác của Bảo Anh: `[0.79, 0.51, 0.71]` → gần giống
- Mỹ Tâm: `[-0.3, 0.9, 0.2]` → rất khác

FaceNet làm điều tương tự nhưng dùng 512 đặc trưng phức tạp hơn nhiều (khoảng cách giữa các điểm trên mặt, kết cấu da, hình dạng mắt mũi miệng...). Không phải do con người thiết kế — model tự học từ 3.31 triệu ảnh khuôn mặt.

**Kết quả lưu vào file:**
```
embeddings.npy  →  shape (31480, 512)  — 31480 ảnh, mỗi ảnh là 512 số
labels.npy      →  shape (31480,)      — tên người tương ứng mỗi ảnh
images.npy      →  shape (31480, 62, 62, 3)  — ảnh thumbnail nhỏ để hiển thị
```

### 2.2. Cosine Similarity là gì?

Sau khi có 2 embedding (2 danh sách 512 số), cần đo "chúng giống nhau bao nhiêu?".

**Cosine similarity** đo góc giữa 2 vector:
- Kết quả **gần 1.0** → rất giống nhau (cùng hướng)
- Kết quả **gần 0.0** → không liên quan
- Kết quả **gần -1.0** → hoàn toàn khác nhau

Công thức: `sim(A, B) = (A · B) / (||A|| × ||B||)`

**Tại sao dùng cosine mà không dùng khoảng cách Euclidean (khoảng cách thẳng)?**
FaceNet đã chuẩn hóa mỗi embedding về độ dài bằng 1 (L2-normalize). Khi đó, cosine similarity và Euclidean distance cho kết quả tương đương — nhưng cosine trực quan hơn vì nằm trong khoảng [-1, 1] dễ diễn giải.

**Ngưỡng thực nghiệm trong project:**
- `sim > 0.8` → Rất giống, rất có thể cùng người
- `sim 0.7–0.8` → Khá giống, tương đồng nhiều nét
- `sim 0.6–0.7` → Tương đồng vừa
- `sim < 0.6` → Ít giống

Project đặt ngưỡng mặc định `SAME_PERSON_THRESH = 0.7` trong `03_retrieval.py`.

### 2.3. MTCNN là gì?

**MTCNN** (Multi-task Cascaded Convolutional Networks) = mạng AI chuyên phát hiện khuôn mặt trong ảnh.

Nó làm 3 việc cùng lúc:
1. Hỏi: "Có mặt người ở đây không?" → vẽ bounding box
2. Hỏi: "Mặt nằm ở tọa độ chính xác nào?"
3. Hỏi: "Mắt, mũi, miệng nằm ở đâu?" → để căn thẳng mặt

Kết quả: ảnh mặt đã crop, kích thước 160×160 pixel, chuẩn hóa về range [-1, 1].

**Tại sao cần MTCNN?**
Nếu đưa cả ảnh chứa nền (tường, cây, áo quần) vào FaceNet → embedding bị nhiễu → tìm kiếm sai. MTCNN đảm bảo chỉ phần mặt mới vào FaceNet.

**Setting trong project** (dùng ở mọi nơi):
```
image_size=160   # Output crop 160×160
margin=20        # Lấy thêm 20px xung quanh mặt (tránh cắt sát quá)
keep_all=False   # Chỉ lấy 1 mặt có confidence cao nhất
post_process=True # Tự normalize output về [-1, 1]
```

### 2.4. FaceNet là gì?

**FaceNet** = mạng AI chuyên biến ảnh mặt → vector 512 số (embedding).

Model cụ thể: `InceptionResnetV1` — kết hợp kiến trúc Inception (học nhiều scale) và ResNet (mạng rất sâu không bị mất thông tin).

**Đã học sẵn từ đâu?** VGGFace2 dataset — 3.31 triệu ảnh, 9.131 người. Project KHÔNG train lại, chỉ dùng trực tiếp (gọi là Transfer Learning / Feature Extraction).

**Tại sao không train lại?**
- Cần hàng triệu ảnh + hàng tuần GPU → quá tốn kém
- Model pretrained đã rất tốt, dùng luôn cho dataset VN vẫn hiệu quả (AUC = 0.988)

### 2.5. Pha Offline và Online

**Pha Offline** (làm một lần, tốn thời gian):
- Chạy MTCNN + FaceNet cho toàn bộ 31.480 ảnh → lưu ra file `.npy`
- Mất ~10–15 phút, nhưng chỉ làm 1 lần duy nhất

**Pha Online** (mỗi lần query, cực nhanh):
- Load sẵn file `.npy` vào RAM
- Nhận ảnh query → embed (vài giây) → cosine với 31.480 embeddings → trả kết quả
- Dưới 5 giây cho toàn bộ quá trình

**Tại sao thiết kế như vậy?** Vì không thể đợi 15 phút mỗi lần user upload ảnh. Làm nặng trước, dùng nhẹ sau — đây là pattern chuẩn của mọi hệ thống retrieval thực tế.

---

## 3. Sơ đồ pipeline tổng quát

```
╔══════════════════════════════════════════════════════════════╗
║                    PHA OFFLINE (chạy 1 lần)                  ║
╚══════════════════════════════════════════════════════════════╝

  custom_dataset/               LFW dataset (tự tải)
  ├── Ca sĩ/                    ├── Ảnh đã crop sẵn
  │   └── ca sĩ Bảo Anh/       └── 3.023 ảnh, 62 người
  │       └── anh1.jpg
  └── VN-celeb/
      └── 1/0.png
           │                              │
           ▼ [02b_embed_custom.py]        ▼ [01+02]
      MTCNN detect face            Resize + Normalize
           │                              │
           └──────────────┬──────────────┘
                          ▼
              InceptionResnetV1 (FaceNet VGGFace2)
                          │
                          ▼
              Vector 512D per image
                          │
                          ▼
              Lưu vào embeddings/
              ├── embeddings.npy  (31480, 512)
              ├── labels.npy     (31480,) — tên người
              └── images.npy     (31480, 62, 62, 3) — thumbnail

╔══════════════════════════════════════════════════════════════╗
║             SAU KHI CÓ EMBEDDINGS — Các bước phân tích      ║
╚══════════════════════════════════════════════════════════════╝

  embeddings.npy
       │
       ├─── [03_retrieval.py] ──→ Top-K query + ROC/AUC/EER
       ├─── [04_cluster.py]  ──→ KMeans clustering → cluster_labels.npy
       └─── [05_visualize.py] ─→ PCA + t-SNE + Heatmap

╔══════════════════════════════════════════════════════════════╗
║                    PHA ONLINE — Query thời gian thực         ║
╚══════════════════════════════════════════════════════════════╝

  Ảnh user ──→ MTCNN ──→ FaceNet ──→ query_emb (512,)
                                           │
  embeddings.npy (31480, 512) ─────────────┤
                                           ▼
                                   Cosine Similarity
                                           │
                                           ▼
                                    Top-K kết quả
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
             query_external.py         app.py               webcam_query.py
             (CLI, lưu PNG)       (Streamlit Web)          (Camera realtime)
```

---

## 4. Hai luồng dữ liệu

### Luồng A — LFW (Dataset người Tây, dùng để test)

**Dùng khi:** Muốn chạy nhanh từ đầu, không cần chuẩn bị dữ liệu.

**Scripts:** `01_preprocess.py` → `02_embed.py` → `03→04→05`

**Đặc điểm quan trọng:**
- LFW tự tải về qua `sklearn.fetch_lfw_people` (~168MB, chờ lần đầu)
- Ảnh LFW **đã được crop sẵn** — chỉ chứa mặt người, không có nền
- Vì vậy `02_embed.py` **KHÔNG dùng MTCNN** để detect, resize thẳng lên 160×160
- Kết quả: ~3.023 ảnh, 62 người (chỉ lấy người có ≥ 20 ảnh)

### Luồng B — Custom Dataset VN (Dataset chính của đồ án)

**Dùng khi:** Muốn chạy với dữ liệu người Việt.

**Scripts:** CHỈ `02b_embed_custom.py` → sau đó `03→04→05` (giống hệt luồng A)

**Đặc điểm quan trọng:**
- Ảnh thô từ Kaggle — chứa cả nền, áo quần, nhiều người trong khung...
- **BẮT BUỘC dùng MTCNN** để detect và crop mặt trước khi embed
- Tự nhận dạng cấu trúc thư mục 1 cấp (`Người/ảnh.jpg`) và 2 cấp (`Danh mục/Người/ảnh.jpg`)
- Ảnh nào MTCNN không detect được mặt → bỏ qua (182 ảnh)

### Sự khác biệt cốt lõi

```
              LFW                    Custom VN
              ───                    ─────────
Ảnh đầu vào  Đã crop sẵn            Ảnh thô nguyên gốc
Bước detect  BỎ QUA                 MTCNN bắt buộc
Script embed 02_embed.py             02b_embed_custom.py
Số người     62 (Tây)               1.244 (Việt Nam)
Số ảnh cuối ~3.023                  ~31.480
```

---

## 5. src/utils.py

**Vai trò:** File nền tảng, không chạy độc lập. Được import bởi hầu hết các file khác.

**3 việc chính nó làm:**

**Việc 1 — Định nghĩa đường dẫn chuẩn:**
```python
ROOT_DIR    = thư mục gốc project (face_similarity_project/)
EMBED_DIR   = ROOT_DIR/embeddings/
RESULTS_DIR = ROOT_DIR/results/
```
Tự tạo 2 thư mục này nếu chưa tồn tại. Vì vậy bạn không cần tự tạo thư mục thủ công.

**Việc 2 — Lưu/tải embeddings:**
- `save_embeddings(embeddings, labels, images)` → ghi 3 file `.npy`
- `load_embeddings(with_images=False)` → đọc lại, trả về tuple

**Việc 3 — Lưu biểu đồ:**
- `save_figure(fig, "ten_file.png")` → lưu vào `results/` với DPI 150

**Tại sao cần file này?** Nếu không có, mỗi file sẽ tự định nghĩa đường dẫn riêng → dễ sai khác nhau → khi di chuyển project vào thư mục khác sẽ bị lỗi. Tập trung vào 1 file duy nhất → dễ sửa.

---

## 6. src/01_preprocess.py

**Vai trò:** Bước 1 trong luồng LFW — tải và khám phá dữ liệu. Không liên quan đến custom dataset.

**Tham số chính:**
```python
MIN_FACES_PER_PERSON = 20   # Chỉ lấy người có ít nhất 20 ảnh
RESIZE_FACTOR = 0.5          # Ảnh LFW gốc khá lớn, resize về 50%
```

**Quy trình:**
1. Gọi `fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)` → tự tải về
2. Ảnh trả về: shape `(N, H, W, 3)`, dtype `float32`, range `[0, 1]` — **đã chuẩn hóa sẵn**
3. Chuyển nhãn số → nhãn tên người: `labels = [target_names[t] for t in lfw.target]`
4. In thống kê: tổng ảnh, số người, top-10 người nhiều ảnh nhất
5. Lưu 2 hình: `01_sample_faces.png` (lưới 16 ảnh mẫu) + `01_class_distribution.png` (biểu đồ cột)

**Kết quả mong đợi sau khi chạy:**
- Terminal in ra ~62 người, ~3.023 ảnh
- Thư mục `results/` có 2 file PNG

---

## 7. src/02_embed.py

**Vai trò:** Bước 2 luồng LFW — nhúng embedding. Đây là bước AI cốt lõi đầu tiên.

**Tham số chính:**
```python
DEVICE     = "cuda" if GPU có else "cpu"
BATCH_SIZE = 32    # Xử lý 32 ảnh cùng lúc để tăng tốc
IMAGE_SIZE = 160   # FaceNet yêu cầu đúng 160×160
```

**Quy trình:**

Bước 1 — Load LFW (giống 01, nhưng không vẽ gì):
```python
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)
images = lfw.images  # (N, H, W, 3), float32, [0, 1] — ảnh đã là face crop
```

Bước 2 — Khởi tạo FaceNet:
```python
resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
# classify=False → bỏ lớp phân loại cuối, chỉ lấy 512D embedding
```

Bước 3 — Hàm `preprocess_face_direct` — chuẩn hóa thủ công (vì LFW không qua MTCNN):
```
img_np (float32, [0,1])
  → PIL Image (uint8)
  → resize 160×160
  → /255 → float32 [0,1]
  → (pixel - 0.5) / 0.5  → float32 [-1, 1]   ← FaceNet yêu cầu range này
  → tensor (3, 160, 160)
```

Bước 4 — Loop theo batch:
```
Với mỗi batch 32 ảnh:
  → stack thành tensor (32, 3, 160, 160)
  → resnet(batch) → embedding (32, 512)
  → thu thập lại
```

Bước 5 — Lưu kết quả qua `save_embeddings()`:
```
embeddings.npy → (N, 512)
labels.npy     → (N,) — tên người
images.npy     → (N, H, W, 3) — ảnh gốc LFW (để hiển thị kết quả)
```

**Điểm quan trọng nhất:** File này KHÔNG dùng MTCNN vì LFW đã là face crop. Nếu bạn đưa ảnh thô vào đây → embedding sẽ nhiễu.

---

## 8. src/02b_embed_custom.py

**Vai trò:** Bước 2 luồng Custom — nhúng embedding từ dataset người Việt. Đây là file AI quan trọng nhất trong project.

**Tham số chính:** Giống `02_embed.py` + thêm:
```python
DATASET_DIR   = project_root/custom_dataset/
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
```

### Phần 1 — Quét thư mục (`scan_dataset` + `_scan_one_root`)

File tự nhận dạng 2 kiểu cấu trúc thư mục:

**Kiểu 2 cấp** (Vietnamese Celebrity Faces):
```
custom_dataset/
└── Ca sĩ/           ← Category
    └── ca sĩ Bảo Anh/  ← Tên người (dùng làm nhãn)
        ├── anh1.jpg
        └── anh2.jpg
```
Label = "ca sĩ Bảo Anh"

**Kiểu 1 cấp** (VN-Celeb):
```
custom_dataset/
└── VN-celeb/
    └── 1/           ← Tên người (dùng làm nhãn, dạng số)
        ├── 0.png
        └── 1.png
```
Label = "1"

Logic phân biệt: nếu folder con chứa ảnh trực tiếp → kiểu 1 cấp; nếu chứa subfolder → kiểu 2 cấp.

Kết quả: list các tuple `(đường_dẫn_ảnh, tên_người)`.

### Phần 2 — Khởi tạo model (`build_models`)

```python
mtcnn  = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True)
resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
```

### Phần 3 — Nhúng embedding (`embed_dataset`)

Loop theo batch (32 ảnh/batch):
```
Với mỗi ảnh trong batch:
  → PIL.Image.open().convert("RGB")
  → mtcnn(pil_img) → tensor (3, 160, 160) hoặc None
  
  Nếu None → skipped += 1, bỏ qua ảnh này

Với các ảnh detect được:
  → stack thành tensor (B, 3, 160, 160)
  → resnet(batch) → embeddings (B, 512)
  
  Lưu thumbnail nhỏ để hiển thị:
  → pil.resize((62, 62)) → numpy (62, 62, 3), float32, [0, 1]
```

Cuối cùng:
```
embeddings → (N, 512)
labels_arr → (N,)  — tên người
images_arr → (N, 62, 62, 3)  — thumbnail nhỏ khác với LFW (ở đây lưu 62×62)
skipped    → số ảnh MTCNN không phát hiện được mặt
```

**Lưu ý quan trọng:** File này **ghi đè** toàn bộ `embeddings.npy` hiện có. Nghĩa là chạy luồng Custom sau luồng LFW → embeddings cũ (LFW) bị thay thế hoàn toàn.

---

## 9. src/03_retrieval.py

**Vai trò:** Bước 3 — Chạy truy hồi Top-K và đánh giá chất lượng bằng ROC/AUC/EER.

**Tham số chính:**
```python
TOP_K              = 5     # Trả về 5 kết quả giống nhất
SAME_PERSON_THRESH = 0.7   # Ngưỡng cosine để coi là "cùng người"
```

### Phần 1 — Truy hồi Top-K (`query_top_k`)

```python
query_emb = embeddings[query_idx]       # (512,) — embedding của ảnh cần tìm
scores = cosine_similarity(             # So sánh với toàn bộ database
    query_emb.reshape(1,-1),
    embeddings
)[0]                                    # → (N,) — điểm với từng ảnh

scores[query_idx] = -1.0                # Loại chính nó ra (tránh ảnh giống nhất là chính nó)
top_indices = argsort(scores)[::-1][:5] # Lấy 5 index có điểm cao nhất
```

Script demo: chọn ngẫu nhiên 3 ảnh từ dataset làm query, vẽ kết quả:
- Viền **xanh lá** = kết quả trả về đúng người (label khớp với query)
- Viền **đỏ** = kết quả khác người

Lưu `results/03_retrieval_query1.png`, `query2.png`, `query3.png`.

### Phần 2 — Phân phối Similarity (`plot_similarity_distribution`)

Lấy ngẫu nhiên 20.000 cặp ảnh, vẽ histogram:
- **Cột xanh**: điểm cosine của các cặp **cùng người** → thường cao
- **Cột đỏ**: điểm cosine của các cặp **khác người** → thường thấp
- **Đường đứt**: ngưỡng 0.7

Nếu 2 cột tách biệt rõ → model phân biệt tốt. Nếu overlap nhiều → model kém.

Lưu `results/03_similarity_distribution.png`.

### Phần 3 — Đánh giá với threshold (`evaluate_threshold`)

Với ngưỡng 0.7: duyệt qua tất cả cặp ảnh trong dataset (O(N²)):
- **TP** (True Positive): similarity ≥ 0.7 VÀ thực sự cùng người → đúng
- **FP** (False Positive): similarity ≥ 0.7 VÀ thực ra khác người → sai
- **FN** (False Negative): similarity < 0.7 VÀ thực ra cùng người → bỏ sót
- **TN** (True Negative): similarity < 0.7 VÀ thực ra khác người → đúng

Tính: Precision, Recall, F1, Accuracy.

### Phần 4 — ROC Curve + AUC + EER (`plot_roc_curve`)

**Đây phần quan trọng nhất để bảo vệ đồ án.**

Lấy 50.000 cặp ảnh ngẫu nhiên:
- `y_true` = 1 nếu cùng người, 0 nếu khác người
- `y_score` = điểm cosine similarity của cặp đó

Quét mọi threshold từ -1 → 1:
- Ở mỗi threshold, tính TPR (True Positive Rate) và FPR (False Positive Rate)
- Vẽ đường cong FPR-TPR → đó là **ROC Curve**
- **AUC** = diện tích dưới đường cong → càng gần 1.0 càng tốt

**EER** (Equal Error Rate) = điểm ngưỡng mà tỷ lệ nhận nhầm (FPR) ≈ tỷ lệ bỏ sót (FNR):
- Càng thấp càng tốt
- Threshold tại EER là ngưỡng "cân bằng" nhất

**Kết quả thực tế đạt được:**
- AUC = 0.988 → xuất sắc (model phân biệt đúng 98.8% trường hợp)
- EER = 0.068 → chỉ nhầm 6.8%
- Threshold EER ≈ 0.482

Lưu `results/03_roc_curve.png`.

---

## 10. src/04_cluster.py

**Vai trò:** Bước 4 — Phân cụm tự động (Unsupervised Clustering) trong không gian embedding.

**Tư** **duy cơ bản:** Nếu FaceNet học tốt, các ảnh của cùng một người sẽ nằm gần nhau trong không gian 512D. KMeans sẽ tự phát hiện các "cụm" đó mà không cần biết tên người.

**Tham số chính:**
```python
K_RANGE      = range(3, 16)  # Thử k từ 3 đến 15
RANDOM_STATE = 42
```

### Phần 1 — Elbow Method (`elbow_method`)

Vì không biết nên chia thành bao nhiêu cụm, ta thử tất cả từ 3 đến 15:

Với mỗi k:
1. Chạy `KMeans(n_clusters=k, random_state=42, n_init="auto").fit(embeddings)`
2. Tính **inertia** = tổng khoảng cách từ mỗi điểm đến tâm cụm của nó → nhỏ = cụm gọn
3. Tính **Silhouette Score** (sample 1000 điểm để nhanh hơn):
   - Với mỗi điểm: `s = (khoảng cách đến cụm gần nhất - khoảng cách đến cụm mình) / max(...)`
   - Score cao → điểm nằm đúng cụm, xa cụm khác → clustering tốt

Vẽ 2 đồ thị đồng thời → `results/04_elbow_silhouette.png`:
- **Elbow curve**: inertia giảm dần theo k, tìm điểm "khuỷu tay" (chỗ giảm chậm lại)
- **Silhouette curve**: chọn k có điểm cao nhất

Script tự chọn k tốt nhất theo Silhouette Score và dùng k đó cho bước tiếp.

### Phần 2 — KMeans cuối (`run_kmeans`)

Chạy KMeans với k tốt nhất đã chọn. Kết quả: mảng `cluster_labels` với shape `(N,)` — mỗi ảnh được gán vào cụm nào.

### Phần 3 — Phân tích cụm (`analyse_clusters`)

In top-3 người nhiều ảnh nhất trong mỗi cụm. Ví dụ chạy trên LFW:
```
Cụm 0 [178 ảnh]: Silva(48), Agassi(36), Abbas(28)
Cụm 1 [237 ảnh]: Powell(236), Zemin(1)
Cụm 12 [561 ảnh]: Bush(530), Bremer(16), Daschle(15)
```
Cụm 1 và 12 "thuần" (dominated bởi 1 người) → model phân biệt tốt.
Cụm 0 lẫn 3 người → hoặc họ trông giống nhau, hoặc không đủ dữ liệu để tách.

### Phần 4 — Vẽ ảnh mẫu mỗi cụm (`plot_cluster_samples`)

Lưới k hàng × 5 cột, mỗi hàng là 5 ảnh mẫu của 1 cụm. Lưu `results/04_cluster_samples_k{k}.png`.

### Kết quả quan trọng nhất:

Lưu `embeddings/cluster_labels.npy` — mảng nhãn cụm. File này được dùng tiếp trong `05_visualize.py` để tô màu scatter plot theo cụm.

---

## 11. src/05_visualize.py

**Vai trò:** Bước 5 — Vẽ trực quan hóa để "nhìn thấy" không gian embedding 512D bằng mắt thường.

**Tham số chính:**
```python
TSNE_PERPLEXITY    = 30    # Số "hàng xóm" mỗi điểm quan tâm
TSNE_N_ITER        = 1000  # Số vòng lặp tối ưu
N_DISPLAY_CLASSES  = 15    # Chỉ tô màu 15 người nhiều ảnh nhất
```

### Phần 1 — PCA Variance (`plot_pca_variance`)

Chạy PCA full (512 components), vẽ đồ thị "tích lũy phương sai":
- Trục x: số chiều dùng
- Trục y: bao nhiêu % thông tin giữ được
- Đường đứt đỏ tại 90%, đường đứt cam tại 95%

Kết quả ví dụ: cần ~100 chiều để giữ 90% thông tin → embedding space khá "dense".

Lưu `results/05_pca_variance.png`.

### Phần 2 — PCA 2D Scatter (`reduce_pca` + `plot_scatter`)

Giảm 512D → 2D bằng PCA tuyến tính (nhanh, vài giây):
- Scatter plot: mỗi điểm = 1 ảnh, tô màu theo người (15 màu cho 15 người nhiều ảnh nhất, phần còn lại màu xám)
- Nếu các điểm cùng màu tụ thành đám → embedding có cấu trúc tốt

Lưu `results/05_pca_by_person.png`.

Nếu có `cluster_labels.npy` → vẽ thêm scatter tô màu theo cụm:
Lưu `results/05_pca_by_cluster.png`.

### Phần 3 — t-SNE 2D Scatter (`reduce_tsne`)

Giảm 512D → 2D bằng t-SNE phi tuyến (chậm hơn, 1-2 phút, nhưng đẹp hơn):

Kỹ thuật tối ưu: không chạy t-SNE thẳng trên 512D (rất chậm):
```
512D → PCA → 50D (nhanh, vài giây)
50D  → t-SNE → 2D (chậm hơn nhưng chấp nhận được)
```

Xử lý tương thích sklearn version (sklearn ≥ 1.5 đổi tên tham số):
```python
if sklearn >= 1.5:  tsne_kwargs["max_iter"] = 1000
else:               tsne_kwargs["n_iter"]   = 1000
```

t-SNE tốt hơn PCA vì nó giữ được cấu trúc lân cận (hàng xóm gần nhau ở 512D → gần nhau ở 2D), còn PCA chỉ giữ phương sai toàn cục.

Lưu `results/05_tsne_by_person.png` và `05_tsne_by_cluster.png`.

### Phần 4 — Cosine Similarity Heatmap (`plot_similarity_heatmap`)

Lấy top 10 người nhiều ảnh nhất, tính ma trận cosine N×N, vẽ heatmap:
- Các ô trên đường chéo chính (cùng người) → màu đỏ đậm (similarity cao)
- Các ô ngoài đường chéo (khác người) → màu xanh nhạt (similarity thấp)
- Vạch trắng phân ranh giới giữa các người

Đây là bằng chứng trực quan rằng FaceNet đã học được "ảnh cùng người nên có vector giống nhau".

Lưu `results/05_similarity_heatmap.png`.

---

## 12. query_external.py

**Vai trò:** Cho phép tra cứu từ **ảnh bất kỳ ngoài dataset** qua dòng lệnh (CLI).

**Cách dùng:**
```powershell
python query_external.py "images/selfie.jpg"          # Top-5 mặc định
python query_external.py "anh.jpg" --topk 8           # Top-8
```

**Quy trình chi tiết:**

1. Đọc ảnh đầu vào bằng PIL (`.jpg`, `.png`, `.webp`, ...)
2. MTCNN detect mặt → tensor (3, 160, 160) → `None` nếu không có mặt
3. Nếu `None` → in hướng dẫn (ảnh mờ, không nhìn thẳng, mặt quá nhỏ...) → thoát
4. FaceNet → query_emb (512,)
5. `cosine_similarity(query_emb, all_embeddings)` → scores (N,)
6. Sort giảm dần → Top-K index và score
7. In ra terminal với thanh bar `█` thể hiện độ tương đồng
8. Vẽ hình: ảnh query (face crop) + K ảnh kết quả → lưu `results/query_external_result.png`
9. `os.startfile(result_path)` → tự mở ảnh PNG bằng ứng dụng mặc định của Windows

**Kết quả in trên terminal:**
```
#1  ca sĩ Jang Mi             sim=0.7140  ████████████████  ✓ Khá giống
#2  ca sĩ Thùy Chi            sim=0.7050  ███████████████   ~ Tương đồng
...
```

---

## 13. webcam_query.py

**Vai trò:** Nhận diện khuôn mặt **realtime qua camera**, hiển thị Top-K kết quả.

**Cách dùng:**
```powershell
python webcam_query.py               # Top-3, camera 0
python webcam_query.py --topk 5      # Top-5
python webcam_query.py --camera 1    # Dùng camera USB ngoài
```

**Các thông số quan trọng:**
```python
FRAME_SKIP  = 4     # Chỉ chạy AI mỗi 4 frame (tăng FPS hiển thị)
CONF_THRESH = 0.85  # Chỉ nhận mặt có confidence ≥ 85%
```

**Quy trình mỗi frame:**

```
1. cv2.VideoCapture → đọc frame BGR
2. Nếu frame_count % 4 == 0:
     → PIL(frame RGB) → mtcnn.detect() → boxes, probs
     → Nếu prob < 0.85 → bỏ qua
     → mtcnn(pil) → tensor (3,160,160)
     → resnet(tensor) → embedding (512,)
     → cosine_similarity → Top-K
3. Vẽ bounding box xanh quanh mặt
4. Vẽ tên người có điểm cao nhất lên trên bounding box (tiếng Việt qua PIL)
5. Ghép sidebar bên phải: Top-K với thumbnail + score + progress bar
6. cv2.imshow() → hiển thị
```

**Tại sao dùng FRAME_SKIP=4?**
Chạy AI mỗi frame sẽ tốn kém, webcam sẽ lag. Chỉ chạy AI mỗi 4 frame nhưng vẫn hiển thị bounding box từ lần chạy trước → người dùng thấy mượt.

**Hỗ trợ tiếng Việt:**
OpenCV vẽ chữ không hỗ trợ Unicode. Hack: chuyển frame sang PIL → dùng font Windows (Segoe UI, Arial...) → vẽ → chuyển lại BGR.

**Phím tắt:**
| Phím | Tác dụng |
|---|---|
| `Q` hoặc `ESC` | Thoát |
| `S` | Chụp snapshot → `results/webcam_snapshot.png` |
| `SPACE` | Tạm dừng / Tiếp tục |

---

## 14. app.py

**Vai trò:** Giao diện web đầy đủ chạy trên trình duyệt, không cần cài gì ngoài.

**Cách chạy:**
```powershell
streamlit run app.py
# → Mở http://localhost:8501
```

**Kỹ thuật quan trọng — Cache:**
```python
@st.cache_data      # Cache embeddings (data) — lưu vào RAM
def get_embeddings(): ...

@st.cache_resource  # Cache model (resource) — giữ model trong memory
def get_models(): ...
```
Lần đầu mở app mất ~30 giây load. Lần sau query rất nhanh vì mọi thứ đã cache.

**4 Tab:**

**Tab 1 — Query Upload:**
- Upload ảnh → MTCNN detect → FaceNet embed → cosine → Top-K
- Slider chọn số kết quả (1-10)
- Hiển thị ảnh kết quả + badge màu theo score (xanh ≥0.65, xanh dương ≥0.50, vàng <0.50)
- Biểu đồ ngang (horizontal bar) cosine score của Top-K

**Tab 2 — Dataset Info:**
- 4 metric box: tổng ảnh, số người, kích thước embedding (512D), người nhiều ảnh nhất
- Biểu đồ cột Top 50 người
- Lưới 20 ảnh thumbnail ngẫu nhiên từ dataset

**Tab 3 — Kết quả & Biểu đồ:**
- Hiển thị các PNG đã được tạo từ pipeline (03, 05)
- Nhóm theo: Retrieval, ROC Curve, PCA, t-SNE, Heatmap
- Chỉ hiện nếu file tồn tại trong `results/`

**Tab 4 — Clustering:**
- Hiển thị PNG clustering (04_elbow_silhouette, 04_cluster_samples_k*.png)
- Thống kê cụm: số cụm, phân phối kích thước (biểu đồ cột màu plasma)
- Đọc từ `embeddings/cluster_labels.npy` nếu có

---

## 15. run_pipeline.py

**Vai trò:** Chạy toàn bộ 5 bước pipeline LFW bằng 1 lệnh duy nhất.

**Cách dùng:**
```powershell
python run_pipeline.py
```

**Cách hoạt động:**

Định nghĩa 5 bước theo thứ tự:
```python
STEPS = [
    ("Bước 1 — Preprocess",   "src/01_preprocess.py"),
    ("Bước 2 — Embedding",    "src/02_embed.py"),
    ("Bước 3A — Retrieval",   "src/03_retrieval.py"),
    ("Bước 3B — Clustering",  "src/04_cluster.py"),
    ("Bước 4 — Visualization","src/05_visualize.py"),
]
```

Với mỗi bước: `subprocess.run([python, script], check=False)` → nếu `returncode != 0` → in lỗi → `sys.exit()` ngay (không chạy bước tiếp theo).

**Lưu ý:** File này chỉ chạy luồng LFW. Luồng custom phải chạy thủ công `02b_embed_custom.py` trước.

---

## 16. Hiểu các số liệu kết quả

### AUC = 0.988 — có nghĩa là gì?

AUC (Area Under Curve) = 98.8% nghĩa là: nếu lấy ngẫu nhiên 1 cặp ảnh cùng người và 1 cặp ảnh khác người, model có 98.8% khả năng nhận ra cặp nào giống hơn.

- AUC = 1.0 → hoàn hảo (không thể đạt được trong thực tế)
- AUC = 0.5 → đoán mò không khác gì tung đồng xu
- AUC = 0.988 → xuất sắc

### EER = 0.068 — có nghĩa là gì?

EER (Equal Error Rate) = 6.8% nghĩa là: tại ngưỡng tối ưu (~0.482):
- 6.8% cặp ảnh khác người bị nhận nhầm là cùng người (False Accept)
- 6.8% cặp ảnh cùng người bị bỏ sót (False Reject)

EER càng thấp càng tốt. 6.8% trên dataset người Việt (model pretrained trên dataset Tây) là rất tốt.

### Precision = 99.88% và Recall = 73.76% — có mâu thuẫn không?

Không mâu thuẫn — chúng đo 2 chiều khác nhau với ngưỡng cố định 0.7:

- **Precision 99.88%**: Khi hệ thống bảo "2 ảnh này cùng người" → đúng đến 99.88%.
  → Hệ thống rất cẩn thận, hầu như không nhầm khi khẳng định.

- **Recall 73.76%**: Trong tất cả các cặp ảnh thực sự cùng người, hệ thống chỉ "bắt được" 73.76%.
  → Hệ thống bỏ sót 26.24% cặp cùng người (vì ngưỡng 0.7 khá cao).

Trade-off bình thường: hạ ngưỡng → recall tăng nhưng precision giảm. ROC Curve chính là để hiệu chỉnh trade-off này.

### Silhouette Score — đọc thế nào?

Sau khi chạy `04_cluster.py`, terminal in:
```
k= 5  inertia=234,500  silhouette=0.1234
k= 8  inertia=198,000  silhouette=0.1456  ← cao nhất → chọn k=8
k=10  inertia=187,000  silhouette=0.1398
```

Score 0.1-0.3 trên dữ liệu mặt người là bình thường — vì khuôn mặt không tách bạch hoàn toàn như các bài clustering đơn giản hơn.

---

## 17. Hạn chế và hướng phát triển

### Hạn chế 1 — Tìm kiếm brute-force O(N)

Hiện tại: với mỗi query, tính cosine với **toàn bộ** 31.480 embeddings.
- Với 31K ảnh: vẫn nhanh (dưới 1 giây trên CPU vì NumPy vectorized)
- Với 1 triệu ảnh: sẽ chậm

Nâng cấp: dùng **FAISS** (Facebook AI Similarity Search) — chỉ mục approximate nearest neighbor, query O(log N) thay vì O(N).

### Hạn chế 2 — Model pretrained trên người Tây

VGGFace2 chủ yếu người châu Âu → có thể kém hơn cho người châu Á.

Nâng cấp: Fine-tune thêm với dữ liệu VN hoặc dùng model pretrained trên dữ liệu châu Á (MS-Celeb-1M với người Á châu).

### Hạn chế 3 — Ngưỡng threshold chưa được hiệu chỉnh theo domain

Ngưỡng 0.7 là kinh nghiệm chung. Với dataset VN có thể ngưỡng tối ưu khác.

Nâng cấp: dùng ngưỡng từ EER (≈ 0.482) thay vì 0.7.

### Hạn chế 4 — Embed lại từ đầu khi thêm dữ liệu mới

Thêm 1 người mới → phải chạy lại `02b_embed_custom.py` cho toàn bộ 31K ảnh.

Nâng cấp: tính năng `--append` — chỉ embed folder mới rồi ghép vào `.npy` hiện có.

---

## 18. Câu hỏi bảo vệ

**Hỏi: "AI của project các em nằm ở đâu khi không train model?"**

Trả lời: AI nằm ở 3 tầng:
1. **FaceNet (InceptionResnetV1)** đã học cách biểu diễn khuôn mặt thành vector 512D sao cho mặt giống nhau → vector gần nhau. Đây là Representation Learning — phần AI sâu nhất.
2. **Cosine similarity trong embedding space** không đo pixel, mà đo đặc trưng học được — đây là Metric Learning.
3. **KMeans clustering** tự khám phá cấu trúc nhóm không cần nhãn — đây là Unsupervised Learning.

---

**Hỏi: "Tại sao dùng cosine similarity mà không dùng khoảng cách Euclidean?"**

Trả lời: FaceNet đã L2-normalize embedding — nghĩa là mọi vector đều có độ dài bằng 1, nằm trên mặt cầu 512D. Trong không gian đó, cosine similarity và Euclidean distance cho thứ tự ranking giống nhau. Tuy nhiên cosine trực quan hơn vì nằm trong [-1, 1] dễ đặt ngưỡng, và không bị ảnh hưởng bởi scale (quan trọng với các model khác không normalize).

---

**Hỏi: "AUC = 0.988 nghĩa là gì?"**

Trả lời: Lấy ngẫu nhiên 1 cặp ảnh cùng người và 1 cặp ảnh khác người, model có 98.8% khả năng xác định đúng cặp nào giống hơn. Đây là chỉ số đánh giá chuẩn quốc tế cho bài toán face verification.

---

**Hỏi: "KMeans dùng để làm gì? Liên quan gì đến bài toán retrieval?"**

Trả lời: KMeans không dùng trực tiếp để retrieval. Nó dùng để **phân tích và giải thích** embedding space — chứng minh rằng FaceNet đã học được biểu diễn có cấu trúc thực sự (các ảnh cùng người tự nhiên tụ thành cụm mà không cần cung cấp nhãn). Kết quả clustering cũng được dùng để tô màu scatter plot trong PCA/t-SNE.

---

**Hỏi: "Tại sao cần MTCNN? Sao không đưa ảnh thẳng vào FaceNet?"**

Trả lời: FaceNet chỉ hoạt động tốt khi đầu vào là ảnh mặt đã crop, kích thước 160×160, không có nền. Nếu đưa cả ảnh chụp thông thường (có nền, có áo quần, nhiều người...) → FaceNet sẽ encode cả nền vào vector → embedding nhiễu → retrieval sai. MTCNN là bước tiền xử lý bắt buộc.

---

**Hỏi: "Tại sao LFW không cần MTCNN nhưng custom dataset thì cần?"**

Trả lời: LFW (Labeled Faces in the Wild) là dataset chuẩn học thuật — ảnh đã được xử lý sẵn, chỉ chứa mặt người, không có nền. `02_embed.py` chỉ cần resize về 160×160 và normalize là đủ. Ngược lại, dataset người Việt tải từ Kaggle là ảnh thô — có nền, nhiều người, nhiều góc chụp — nên `02b_embed_custom.py` buộc phải dùng MTCNN detect.

---

## 19. Checklist tự kiểm tra

Trả lời được những câu dưới đây = bạn đã hiểu sâu project.

✅ Embedding là gì? Tại sao ảnh mặt được biến thành 512 con số?

✅ Tại sao 2 ảnh cùng người → embedding gần nhau? Ai đảm bảo điều đó? (FaceNet pretrained Triplet Loss)

✅ Cosine similarity trả về giá trị gì? 0.9 nghĩa là gì? 0.3 nghĩa là gì?

✅ MTCNN làm gì? Khi nào cần, khi nào không cần?

✅ Sự khác biệt giữa `02_embed.py` và `02b_embed_custom.py`?

✅ Embeddings được lưu ở đâu, trong file gì, shape là bao nhiêu?

✅ Tại sao `images.npy` lưu ảnh 62×62 (không phải 160×160)?

✅ Retrieval và Classification khác nhau thế nào? Tại sao không dùng Classification?

✅ ROC Curve vẽ cái gì? AUC = 0.988 nghĩa là gì?

✅ EER là gì? EER = 0.068 tốt hay xấu?

✅ Tại sao t-SNE chạy chậm hơn PCA? Project xử lý điều đó thế nào?

✅ KMeans trong project dùng để làm gì? Nó có liên quan trực tiếp đến retrieval không?

✅ Bước nào chạy offline (1 lần)? Bước nào chạy online (mỗi query)?

✅ `app.py` cache gì? Tại sao cần cache?

✅ `webcam_query.py` chạy AI mỗi mấy frame? Tại sao không mỗi frame?
