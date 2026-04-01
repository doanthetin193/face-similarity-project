# 📖 Hướng Dẫn Chạy Project — Face Similarity Retrieval System

> **Đọc phần này trước:**  
> Mọi lệnh đều phải chạy trong thư mục gốc của project và **kích hoạt môi trường ảo (.venv) trước**.

---

## ⚡ Bước bắt buộc trước khi làm bất cứ gì

```powershell
# Mở terminal, cd vào thư mục project
cd D:\AP\face_similarity_project

# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Dấu hiệu thành công: terminal hiện (.venv) ở đầu dòng
# (.venv) PS D:\AP\face_similarity_project>
```

---

## 🗺️ Sơ đồ tổng quan

```
Lần đầu chạy?
    │
    ├── Muốn dùng LFW (dataset Tây, tự tải, đơn giản)?
    │       └── Kịch bản A
    │
    └── Muốn dùng dataset Việt Nam (custom_dataset/)?
            └── Kịch bản B

Đã embed rồi (có embeddings.npy)?
    │
    ├── Xem kết quả phân tích?    → Kịch bản C
    ├── Chạy Web UI (Streamlit)?  → Kịch bản D
    ├── Chạy Webcam real-time?    → Kịch bản E
    └── Query ảnh bất kỳ (CLI)?  → Kịch bản F
```

---

## 🅰️ Kịch bản A — Lần đầu, dùng LFW Dataset (đơn giản nhất)

> Dataset LFW (~168MB) tự tải về, không cần chuẩn bị gì thêm.

```powershell
# Chạy toàn bộ pipeline 1 lệnh (khuyến nghị lần đầu)
python run_pipeline.py
```

Pipeline sẽ tự chạy tuần tự 5 bước, mất khoảng **10–15 phút**:

| Bước | Script | Thời gian | Mô tả |
|---|---|---|---|
| 1 | `01_preprocess.py` | ~2 phút | Tải LFW, vẽ thống kê |
| 2 | `02_embed.py` | ~5 phút | Embed ảnh → `embeddings.npy` |
| 3 | `03_retrieval.py` | ~2 phút | Top-K retrieval + ROC Curve |
| 4 | `04_cluster.py` | ~2 phút | KMeans clustering |
| 5 | `05_visualize.py` | ~2 phút | PCA + t-SNE + Heatmap |

✅ Kết quả lưu vào thư mục `results/`

---

## 🅱️ Kịch bản B — Lần đầu, dùng Dataset Việt Nam (Vietnamese Celebrity)

> Yêu cầu: thư mục `custom_dataset/` đã có ảnh (Ca sĩ, Diễn viên, Hoa hậu, VN-celeb).

```powershell
# Bước duy nhất cần làm (mất ~10–15 phút, chỉ làm 1 lần)
python src/02b_embed_custom.py
```

Khi chạy xong, terminal sẽ hiện:
```
[OK] Hoan tat embedding!
     Anh da embed : 31,480
     Anh bo qua  : 182  (khong detect duoc mat)
```

✅ File `embeddings/embeddings.npy`, `labels.npy`, `images.npy` đã được tạo.  
➡️ Tiếp tục theo **Kịch bản C** bên dưới.

---

## 🅲 Kịch bản C — Đã embed rồi, chạy phân tích đầy đủ

> Dùng khi đã có `embeddings.npy` và muốn tạo lại / xem toàn bộ kết quả.

```powershell
# Bước 3 — Top-K Retrieval + ROC Curve + AUC
python src/03_retrieval.py

# Bước 4 — KMeans Clustering (Elbow + Silhouette)
python src/04_cluster.py

# Bước 5 — PCA + t-SNE + Heatmap
python src/05_visualize.py
```

> Mỗi script chạy độc lập, chạy bước nào ra kết quả bước đó.  
> Nếu chỉ muốn xem ROC: chỉ cần chạy `03_retrieval.py`.  
> Nếu chỉ muốn clustering: chỉ cần chạy `04_cluster.py`.

✅ Kết quả lưu vào `results/` (xem bảng bên dưới).

---

## 🅳 Kịch bản D — Chạy Web UI (Streamlit)

> Giao diện web đầy đủ: upload ảnh, xem biểu đồ, thống kê dataset.

```powershell
streamlit run app.py
```

Mở trình duyệt vào: **http://localhost:8501**

| Tab | Chức năng |
|---|---|
| 🔍 Query Upload | Upload ảnh → tìm Top-K khuôn mặt giống nhất |
| 📊 Dataset Info | Thống kê dataset, lưới ảnh mẫu |
| 📈 Kết quả & Biểu đồ | Xem ROC, PCA, t-SNE, Retrieval results |
| 🗂️ Clustering | Xem phân nhóm KMeans |

> ⚠️ Lần đầu mở app sẽ mất ~20–30 giây để load model FaceNet.  
> Lần sau query rất nhanh vì model và embeddings được cache lại.

**Tắt Streamlit:** Nhấn `Ctrl+C` trong terminal.

---

## 🅴 Kịch bản E — Chạy Webcam Real-time

> Nhận diện khuôn mặt trực tiếp qua camera, hiển thị Top-K giống nhất theo thời gian thực.

```powershell
# Mặc định: Top-3, camera 0 (camera tích hợp)
python webcam_query.py

# Tuỳ chọn thêm:
python webcam_query.py --topk 5      # Hiện Top-5 kết quả
python webcam_query.py --camera 1    # Dùng camera ngoài (USB)
python webcam_query.py --topk 3 --camera 0   # Gộp cả 2
```

**Phím tắt trong cửa sổ webcam:**

| Phím | Chức năng |
|---|---|
| `Q` hoặc `ESC` | Thoát |
| `S` | Chụp snapshot → lưu `results/webcam_snapshot.png` |
| `SPACE` | Tạm dừng / tiếp tục nhận diện |

> ⚠️ Lần đầu chạy mất ~20–30 giây để load model (sau đó ngay lập tức).  
> ⚠️ Cần màn hình / GUI — không chạy được qua SSH remote.

---

## 🅵 Kịch bản F — Query ảnh bất kỳ từ ngoài (CLI)

> Tìm người giống nhất với ảnh bạn cung cấp, không cần mở web hay webcam.

```powershell
# Cơ bản — tìm Top-5 mặc định
python query_external.py "images/haoo.jpg"

# Chỉ định Top-K
python query_external.py "C:/Users/ban/Pictures/selfie.jpg" --topk 8

# Dùng đường dẫn tương đối
python query_external.py "anh.jpg" --topk 3
```

Kết quả:
- In ra terminal: tên + similarity score + nhận xét (Match / Tương đồng / Ít giống)
- Lưu ảnh kết quả: `results/query_external_result.png` (tự mở lên xem)

---

## 📁 Bảng kết quả đầu ra

| File | Tạo từ script | Mô tả |
|---|---|---|
| `results/01_sample_faces.png` | `01_preprocess.py` | Lưới ảnh mẫu |
| `results/01_class_distribution.png` | `01_preprocess.py` | Phân phối số ảnh/người |
| `results/03_retrieval_query1~3.png` | `03_retrieval.py` | Ảnh Top-K retrieval mẫu |
| `results/03_similarity_distribution.png` | `03_retrieval.py` | Phân phối cosine similarity |
| `results/03_roc_curve.png` | `03_retrieval.py` | **ROC Curve + AUC** |
| `results/04_elbow_silhouette.png` | `04_cluster.py` | Elbow + Silhouette |
| `results/04_cluster_samples_k*.png` | `04_cluster.py` | Ảnh mẫu từng cụm |
| `results/05_pca_by_person.png` | `05_visualize.py` | PCA màu theo người |
| `results/05_pca_by_cluster.png` | `05_visualize.py` | PCA màu theo cụm |
| `results/05_tsne_by_person.png` | `05_visualize.py` | t-SNE màu theo người |
| `results/05_tsne_by_cluster.png` | `05_visualize.py` | t-SNE màu theo cụm |
| `results/05_similarity_heatmap.png` | `05_visualize.py` | Heatmap cosine similarity |
| `results/query_external_result.png` | `query_external.py` | Kết quả query CLI |
| `results/webcam_snapshot.png` | `webcam_query.py` | Ảnh chụp từ webcam |

---

## ❓ Xử lý lỗi thường gặp

### Lỗi 1: `Chưa có file embeddings. Hãy chạy 02_embed.py trước`
```
→ Chưa embed. Chạy: python src/02b_embed_custom.py
```

### Lỗi 2: `Không mở được camera 0`
```
→ Thử camera khác: python webcam_query.py --camera 1
→ Kiểm tra camera có đang bị app khác dùng không (Zoom, Teams...)
```

### Lỗi 3: `Không phát hiện được khuôn mặt trong ảnh`
```
→ Ảnh cần: khuôn mặt rõ ràng, nhìn thẳng, đủ sáng
→ Khuôn mặt chiếm ít nhất 1/4 diện tích ảnh
```

### Lỗi 4: `UnicodeEncodeError` khi chạy script
```
→ Chạy với: $env:PYTHONIOENCODING="utf-8"; python src/ten_script.py
```

### Lỗi 5: Streamlit hiện `No module named 'streamlit'`
```
→ Đảm bảo đã kích hoạt venv: .venv\Scripts\activate
→ Hoặc cài lại: pip install streamlit
```

---

## 🔄 Tóm tắt nhanh (Cheat Sheet)

```powershell
# === SETUP (chỉ làm 1 lần) ===
.venv\Scripts\activate
python src/02b_embed_custom.py          # Embed dataset VN (~15 phút)

# === PHÂN TÍCH (mỗi khi cần) ===
python src/03_retrieval.py              # Retrieval + ROC Curve
python src/04_cluster.py               # Clustering
python src/05_visualize.py             # PCA + t-SNE

# === DEMO ===
streamlit run app.py                   # Web UI → http://localhost:8501
python webcam_query.py                 # Webcam real-time
python query_external.py "anh.jpg"    # Query ảnh bất kỳ

# === PIPELINE ĐẦY ĐỦ LFW (1 lệnh) ===
python run_pipeline.py
```
