# PHIẾU ĐĂNG KÝ ĐỀ TÀI
## Môn học: Lập trình Trí tuệ Nhân tạo

---

## Tên đề tài

**Hệ thống Truy hồi Khuôn mặt Tương đồng ứng dụng Deep Learning**
*(Face Similarity Retrieval System using Deep Learning)*

---

## Thông tin nhóm

| | |
|---|---|
| **Số thành viên** | 3 người |
| **Môn học** | Lập trình Trí tuệ Nhân tạo |

---

## 1. Bối cảnh & Động lực thực hiện

Trong thời đại số, nhu cầu tìm kiếm và nhận diện khuôn mặt ngày càng phổ biến — từ hệ thống bảo mật, mạng xã hội cho đến các ứng dụng giải trí. Một xu hướng đặc biệt đang nổi lên là các **ứng dụng hẹn hò thế hệ mới** (như Tinder, Bumble, hay các app nội địa) đang tích hợp tính năng gợi ý kết bạn dựa trên độ tương đồng ngoại hình — tức là hệ thống tự động phân tích ảnh khuôn mặt và tìm những người có nét tương đồng để gợi ý kết nối.

Nhóm nhận thấy đây là bài toán AI thực tế, có giá trị ứng dụng cao và phù hợp với nền tảng kỹ thuật của môn học. Vì vậy, chúng em xây dựng một hệ thống truy hồi khuôn mặt tương đồng hoàn chỉnh — vừa đạt yêu cầu học thuật, vừa là nền tảng để phát triển thành sản phẩm thực tế trong tương lai.

---

## 2. Mô tả đề tài

Đề tài xây dựng một **hệ thống tìm kiếm khuôn mặt tương đồng** hoạt động theo nguyên lý **Representation Learning**: mỗi khuôn mặt được ánh xạ thành một vector đặc trưng 512 chiều (face embedding) bằng mô hình học sâu **FaceNet (InceptionResnetV1, pretrained trên VGGFace2 — 3.31 triệu danh tính)**. Sau đó, bài toán tìm kiếm được quy về đo khoảng cách trong không gian vector này.

Hệ thống bao gồm đầy đủ các module:

### 🔍 Tìm kiếm tương đồng (Face Retrieval)
Người dùng cung cấp một ảnh bất kỳ chứa khuôn mặt. Hệ thống tự động phát hiện mặt (MTCNN), trích xuất embedding, rồi tính **cosine similarity** với toàn bộ cơ sở dữ liệu để trả về Top-K khuôn mặt giống nhất trong vài mili-giây.

### 🗂️ Phân cụm tự động (Unsupervised Clustering)
Ứng dụng thuật toán **KMeans** để tự động nhóm hàng nghìn khuôn mặt theo đặc trưng ngoại hình mà không cần nhãn — mô phỏng cách một hệ thống có thể tự học phân loại nhóm người dùng theo "kiểu mặt".

### 📈 Đánh giá hiệu năng khoa học (ROC Curve & AUC)
Hệ thống được đánh giá bằng **đường cong ROC** và chỉ số **AUC (Area Under Curve)** — đây là chuẩn đánh giá quốc tế trong bài toán Face Verification. Kết quả đạt được: **AUC = 0.988**, tức là mô hình phân biệt đúng "cùng người / khác người" đến **98.8%** trường hợp.

### 🌐 Giao diện Web trực quan (Streamlit)
Xây dựng ứng dụng web với 4 tab chức năng: upload ảnh và xem kết quả ngay trên trình duyệt, thống kê dataset, xem toàn bộ biểu đồ phân tích, và khám phá phân cụm — không cần cài đặt, chạy trực tiếp trên `localhost`.

### 📷 Nhận diện Webcam thời gian thực
Demo trực tiếp qua camera máy tính: hệ thống phát hiện khuôn mặt, xử lý và hiển thị Top-3 người giống nhất (kèm ảnh thumbnail) **theo thời gian thực**. Màn hình hiển thị ~30 FPS; AI phân tích mỗi 4 frame một lần (FRAME_SKIP=4) để tối ưu hiệu năng CPU.

### 🔬 Trực quan hóa không gian đặc trưng
Dùng **PCA** và **t-SNE** để chiếu không gian embedding 512 chiều xuống 2D, cho thấy cấu trúc tự nhiên của dữ liệu khuôn mặt — một cách thể hiện trực quan rằng AI đã "học" được cách phân biệt người này với người khác.

---

## 3. Dataset

| Nguồn | Số ảnh | Số người |
|---|---|---|
| Vietnamese Celebrity Faces (Kaggle) | ~8.500 | 224 |
| VN-Celeb (Kaggle) | ~23.000 | 1.020 |
| **Tổng cộng** | **~31.500 ảnh** | **1.244 người** |

> Dataset hoàn toàn là **người Việt Nam** — ca sĩ, diễn viên, hoa hậu — mang tính thực tế và gần gũi với người dùng nội địa.

---

## 4. Yếu tố AI trong đề tài

> "Chúng em không chỉ *dùng* AI — chúng em *khai thác* không gian biểu diễn mà AI đã học được để xây dựng các hệ thống thông minh phía trên."

| Thành phần AI | Kỹ thuật |
|---|---|
| Face Detection | MTCNN — Multi-task Cascaded CNN |
| Representation Learning | FaceNet / InceptionResnetV1 (VGGFace2) |
| Metric Learning | Cosine Similarity trong embedding space |
| Unsupervised Learning | KMeans Clustering |
| Dimensionality Reduction | PCA, t-SNE |
| Model Evaluation | ROC Curve, AUC, EER |

---

## 5. Giá trị thực tế & Hướng phát triển

Hệ thống này không chỉ là bài tập học thuật — nó là **nguyên mẫu (prototype) có thể mở rộng** thành:

### 💘 Ứng dụng hẹn hò thông minh
Tính năng "Match theo ngoại hình tương đồng" — gợi ý những người có nét mặt gần giống nhau hoặc với một idol/người nổi tiếng mà bạn thích, tương tự tính năng đang xuất hiện trên các nền tảng hẹn hò quốc tế.

### 🔎 Hệ thống tìm kiếm người nổi tiếng
Người dùng chụp ảnh selfie → hệ thống gợi ý ca sĩ/diễn viên Việt Nam nào có nét mặt giống nhất.

### 🛡️ Xác minh danh tính & Bảo mật
Nền tảng cho các hệ thống xác thực khuôn mặt trong môi trường doanh nghiệp.

```
[Prototype hiện tại]              [Sản phẩm tương lai]
────────────────────              ───────────────────────
Dataset người nổi tiếng  →        Ảnh thật của người dùng
Streamlit Web UI         →        Mobile App (Flutter/React Native)
Webcam demo              →        Camera on-device realtime
facenet-pytorch (PC)     →        TFLite (on-device, offline)
Cosine similarity        →        "Match Score" cá nhân hóa
```

---

## 6. Sản phẩm demo khi bảo vệ

| Demo | Mô tả |
|---|---|
| **Web UI** | Upload ảnh bất kỳ → xem Top-K khuôn mặt giống nhất ngay trên browser |
| **Webcam live** | Bật camera → hệ thống nhận diện và hiển thị kết quả thời gian thực |
| **ROC/AUC chart** | Minh chứng hiệu năng khoa học: AUC = 0.988 |
| **PCA / t-SNE plot** | Trực quan hóa cách AI "hiểu" sự khác biệt giữa các khuôn mặt |
| **Clustering** | Hệ thống tự phân nhóm hàng nghìn khuôn mặt mà không cần nhãn |

---

## 7. Tóm tắt

Đề tài kết hợp nhiều kỹ thuật AI hiện đại (Deep Learning, Metric Learning, Unsupervised Clustering, Dimensionality Reduction) trên bộ dữ liệu người Việt Nam thực tế, có giao diện web và demo trực quan, đồng thời mang định hướng ứng dụng rõ ràng trong lĩnh vực đang phát triển mạnh là **AI-powered social & dating applications**.
