# Tài liệu tổng thể cực chi tiết
## Face Similarity Retrieval System

Tài liệu này giải thích toàn bộ hệ thống từ góc nhìn kỹ thuật, học thuật, và triển khai thực tế.
Mục tiêu là giúp bạn hiểu sâu bản chất bài toán, hiểu vì sao chọn từng thuật toán, và biết cách trả lời khi bảo vệ đồ án.

---

## 1. Bài toán thực sự của project là gì

### 1.1. Không phải bài toán phân loại thuần
Bài toán của project là tìm khuôn mặt giống nhất trong một kho ảnh lớn.

Đầu vào:
- Một ảnh query bất kỳ (ảnh ngoài dataset hoặc ảnh từ webcam)

Đầu ra:
- Top K ảnh giống nhất trong database
- Điểm tương đồng cho từng ảnh

Điểm quan trọng:
- Đây là bài toán truy hồi theo độ tương đồng, không phải bài toán phân loại cứng kiểu người A, người B.
- Khi thêm người mới vào dữ liệu, hệ thống vẫn chạy được mà không cần train lại mô hình từ đầu.

### 1.2. Vì sao không dùng classifier truyền thống
Nếu dùng phân loại softmax:
- Mỗi người là một lớp
- Thêm người mới phải huấn luyện lại
- Không phù hợp cho môi trường dữ liệu động

Với cách tiếp cận embedding:
- Mỗi ảnh mặt chuyển thành một vector đặc trưng
- Tìm hàng xóm gần nhất trong không gian vector
- Hệ thống mở rộng linh hoạt, phù hợp thực tế hơn

---

## 2. Ý tưởng thuật toán từ mức high-level

Project kết hợp nhiều khối thuật toán, mỗi khối giải một phần bài toán:

1. MTCNN
- Nhiệm vụ: phát hiện và crop khuôn mặt từ ảnh thô
- Giải quyết: loại bỏ nền nhiễu, tập trung đúng vùng mặt

2. FaceNet InceptionResnetV1 pretrained VGGFace2
- Nhiệm vụ: biến ảnh mặt thành vector 512 chiều
- Giải quyết: mã hóa đặc trưng danh tính thành dạng số để so sánh

3. Cosine Similarity
- Nhiệm vụ: đo độ gần giữa query vector và database vectors
- Giải quyết: xếp hạng Top K ảnh giống nhất

4. Truy hồi kiểu K-nearest neighbors trong embedding space
- Thực thi hiện tại: tính cosine với tất cả vector rồi sort giảm dần
- Bản chất: tìm K hàng xóm gần nhất theo metric similarity

5. KMeans Clustering
- Nhiệm vụ: phân cụm tự động trong không gian embedding
- Giải quyết: kiểm tra xem dữ liệu có cấu trúc cụm tự nhiên hay không

6. PCA và t-SNE
- Nhiệm vụ: giảm chiều từ 512 về 2D để trực quan hóa
- Giải quyết: giúp con người quan sát được cấu trúc embedding space

7. ROC, AUC, EER
- Nhiệm vụ: đánh giá chất lượng phân biệt cùng người khác người
- Giải quyết: có số liệu khoa học thay vì chỉ nhìn kết quả demo

---

## 3. Kiến trúc hệ thống tổng thể

Có 2 pha chính:

### 3.1. Pha offline (nặng, chạy trước)
Mục tiêu:
- Xây dựng ngân hàng embedding cho toàn bộ dataset

Luồng:
1. Quét dữ liệu ảnh
2. Detect face (đối với custom)
3. Embedding từng ảnh thành vector 512 chiều
4. Lưu ra embeddings.npy, labels.npy, images.npy

### 3.2. Pha online (nhanh, chạy khi người dùng query)
Mục tiêu:
- Nhận ảnh query và trả Top K nhanh

Luồng:
1. Detect và embed ảnh query
2. So cosine với ngân hàng embeddings
3. Sắp xếp và trả Top K
4. Hiển thị score + ảnh kết quả

Ý nghĩa kiến trúc:
- Tách offline và online giúp online phản hồi nhanh
- Chỉ cần làm nặng một lần ở pha offline

---

## 4. Dữ liệu và hai luồng chính của project

Project hỗ trợ hai luồng độc lập:

## 4.1. Luồng LFW
Mục tiêu:
- Luồng chuẩn học thuật, dễ chạy từ đầu

Đặc điểm:
- Dữ liệu lấy từ sklearn fetch_lfw_people
- Ảnh LFW vốn đã là face crop
- Không cần detect lại bằng MTCNN ở bước embedding LFW

Script chính:
- src/01_preprocess.py
- src/02_embed.py
- src/03_retrieval.py
- src/04_cluster.py
- src/05_visualize.py

## 4.2. Luồng custom dataset người Việt
Mục tiêu:
- Dùng dữ liệu thực tế của đồ án

Đặc điểm:
- Dữ liệu ảnh thô trong custom_dataset
- Cần MTCNN detect face trước khi embed
- Hỗ trợ cả cấu trúc thư mục 1 cấp và 2 cấp

Script chính:
- src/02b_embed_custom.py
- Sau đó dùng lại src/03, src/04, src/05 giống luồng LFW

## 4.3. Điểm khác biệt cốt lõi giữa hai luồng

1. Giai đoạn detect mặt
- LFW: bypass detect trong bước embed
- Custom: detect bắt buộc bằng MTCNN

2. Tính thực tế
- LFW: benchmark và demo nhanh
- Custom: gần bài toán thực tế và dữ liệu nội địa

---

## 5. Tiền xử lý và chuẩn hóa dữ liệu

## 5.1. Với LFW
Trong src/02_embed.py:
- Resize về 160 x 160
- Normalize về miền mà FaceNet kỳ vọng
- Stack batch để tăng tốc suy luận

## 5.2. Với custom dataset
Trong src/02b_embed_custom.py:
- Đọc từng ảnh RGB
- Dùng MTCNN detect mặt chính
- Nếu không detect được thì bỏ qua
- Với ảnh detect được:
  - crop và chuẩn hóa
  - đưa vào FaceNet

## 5.3. Vì sao bước này quan trọng
Nếu không chuẩn hóa đúng:
- Vector embedding nhiễu
- Similarity kém ổn định
- Retrieval sai nhiều

---

## 6. MTCNN là gì và tại sao cần

MTCNN là mạng phát hiện mặt theo kiến trúc cascade nhiều tầng.

Ba stage:
1. P-Net
- Quét nhanh toàn ảnh để đề xuất vùng nghi có mặt

2. R-Net
- Lọc bớt false positive

3. O-Net
- Tinh chỉnh bounding box
- Ước lượng landmark mặt

Lợi ích khi dùng MTCNN:
- Tách đúng vùng mặt khỏi nền
- Cải thiện tính ổn định embedding
- Làm retrieval chính xác hơn nhiều so với embed ảnh full scene

Trong project:
- query_external.py và webcam_query.py dùng MTCNN online
- src/02b_embed_custom.py dùng MTCNN offline

---

## 7. FaceNet InceptionResnetV1 và bản chất embedding

## 7.1. FaceNet làm gì
FaceNet không trả nhãn lớp trực tiếp trong project này.
FaceNet trả một vector đặc trưng 512 chiều cho mỗi khuôn mặt.

Trực giác:
- Cùng người: vector gần nhau
- Khác người: vector xa nhau

## 7.2. Vì sao dùng pretrained VGGFace2
- Không cần train từ đầu
- Chất lượng tốt trên dữ liệu mặt đa dạng
- Phù hợp phạm vi đồ án và tài nguyên máy

## 7.3. Metric learning và Triplet Loss (bản chất học sâu phía sau)
Trong quá trình huấn luyện gốc của FaceNet, mô hình học bằng Triplet Loss:
- Anchor: ảnh người X
- Positive: ảnh khác của cùng người X
- Negative: ảnh người Y

Mục tiêu:
- kéo Anchor gần Positive
- đẩy Anchor xa Negative

Nhờ vậy khi dùng inference:
- embedding space có cấu trúc danh tính rõ
- có thể truy hồi bằng khoảng cách hoặc similarity

---

## 8. Truy hồi Top K: bản chất giống kNN trong embedding space

## 8.1. Hệ truy hồi hiện tại hoạt động thế nào
Sau khi có query embedding q:
1. Tính similarity với mọi e_i trong database
2. Sort giảm dần
3. Lấy K phần tử đầu

Đây là truy hồi exact nearest neighbors theo cosine.

## 8.2. Vì sao có thể coi là kNN
- Bản chất kNN là tìm K hàng xóm gần nhất theo metric
- Ở đây metric là cosine similarity
- Hệ hiện tại không dùng lớp KNeighborsClassifier của sklearn, nhưng logic giống kNN retrieval

## 8.3. Công thức cosine
sim(q, e_i) = (q . e_i) / (||q|| ||e_i||)

Diễn giải:
- sim càng gần 1: càng giống
- sim thấp: càng khác

---

## 9. Đánh giá hệ thống: threshold, ROC, AUC, EER

Trong src/03_retrieval.py có phần đánh giá verification.

## 9.1. Verification vs Retrieval
- Retrieval: trả Top K theo ranking
- Verification: quyết định cặp ảnh có cùng người hay không

## 9.2. Threshold
Đặt ngưỡng t:
- sim >= t: dự đoán cùng người
- sim < t: dự đoán khác người

## 9.3. ROC và AUC
- ROC: đường trade-off giữa True Positive Rate và False Positive Rate khi quét mọi ngưỡng
- AUC: diện tích dưới ROC, càng gần 1 càng tốt

## 9.4. EER
- EER là điểm mà False Accept Rate xấp xỉ False Reject Rate
- EER thấp là tốt

Ý nghĩa cho bảo vệ:
- Có thể chứng minh chất lượng mô hình bằng chỉ số chuẩn quốc tế
- Không chỉ demo bằng hình ảnh cảm tính

---

## 10. KMeans Clustering: vì sao dùng và giải quyết gì

## 10.1. Mục tiêu
Không cần nhãn, vẫn muốn xem embedding có tự tách cụm không.

## 10.2. Thuật toán KMeans làm gì
1. Chọn K tâm cụm ban đầu
2. Gán mỗi điểm vào tâm gần nhất
3. Cập nhật tâm cụm theo trung bình
4. Lặp tới hội tụ

## 10.3. Chọn K bằng Elbow và Silhouette
- Elbow: xem độ giảm inertia theo K
- Silhouette: đo độ tách cụm, càng cao càng tốt

Trong project:
- src/04_cluster.py thử K từ 3 đến 15
- Vẽ đồng thời Elbow và Silhouette
- Lưu cluster_labels.npy để dùng tiếp cho visualization

## 10.4. Giá trị học thuật
Nếu embedding tốt:
- cùng danh tính thường vào cụm gần nhau
- phân cụm có ý nghĩa cấu trúc thay vì ngẫu nhiên

---

## 11. Giảm chiều và trực quan: PCA, t-SNE, heatmap

## 11.1. Vì sao cần giảm chiều
Embedding là 512 chiều, con người không quan sát trực tiếp được.

## 11.2. PCA
- Giảm chiều tuyến tính
- Nhanh, ổn định
- Giữ xu hướng phương sai toàn cục

## 11.3. t-SNE
- Giảm chiều phi tuyến
- Tốt để quan sát cấu trúc lân cận
- Chậm hơn PCA, đặc biệt khi dữ liệu lớn

## 11.4. Heatmap cosine
- Xem ma trận tương đồng trong một tập con người
- Quan sát khối tương đồng và ranh giới nhóm

Script:
- src/05_visualize.py

---

## 12. Vai trò của từng file trong dự án

## 12.1. Nhóm lõi src
- src/01_preprocess.py: tải và thống kê LFW
- src/02_embed.py: embed LFW
- src/02b_embed_custom.py: embed custom dataset
- src/03_retrieval.py: retrieval và evaluation
- src/04_cluster.py: clustering
- src/05_visualize.py: trực quan embedding
- src/utils.py: helper lưu tải dữ liệu

## 12.2. Nhóm vận hành ngoài src
- run_pipeline.py: chạy tuần tự pipeline LFW
- query_external.py: query ảnh bên ngoài bằng CLI
- webcam_query.py: query realtime từ webcam
- app.py: giao diện Streamlit

## 12.3. Nhóm tài liệu
- README.md
- HUONG_DAN_CHAY.md
- TECHNICAL_GUIDE.md
- KE_HOACH_DU_AN.md
- DANG_KY_DE_TAI.md

---

## 13. Cách chạy theo tư duy học sâu bài toán

## 13.1. Lộ trình học hiểu cho người mới
1. Chạy luồng LFW bằng run_pipeline.py để thấy end-to-end nhanh
2. Đọc kết quả ở results để hiểu từng output
3. Chạy luồng custom bằng src/02b_embed_custom.py
4. So sánh chất lượng retrieval giữa hai luồng
5. Phân tích ROC, AUC, threshold
6. Quan sát clustering và visualization

## 13.2. Lộ trình vận hành thực tế
1. Chuẩn bị dữ liệu trong custom_dataset
2. Chạy embed một lần
3. Dùng app hoặc webcam cho demo online
4. Khi thêm dữ liệu mới, chạy lại bước embed

---

## 14. Điểm mạnh hiện tại của thiết kế

1. Tách offline và online rõ ràng
2. Dùng pretrained model mạnh, không cần train nặng
3. Có nhiều chế độ sử dụng: CLI, Web UI, Webcam
4. Có đánh giá học thuật bằng ROC AUC EER
5. Có phân cụm và trực quan giúp giải thích mô hình

---

## 15. Hạn chế và hướng nâng cấp chuyên sâu

## 15.1. Hạn chế hiện tại
1. Truy hồi exact brute-force
- O(N) cho mỗi query
- N lớn sẽ chậm dần

2. Chưa có chỉ mục ANN
- Chưa dùng Faiss hoặc Annoy cho nearest neighbor tốc độ cao

3. Chưa fine-tune theo domain Việt Nam
- Dùng pretrained chung
- Có thể cải thiện thêm nếu fine-tune đúng dữ liệu

4. Chưa có calibrate threshold theo từng miền dữ liệu
- Ngưỡng hiện tại chủ yếu theo thực nghiệm chung

## 15.2. Nâng cấp nên làm
1. Dùng Faiss để tăng tốc retrieval lớn
2. Dùng ArcFace loss để fine-tune domain cụ thể
3. Chuẩn hóa chất lượng ảnh đầu vào bằng module quality assessment
4. Tách service backend để triển khai production
5. Thêm logging, monitoring, versioning embeddings

---

## 16. Câu trả lời mẫu khi bảo vệ

## 16.1. AI của bài nằm ở đâu
AI nằm ở representation learning do FaceNet học được.
Project khai thác embedding space bằng cosine retrieval, clustering, và dimensionality reduction.

## 16.2. Vì sao chọn cosine thay vì Euclidean
Embedding của FaceNet mang tính hướng đặc trưng rõ.
Cosine đo độ tương đồng theo hướng vector và thường ổn định hơn cho retrieval danh tính.

## 16.3. Vì sao có cả KMeans
KMeans không dùng để dự đoán người cụ thể.
KMeans dùng để kiểm tra cấu trúc không giám sát của embedding space, tăng giá trị học thuật.

## 16.4. Vì sao có cả LFW và custom
- LFW: benchmark chuẩn, dễ tái lập
- Custom: dữ liệu thực tế bài toán người Việt

---

## 17. Sơ đồ pipeline tóm tắt dễ nhớ

Luồng custom:
1. Thu thập ảnh người Việt từ Kaggle
2. Tổ chức thư mục custom_dataset
3. MTCNN detect và crop mặt
4. FaceNet embedding 512 chiều
5. Lưu embeddings ngân hàng
6. Query ảnh mới bằng cosine Top K
7. Đánh giá ROC AUC
8. Clustering KMeans
9. Visualize PCA t-SNE

Luồng LFW:
1. Tải LFW bằng sklearn
2. Preprocess và thống kê
3. Embed trực tiếp từ face crop
4. Các bước retrieval, evaluate, cluster, visualize giống custom

---

## 18. Kết luận

Bản chất project là một hệ thống truy hồi khuôn mặt dựa trên embedding, trong đó:
- MTCNN giải bài toán phát hiện mặt
- FaceNet giải bài toán biểu diễn danh tính
- Cosine similarity giải bài toán tìm hàng xóm gần nhất
- KMeans và PCA t-SNE giải bài toán phân tích cấu trúc dữ liệu
- ROC AUC EER giải bài toán đánh giá khoa học

Đây là kiến trúc rất điển hình của nhiều hệ thống nhận diện khuôn mặt trong thực tế, chỉ khác ở quy mô và hạ tầng triển khai.

---

## 19. Checklist tự kiểm tra bạn đã hiểu sâu chưa

1. Bạn có phân biệt được retrieval và classification không
2. Bạn có giải thích được vì sao phải detect mặt trước khi embed không
3. Bạn có mô tả được sự khác nhau giữa luồng LFW và custom không
4. Bạn có nói được vì sao cosine là metric chính ở đây không
5. Bạn có giải thích được KMeans dùng để làm gì trong đồ án không
6. Bạn có đọc và diễn giải được AUC, EER khi nhìn kết quả không
7. Bạn có chỉ ra được bước nào chạy offline, bước nào online không

Nếu trả lời được 7 câu trên, bạn đã nắm khá sâu bài toán này.
