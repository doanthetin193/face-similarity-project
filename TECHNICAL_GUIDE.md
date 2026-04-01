# 📚 Hướng Dẫn Kỹ Thuật — Face Similarity Retrieval System

> Tài liệu này giải thích chi tiết các kiến thức kỹ thuật, thuật toán, và cách chúng được áp dụng vào project.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Face Detection — MTCNN](#2-face-detection--mtcnn)
3. [Face Recognition — FaceNet](#3-face-recognition--facenet)
4. [Metric Learning & Triplet Loss](#4-metric-learning--triplet-loss)
5. [Cosine Similarity](#5-cosine-similarity)
6. [Clustering — KMeans](#6-clustering--kmeans)
7. [Dimensionality Reduction](#7-dimensionality-reduction)
8. [Transfer Learning & Pretrained Model](#8-transfer-learning--pretrained-model)
9. [Pipeline trong Project](#9-pipeline-trong-project)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Tổng quan kiến trúc

### 1.1. Bài toán cần giải quyết

**Input:** Một ảnh khuôn mặt bất kỳ

**Output:** Top-K khuôn mặt giống nhất trong database

**Thách thức:**
- Khuôn mặt thay đổi theo góc chụp, ánh sáng, tuổi tác
- Database lớn (~31K ảnh) → cần tìm kiếm nhanh
- Không có label để train từ đầu → dùng pretrained model

### 1.2. Giải pháp tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Ảnh khuôn mặt                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  MTCNN Face Detection         │
          │  • Tìm vị trí khuôn mặt       │
          │  • Crop & align → 160×160     │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │  FaceNet Embedding            │
          │  • InceptionResnetV1          │
          │  • Pretrained VGGFace2        │
          │  • Output: vector 512D        │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │  Embedding Space (512D)       │
          │  • Mỗi ảnh = 1 điểm 512D     │
          │  • Gần nhau = giống nhau      │
          └──────────┬───────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ Cosine Similarity │    │ KMeans Clustering │
│ • So sánh vector  │    │ • Phân nhóm tự động│
│ • Top-K retrieval │    │ • Tìm cấu trúc    │
└──────────────────┘    └──────────────────┘
```

**File liên quan:**
- [src/02_embed.py](src/02_embed.py) — Embedding extraction
- [src/03_retrieval.py](src/03_retrieval.py) — Cosine similarity & retrieval
- [src/04_cluster.py](src/04_cluster.py) — KMeans clustering

---

## 2. Face Detection — MTCNN

### 2.1. MTCNN là gì?

**MTCNN** = Multi-task Cascaded Convolutional Networks

Là thuật toán detect khuôn mặt + landmark (mắt, mũi, miệng) rất chính xác.

### 2.2. Kiến trúc 3 stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: P-Net (Proposal Network)                           │
│  • CNN nhỏ, nhanh                                            │
│  • Quét toàn bộ ảnh ở nhiều scale                           │
│  • Đưa ra ~1000 vùng có thể có mặt (proposals)              │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: R-Net (Refine Network)                             │
│  • CNN phức tạp hơn                                          │
│  • Lọc false positive từ P-Net                               │
│  • Còn ~100 proposals                                        │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: O-Net (Output Network)                             │
│  • CNN phức tạp nhất                                         │
│  • Output cuối: bounding box + 5 facial landmarks           │
│  • Landmarks: 2 mắt, mũi, 2 góc miệng                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3. Multi-task Learning

Mỗi stage không chỉ detect có/không có mặt, mà làm **3 tasks cùng lúc:**

1. **Face Classification** — có phải mặt không? (binary)
2. **Bounding Box Regression** — tọa độ (x, y, w, h) chính xác
3. **Facial Landmark Localization** — vị trí 5 điểm đặc trưng

**Loss function:**
```
L = α·L_cls + β·L_box + γ·L_landmark

L_cls      = cross-entropy (face vs non-face)
L_box      = L2 loss (predicted box vs ground truth)
L_landmark = L2 loss (predicted points vs ground truth)
```

### 2.4. Trong project

**File:** [query_external.py](query_external.py), [src/02b_embed_custom.py](src/02b_embed_custom.py)

```python
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    image_size=160,      # Output crop size
    margin=20,           # Lấy thêm vùng xung quanh
    keep_all=False,      # Chỉ lấy mặt confidence cao nhất
    device='cpu',
    post_process=True,   # Normalize về [-1, 1]
)

# Detect & crop
img = Image.open('anh.jpg').convert('RGB')
face_tensor = mtcnn(img)  # → tensor (3, 160, 160), range [-1, 1]

if face_tensor is None:
    print("Không detect được mặt!")
```

**Output:**
- Tensor 3×160×160 (RGB)
- Đã aligned (xoay cho mắt nằm ngang)
- Đã normalized về [-1, 1]

---

## 3. Face Recognition — FaceNet

### 3.1. FaceNet là gì?

**Paper:** "FaceNet: A Unified Embedding for Face Recognition and Clustering" (Google, 2015)

**Ý tưởng cốt lõi:**
> Thay vì classify trực tiếp "đây là người A hay B hay C...", hãy biến khuôn mặt thành một vector số sao cho:
> - **Cùng người** → vector gần nhau
> - **Khác người** → vector xa nhau

### 3.2. Embedding Space

**Embedding** = biểu diễn ảnh dưới dạng vector số trong không gian nhiều chiều.

```
Ảnh khuôn mặt        Embedding 512D
     A1    ────►    [0.12, -0.45, 0.78, ...]  ─┐
     A2    ────►    [0.14, -0.43, 0.76, ...]  ─┼─ Gần nhau (cùng người A)
     A3    ────►    [0.13, -0.44, 0.79, ...]  ─┘

     B1    ────►    [-0.56, 0.32, -0.11, ...] ─┐
     B2    ────►    [-0.54, 0.34, -0.09, ...] ─┼─ Gần nhau (cùng người B)
     B3    ────►    [-0.55, 0.33, -0.10, ...] ─┘

     C1    ────►    [0.89, 0.12, -0.67, ...]  ←─ Xa A và B (người C)
```

**Khoảng cách trong không gian 512D = độ giống nhau:**
- `d(A1, A2) = 0.05` → rất giống
- `d(A1, B1) = 1.82` → khác nhau
- `d(A1, C1) = 2.31` → rất khác

### 3.3. Kiến trúc InceptionResnetV1

Model FaceNet trong project dùng **InceptionResnetV1** — kết hợp 2 kiến trúc nổi tiếng:

**Inception Module:**
```
                    Input
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    1×1 conv      3×3 conv      5×5 conv     MaxPool
        ↓             ↓             ↓           ↓
        └─────────────┴─────────────┴───────────┘
                      ↓
                  Concatenate
```
→ Học features ở nhiều scale cùng lúc (1×1 = chi tiết nhỏ, 5×5 = vùng lớn)

**Residual Connection (ResNet):**
```
        Input (x)
           ↓
           ├──────────────┐
           ↓              │  (skip connection)
       Conv layers        │
           ↓              │
         F(x)             │
           ↓              │
    Add: F(x) + x ←───────┘
           ↓
        Output
```
→ Giúp train deep network (>50 layers) không bị vanishing gradient

**Pipeline đầy đủ:**
```
Input: 160×160×3
   ↓
Stem: Conv → BatchNorm → ReLU
   ↓
5× Inception-Resnet-A blocks    ← 35×35 feature maps
   ↓
Reduction-A
   ↓
10× Inception-Resnet-B blocks   ← 17×17 feature maps
   ↓
Reduction-B
   ↓
5× Inception-Resnet-C blocks    ← 8×8 feature maps
   ↓
Average Pooling (8×8 → 1×1)
   ↓
Dropout (0.6)
   ↓
Fully Connected → 512D
   ↓
L2 Normalization  (vector có độ dài = 1)
   ↓
Output: embedding 512D
```

### 3.4. Trong project

**File:** [src/02_embed.py](src/02_embed.py), [src/02b_embed_custom.py](src/02b_embed_custom.py)

```python
from facenet_pytorch import InceptionResnetV1

# Load pretrained model
resnet = InceptionResnetV1(
    pretrained='vggface2',  # Đã train trên VGGFace2 dataset
    classify=False,         # Không classify, chỉ lấy embedding
).eval()

# Embed một batch ảnh
face_tensor = torch.randn(32, 3, 160, 160)  # Batch 32 ảnh
with torch.no_grad():
    embeddings = resnet(face_tensor)  # → (32, 512)

print(embeddings.shape)       # (32, 512)
print(embeddings[0].norm())   # ≈ 1.0 (đã L2-normalized)
```

**Output:**
- Shape: `(N, 512)` — mỗi ảnh → 1 vector 512 số thực
- Normalized: `||embedding|| = 1` (độ dài vector = 1)
- Range: `[-1, 1]` (mỗi chiều)

**Lưu embeddings:**
```python
np.save('embeddings/embeddings.npy', embeddings)  # (31480, 512)
np.save('embeddings/labels.npy', labels)          # (31480,) — tên người
```

---

## 4. Metric Learning & Triplet Loss

### 4.1. Metric Learning là gì?

**Định nghĩa:** Học một hàm khoảng cách (metric) sao cho items giống nhau có distance nhỏ, items khác nhau có distance lớn.

**So sánh với Classification:**

| | Classification | Metric Learning |
|---|---|---|
| Output | Xác suất các class | Embedding vector |
| Loss | Cross-entropy | Triplet Loss / Contrastive Loss |
| Ứng dụng | Phân loại cố định (N classes) | So sánh similarity (∞ classes) |
| Ví dụ | "Đây là người A" | "Ảnh này giống ảnh kia 85%" |

**Ưu điểm Metric Learning:**
- Không cần retrain khi thêm người mới vào database
- Có thể query với người chưa nhìn thấy bao giờ

### 4.2. Triplet Loss

**Ý tưởng:** Mỗi lần train, đưa vào 3 ảnh:

```
┌─────────────────────────────────────────────────────────────┐
│  Anchor (A)     — Ảnh gốc (người X)                          │
│  Positive (P)   — Ảnh khác cùng người X                      │
│  Negative (N)   — Ảnh người Y (khác X)                       │
└─────────────────────────────────────────────────────────────┘
```

**Mục tiêu:**
- Kéo Anchor gần Positive: `d(A, P)` nhỏ
- Đẩy Anchor xa Negative: `d(A, N)` lớn
- Đảm bảo margin: `d(A, N) - d(A, P) ≥ margin`

**Loss function:**
```
L_triplet = max(0, ||f(A) - f(P)||² - ||f(A) - f(N)||² + α)

Trong đó:
  f(x)   = embedding của ảnh x
  ||·||² = squared Euclidean distance
  α      = margin (thường 0.2)
```

**Ví dụ cụ thể:**

```python
# Giả sử:
emb_anchor   = [0.1, 0.2, 0.3]  # embedding người A
emb_positive = [0.12, 0.21, 0.29]  # cũng người A
emb_negative = [0.8, 0.1, -0.5]    # người B

d_ap = np.linalg.norm(emb_anchor - emb_positive)**2  # = 0.0006
d_an = np.linalg.norm(emb_anchor - emb_negative)**2  # = 1.54

loss = max(0, d_ap - d_an + 0.2)
     = max(0, 0.0006 - 1.54 + 0.2)
     = max(0, -1.3394)
     = 0  ← loss = 0 vì đã phân biệt tốt rồi
```

**Nếu model chưa tốt:**
```python
d_ap = 0.8   # anchor và positive xa nhau
d_an = 0.5   # anchor và negative lại gần nhau (!!)

loss = max(0, 0.8 - 0.5 + 0.2)
     = 0.5   ← loss > 0 → model phải học tiếp
```

### 4.3. Triplet Mining

**Vấn đề:** Nếu chọn random triplets → nhiều triplet quá dễ (loss = 0) → model không học được gì.

**Giải pháp: Hard Triplet Mining**

```
┌─────────────────────────────────────────────────────────────┐
│  Hard Positive   — Ảnh cùng người nhưng khó nhất (khác góc) │
│  Hard Negative   — Ảnh khác người nhưng giống nhất          │
└─────────────────────────────────────────────────────────────┘
```

**Ví dụ:**
- Anchor: Mặt thẳng người A
- Hard Positive: Người A nhìn nghiêng, đeo kính → khó nhận biết là cùng người
- Hard Negative: Người B nhưng nhìn rất giống A (cùng giới tính, tuổi, sống mũi...)

→ Model phải học cách phân biệt subtle differences

### 4.4. Trong project

**Project không train lại** — chỉ dùng model đã train sẵn bằng Triplet Loss.

Nhưng hiểu Triplet Loss giúp:
- Hiểu tại sao embedding space có cấu trúc tốt (cùng người gần nhau)
- Hiểu tại sao threshold ~0.7 là hợp lý (margin trong training = 0.2)

---

## 5. Cosine Similarity

### 5.1. Định nghĩa

**Cosine similarity** đo góc giữa 2 vector:

```
                    A·B
cos(θ) = ───────────────────
         ||A|| × ||B||

Trong đó:
  A·B    = dot product = Σ(a_i × b_i)
  ||A||  = magnitude = sqrt(Σ(a_i²))
```

**Giá trị:**
- `cos(θ) = 1` → góc 0° → vectors cùng hướng hoàn toàn
- `cos(θ) = 0` → góc 90° → vectors vuông góc (không liên quan)
- `cos(θ) = -1` → góc 180° → vectors ngược hướng

### 5.2. Tại sao dùng Cosine thay vì Euclidean?

**Euclidean Distance:**
```
d_euclidean(A, B) = sqrt(Σ(a_i - b_i)²)
```
→ Bị ảnh hưởng bởi độ dài (magnitude) của vector

**Ví dụ:**
```python
A = [1, 2, 3]
B = [2, 4, 6]  # = 2×A, cùng hướng nhưng dài gấp đôi
C = [1, 2, 4]  # khác A ở chiều cuối

d_euclidean(A, B) = 3.74  # xa
d_euclidean(A, C) = 1.00  # gần (!!)

cos_sim(A, B) = 1.00      # hoàn toàn giống (cùng hướng)
cos_sim(A, C) = 0.99      # hơi khác
```

**FaceNet đã L2-normalize embedding** → `||emb|| = 1` → Cosine = scaled Euclidean.

Nhưng Cosine vẫn được ưa chuộng vì:
- Diễn giải trực quan hơn (góc giữa vectors)
- Không phụ thuộc vào scale của features

### 5.3. Trong project

**File:** [src/03_retrieval.py](src/03_retrieval.py), [query_external.py](query_external.py)

```python
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings của toàn bộ database
embeddings = np.load('embeddings/embeddings.npy')  # (31480, 512)

# Embedding của ảnh query
query_emb = np.array([...])  # (512,)

# Tính similarity với toàn bộ database
scores = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
# → (31480,) — score cho từng ảnh

# Top-5 cao nhất
top5_indices = np.argsort(scores)[::-1][:5]
top5_scores = scores[top5_indices]

print("Top-5 giống nhất:")
for idx, score in zip(top5_indices, top5_scores):
    print(f"  {labels[idx]}: {score:.3f}")
```

**Output ví dụ:**
```
Top-5 giống nhất:
  772: 0.714
  ca sĩ Jang Mi: 0.713
  ca sĩ Thùy Chi: 0.705
  diễn viên Kim Thần: 0.692
  ca sĩ Jang Mi: 0.691
```

### 5.4. Threshold để phân biệt cùng/khác người

**Từ thực nghiệm:**
- `sim > 0.8` → Rất giống, có thể cùng người
- `0.7 < sim < 0.8` → Khá giống, tương đồng nhiều đặc trưng
- `0.6 < sim < 0.7` → Tương đồng vừa phải
- `sim < 0.6` → Ít giống

**Trong retrieval evaluation:**
```python
SAME_PERSON_THRESH = 0.7

if similarity >= SAME_PERSON_THRESH:
    print("Có thể cùng người")
else:
    print("Khác người")
```

**Precision/Recall tại threshold=0.7 (trên LFW):**
- Precision: 99.88% — predict cùng người → 99.88% đúng
- Recall: 73.76% — trong các cặp cùng người thật, bắt được 73.76%

---

## 6. Clustering — KMeans

### 6.1. KMeans là gì?

**Mục tiêu:** Phân N điểm thành K cụm (clusters) sao cho:
- Điểm cùng cụm gần nhau
- Điểm khác cụm xa nhau

**Thuật toán:**
```
1. Random chọn K điểm làm centroids ban đầu
2. Repeat:
     a. Assign: Gán mỗi điểm vào cluster có centroid gần nhất
     b. Update: Tính lại centroid = mean của các điểm trong cluster
3. Until: centroids không đổi đáng kể (hội tụ)
```

**Visualization:**
```
Iteration 0:              Iteration 1:            Iteration 5 (hội tụ):
 × ·  ·                   × ·  ·                  ×····
 · ·× ·                   ·○·× ·                  ○   ·
·  ·  ·                   · · ·                   ·  ○
 ·○ ·  ·                  · ○· ·                  ·   ○
  ·  ·×                   · · ·×                      ·×

× = centroids            ○ = updated centroids
· = data points          màu = cluster assignment
```

### 6.2. Elbow Method — Chọn K tối ưu

**Vấn đề:** Không biết nên chọn K = bao nhiêu?

**Giải pháp:** Chạy KMeans với nhiều giá trị K, vẽ đồ thị **Inertia** (WCSS):

```
Inertia = Within-Cluster Sum of Squares
        = Σ Σ ||x - centroid_i||²
          i x∈C_i
```
→ Tổng khoảng cách từ mỗi điểm đến centroid của cluster nó

**Đồ thị Elbow:**
```
Inertia
  |
  |●
  | ●
  |  ●
  |   ●___
  |       ●___●___●___●
  |─────────────────────── K
     3  5  7  9  11 13 15
         ↑
      "Khuỷu tay"
```

**Chọn K tại điểm "khuỷu":**
- K nhỏ → inertia cao (cluster quá lớn, không tách bạch)
- K lớn → inertia giảm ít (overfitting, mỗi người 1 cluster)
- K tại khuỷu → trade-off tốt

### 6.3. Silhouette Score — Đánh giá chất lượng cluster

**Công thức:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

Trong đó:
  a(i) = avg distance từ điểm i đến các điểm khác cùng cluster
  b(i) = avg distance từ điểm i đến cluster gần nhất khác
```

**Giá trị:**
- `s = 1` → điểm rất gần cluster của nó, xa cluster khác → tốt
- `s = 0` → điểm nằm giữa 2 cluster →애매
- `s = -1` → điểm gần cluster khác hơn cluster của nó → sai cluster

**Silhouette Score trung bình** của toàn dataset → chất lượng clustering:
- `> 0.7` — Rất tốt
- `0.5 - 0.7` — Tốt
- `0.25 - 0.5` — Yếu, có overlap
- `< 0.25` — Rất yếu, cấu trúc cluster không rõ

### 6.4. Trong project

**File:** [src/04_cluster.py](src/04_cluster.py)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

embeddings = np.load('embeddings/embeddings.npy')  # (31480, 512)

# Elbow Method
inertias = []
sil_scores = []
for k in range(3, 16):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    sil = silhouette_score(embeddings, labels, sample_size=1000)
    sil_scores.append(sil)

# Chọn k tốt nhất
best_k = np.argmax(sil_scores) + 3  # +3 vì range bắt đầu từ 3

# Clustering cuối cùng
km_final = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = km_final.fit_predict(embeddings)

# Phân tích cụm
for c in range(best_k):
    idx = np.where(cluster_labels == c)[0]
    print(f"Cluster {c}: {len(idx)} ảnh")
    # Top người trong cụm
    people_in_cluster = labels[idx]  # labels = tên người
    unique, counts = np.unique(people_in_cluster, return_counts=True)
    top3 = unique[np.argsort(counts)[::-1][:3]]
    print(f"  Top-3: {top3}")
```

**Kết quả ví dụ (k=15):**
```
Cluster 0 [178 ảnh]: Silva(48), Agassi(36), Abbas(28)
Cluster 1 [237 ảnh]: Powell(236), Zemin(1)
Cluster 12 [561 ảnh]: Bush(530), Bremer(16), Daschle(15)
```

→ Nhiều cluster "thuần" (dominated bởi 1 người) → model phân biệt tốt

---

## 7. Dimensionality Reduction

### 7.1. Tại sao cần giảm chiều?

**Vấn đề:** Embedding 512D không thể visualize trực tiếp.

**Mục tiêu:** Chiếu 512D → 2D để vẽ scatter plot, nhưng giữ được cấu trúc:
- Điểm gần nhau ở 512D → gần nhau ở 2D
- Điểm xa nhau ở 512D → xa nhau ở 2D

### 7.2. PCA (Principal Component Analysis)

**Ý tưởng:** Tìm các trục có **phương sai lớn nhất**.

**Thuật toán:**
```
1. Center data: X_centered = X - mean(X)
2. Tính covariance matrix: Σ = (1/N) X_centered^T · X_centered
3. Eigendecomposition: Σ = V · D · V^T
     V = eigenvectors (các trục chính)
     D = eigenvalues (phương sai theo mỗi trục)
4. Chọn top-k eigenvectors có eigenvalue lớn nhất
5. Project: X_reduced = X_centered · V[:, :k]
```

**Visualization:**
```
Dữ liệu 2D gốc:         PCA tìm trục chính:      Project lên PC1:

    ·                       ·                     ····
   ·                       ·                     ····
  · ·                     · ·                    ····
 ·   ·                   ·   ·
·     ·      →          ·─────────── PC1   →   (mất chiều PC2)
 ·   ·                  │    ·
  · ·                   │   ·
   ·                    PC2 ·
    ·                       ·
```

**Ưu điểm:**
- Nhanh (chỉ linear transformation)
- Giữ được global structure (variance)

**Nhược điểm:**
- Chỉ tìm được linear relationships
- 512D → 2D mất nhiều thông tin (chỉ giữ ~20% phương sai)

### 7.3. t-SNE (t-distributed Stochastic Neighbor Embedding)

**Ý tưởng:** Giữ **local structure** (điểm gần nhau ở high-D → gần nhau ở low-D).

**Thuật toán (simplified):**
```
1. Tính probability distribution ở high-D:
   p_ij = probability điểm i và j là "neighbors"
        = exp(-||x_i - x_j||²) / Σ exp(-||x_i - x_k||²)

2. Random init low-D coordinates: y_i

3. Tính probability distribution ở low-D:
   q_ij = (1 + ||y_i - y_j||²)^-1 / Σ (1 + ||y_i - y_k||²)^-1

4. Minimize KL divergence: KL(P || Q)
   → Optimize y_i bằng gradient descent
```

**Perplexity:**
- Tham số quan trọng (thường 10-50)
- Cân bằng local vs global structure
- Perplexity = số "neighbors hiệu quả" của mỗi điểm

**Ưu điểm:**
- Giữ local structure rất tốt
- Visualization đẹp, clusters tách biệt rõ

**Nhược điểm:**
- Chậm (O(N²) hoặc O(N log N) với Barnes-Hut)
- Non-deterministic (mỗi lần chạy khác nhau)
- Khoảng cách global không có ý nghĩa (chỉ local có ý nghĩa)

### 7.4. Trong project

**File:** [src/05_visualize.py](src/05_visualize.py)

**PCA:**
```python
from sklearn.decomposition import PCA

# Giảm 512D → 2D
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)  # (31480, 2)

# Phương sai giải thích được
explained = pca.explained_variance_ratio_.sum() * 100
print(f"PCA giữ {explained:.1f}% phương sai")  # ~20-25%

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=5)
```

**t-SNE:**
```python
from sklearn.manifold import TSNE

# PCA 512D → 50D trước để tăng tốc t-SNE
pca_50 = PCA(n_components=50).fit_transform(embeddings)

# t-SNE 50D → 2D
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(pca_50)  # Mất ~1-2 phút

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=5)
```

**So sánh kết quả:**
- PCA → clusters overlap nhiều, nhưng chạy nhanh
- t-SNE → clusters tách biệt rõ, đẹp hơn, nhưng chạy chậm

---

## 8. Transfer Learning & Pretrained Model

### 8.1. Transfer Learning là gì?

**Định nghĩa:** Sử dụng kiến thức model đã học từ task A cho task B.

```
┌─────────────────────────────────────────────────────────┐
│  Task A: Nhận diện 9,131 người trong VGGFace2            │
│  → Model học: edges, textures, facial structures        │
└────────────────────────┬────────────────────────────────┘
                         │ (Transfer)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Task B: Tìm kiếm tương đồng trong 1,244 người VN       │
│  → Dùng features đã học, không train lại                │
└─────────────────────────────────────────────────────────┘
```

**Tại sao hiệu quả?**
- Low-level features (edges, colors) là universal → apply cho mọi ảnh
- Mid-level features (eyes, nose shape) cũng tương tự giữa các dataset
- Chỉ high-level features (identity-specific) cần adapt

### 8.2. VGGFace2 Dataset

**Thông tin:**
- **3.31 triệu ảnh**
- **9,131 identities**
- Đa dạng: góc chụp, tuổi, ánh sáng, chủng tộc (nhưng vẫn thiên phương Tây)

**So với LFW:**

| | LFW | VGGFace2 |
|---|---|---|
| Ảnh | 13K | 3.3M |
| Người | 5.7K | 9.1K |
| Đặc điểm | Ảnh báo chí, frontal | Đa dạng góc, trong tự nhiên |
| Mục đích | Benchmark | Training large models |

**Tại sao pretrain trên VGGFace2?**
- Đủ lớn để model học features tốt
- Đa dạng pose → generalize tốt
- Public, chuẩn học thuật

### 8.3. Pretrained Model trong Project

**File:** [src/02_embed.py](src/02_embed.py)

```python
from facenet_pytorch import InceptionResnetV1

# Load pretrained weights
resnet = InceptionResnetV1(
    pretrained='vggface2',  # ← Tải weights đã train sẵn
    classify=False,         # Không dùng classification head
).eval()

# Model gồm 2 phần:
# 1. Feature extractor (Inception-Resnet blocks) ← Giữ nguyên
# 2. Classification layer (9131 classes)         ← Bỏ đi, thay = Identity

# Ta chỉ dùng phần 1 → output = 512D embedding
```

**File weights tải về:**
- Tự động download lần đầu (~110MB)
- Cache tại `~/.cache/torch/hub/checkpoints/`
- Không cần download lại lần sau

**Transfer từ VGGFace2 → Vietnamese faces:**
- Model đã học: "mắt to", "mũi cao", "khoảng cách 2 mũi"...
- Áp dụng trực tiếp cho người Việt → vẫn hoạt động tốt
- Score cải thiện từ 0.45 (LFW) → 0.71 (Vietnamese Celebrity) cho ảnh châu Á

### 8.4. Fine-tuning vs Feature Extraction

**2 cách dùng pretrained model:**

**Feature Extraction (project ta đang dùng):**
```python
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# Freeze tất cả weights → không train
for param in resnet.parameters():
    param.requires_grad = False

# Chỉ extract features
embeddings = resnet(images)
```

**Fine-tuning (nếu có dataset label + GPU):**
```python
resnet = InceptionResnetV1(pretrained='vggface2')
# Chỉ freeze một số layer đầu
for param in resnet.parameters():
    param.requires_grad = True

# Add classification head mới cho Vietnamese faces
classifier = nn.Linear(512, 224)  # 224 người

# Train lại với learning rate nhỏ
optimizer = torch.optim.Adam([
    {'params': resnet.parameters(), 'lr': 1e-5},  # Fine-tune
    {'params': classifier.parameters(), 'lr': 1e-3}  # Train mới
])
```

**Project không fine-tune vì:**
- Không cần classify → chỉ cần similarity
- Feature extractor đủ tốt rồi
- Tiết kiệm thời gian + GPU

---

## 9. Pipeline trong Project

### 9.1. Quy trình từng bước

```
┌──────────────────────────────────────────────────────────┐
│  OFFLINE (chạy 1 lần, lưu embeddings)                     │
└──────────────────────────────────────────────────────────┘

Bước 1: Load dataset
  Files: src/01_preprocess.py (LFW)
         src/02b_embed_custom.py (Custom)
  →  ~31K ảnh từ custom_dataset/

Bước 2: Face Detection + Embedding
  File: src/02b_embed_custom.py
  Mỗi ảnh:
    → MTCNN detect & crop → 160×160
    → InceptionResnetV1 → 512D embedding
  → Lưu: embeddings.npy (31480, 512)
         labels.npy (31480,)
         images.npy (31480, 62, 62, 3)

┌──────────────────────────────────────────────────────────┐
│  ONLINE (query nhanh, chỉ mất vài giây)                  │
└──────────────────────────────────────────────────────────┘

Bước 3A: Retrieval (nội bộ dataset)
  File: src/03_retrieval.py
  → Chọn 1 ảnh trong dataset làm query
  → Cosine similarity với 31479 ảnh còn lại
  → Top-K cao nhất

Bước 3B: External Query (ảnh từ ngoài)
  File: query_external.py
  → User upload ảnh
  → MTCNN detect & crop
  → FaceNet → 512D embedding
  → Cosine similarity với 31480 embeddings
  → Top-K cao nhất
  → Hiển thị kết quả

Bước 4: Clustering
  File: src/04_cluster.py
  → KMeans trên 31480 embeddings
  → Elbow Method chọn K
  → Phân tích clusters

Bước 5: Visualization
  File: src/05_visualize.py
  → PCA: 512D → 2D
  → t-SNE: 512D → 2D
  → Scatter plots, heatmaps
```

### 9.2. Data Flow

**Embedding Phase:**
```
custom_dataset/
├── Ca sĩ/
│   └── ca sĩ Bảo Anh/
│       ├── anh1.jpg  ──┐
│       └── anh2.jpg  ──┤
└── VN-celeb/           │
    └── 1/              │
        └── 0.png  ─────┤
                        │
                        ▼
                   [MTCNN detect]
                        │
                        ▼
              face crop 160×160×3
                        │
                        ▼
              [InceptionResnetV1]
                        │
                        ▼
              embedding (512,)
                        │
                        ▼
               embeddings.npy  ←─── Lưu tất cả vào đây
               labels.npy
```

**Query Phase:**
```
User ảnh
    │
    ▼
[MTCNN]  →  face crop
    │
    ▼
[FaceNet]  →  query_emb (512,)
    │
    ▼
Load embeddings.npy (31480, 512)
    │
    ▼
[Cosine Similarity]
    │
    ▼
scores (31480,)  →  sort  →  Top-K
    │
    ▼
Hiển thị: Top-K ảnh + tên + scores
```

### 9.3. Files và vai trò

| File | Vai trò | Input | Output |
|---|---|---|---|
| [src/utils.py](src/utils.py) | Helper functions | - | - |
| [src/01_preprocess.py](src/01_preprocess.py) | Load LFW, thống kê | LFW auto-download | Matplotlib figs |
| [src/02_embed.py](src/02_embed.py) | Embed LFW dataset | LFW images | embeddings.npy |
| [src/02b_embed_custom.py](src/02b_embed_custom.py) | Embed custom dataset | custom_dataset/ | embeddings.npy |
| [src/03_retrieval.py](src/03_retrieval.py) | Top-K retrieval nội bộ | embeddings.npy | Matplotlib figs |
| [query_external.py](query_external.py) | Query ảnh ngoài | user image + embeddings.npy | Top-K results |
| [src/04_cluster.py](src/04_cluster.py) | KMeans clustering | embeddings.npy | cluster_labels.npy + figs |
| [src/05_visualize.py](src/05_visualize.py) | PCA, t-SNE, heatmap | embeddings.npy | Matplotlib figs |
| [run_pipeline.py](run_pipeline.py) | Chạy 01→05 LFW | - | - |

---

## 10. Tài liệu tham khảo

### 10.1. Papers

**Face Recognition:**
1. **FaceNet** — Schroff et al., 2015
   - Title: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
   - Link: https://arxiv.org/abs/1503.03832

2. **MTCNN** — Zhang et al., 2016
   - Title: "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
   - Link: https://arxiv.org/abs/1604.02878

3. **DeepFace** (Facebook) — Taigman et al., 2014
   - Tiền thân của FaceNet

**CNN Architectures:**
4. **Inception (GoogLeNet)** — Szegedy et al., 2014
   - Title: "Going Deeper with Convolutions"
   - Link: https://arxiv.org/abs/1409.4842

5. **ResNet** — He et al., 2015
   - Title: "Deep Residual Learning for Image Recognition"
   - Link: https://arxiv.org/abs/1512.03385

**Dimensionality Reduction:**
6. **t-SNE** — van der Maaten & Hinton, 2008
   - Title: "Visualizing Data using t-SNE"
   - Link: https://jmlr.org/papers/v9/vandermaaten08a.html

### 10.2. Courses

**Deep Learning:**
- **CS231n** (Stanford) — Convolutional Neural Networks for Visual Recognition
  - http://cs231n.stanford.edu/
  - Video lectures trên YouTube

- **Andrew Ng Deep Learning Specialization** (Coursera)
  - Course 1: Neural Networks and Deep Learning
  - Course 4: Convolutional Neural Networks

**Machine Learning Cơ bản:**
- **Fast.ai** — Practical Deep Learning for Coders
  - https://course.fast.ai/

### 10.3. Libraries Documentation

- **PyTorch:** https://pytorch.org/docs/
- **facenet-pytorch:** https://github.com/timesler/facenet-pytorch
- **scikit-learn:** https://scikit-learn.org/stable/documentation.html
- **NumPy:** https://numpy.org/doc/

### 10.4. Datasets

- **LFW:** http://vis-www.cs.umass.edu/lfw/
- **VGGFace2:** https://github.com/ox-vgg/vgg_face2
- **CelebA:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **VN-Celeb (Kaggle):** https://www.kaggle.com/datasets/dnguyenhoang/vn-celeb

### 10.5. Từ khóa để research thêm

**Các chủ đề nâng cao:**
- Siamese Networks
- Contrastive Learning
- Hard Negative Mining
- ArcFace, CosFace (cải tiến của FaceNet)
- Few-shot Learning
- Open-set Recognition
- Domain Adaptation

---

## Phụ lục: Thuật ngữ Anh-Việt

| English | Tiếng Việt |
|---|---|
| Embedding | Vector biểu diễn / Nhúng |
| Face Detection | Phát hiện khuôn mặt |
| Face Recognition | Nhận diện khuôn mặt |
| Pretrained Model | Mô hình được train sẵn |
| Transfer Learning | Học truyền tải |
| Metric Learning | Học khoảng cách |
| Triplet Loss | Hàm mất mát bộ ba |
| Cosine Similarity | Độ tương đồng cosine |
| Clustering | Phân cụm |
| Dimensionality Reduction | Giảm chiều |
| Retrieval | Truy vấn / Tìm kiếm |
| Feature Extraction | Trích xuất đặc trưng |
| Convolutional Neural Network | Mạng nơ-ron tích chập |
| Residual Connection | Kết nối tàn dư |
| Attention Mechanism | Cơ chế chú ý |

---

**Chúc bạn học tốt và hiểu sâu về project!** 🚀

_Nếu có thắc mắc về phần nào, hãy đọc lại chi tiết ở mục tương ứng hoặc tham khảo papers gốc._
