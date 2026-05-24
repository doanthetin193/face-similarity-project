# Face Similarity Retrieval System

> **Mon hoc:** Lap trinh Tri tue Nhan tao
> **Bai toan:** Truy hoi khuon mat tuong dong bang deep learning
> **Trang thai hien tai:** Custom dataset nguoi Viet la luong chinh; LFW chi con la luong phu de test nhanh.

---

## 1. Gioi thieu

Project xay dung he thong tim kiem khuon mat tuong dong. Nguoi dung dua vao mot anh khuon mat tu bat ky nguon nao nhu upload web, anh test ngoai hoac webcam. He thong se trich xuat embedding 512 chieu bang FaceNet/InceptionResnetV1 pretrained tren VGGFace2, sau do so sanh voi database embedding bang Cosine Similarity va tra ve Top-K khuon mat giong nhat.

Pipeline chinh:

```text
Anh dau vao
-> MTCNN detect khuon mat + 5 landmarks
-> Face Alignment bang hai mat
-> Crop/resize/normalize ve 160x160
-> FaceNet/InceptionResnetV1 pretrained VGGFace2
-> Embedding 512D
-> Cosine Similarity
-> Top-K khuon mat tuong dong
```

Chuc nang chinh:

- Embed custom dataset nguoi Viet trong `custom_dataset/`.
- Truy hoi Top-K bang Cosine Similarity.
- Query anh ngoai bang CLI qua `query_external.py`.
- Giao dien Web Streamlit qua `app.py`.
- Webcam realtime qua `webcam_query.py`.
- Danh gia ROC/AUC/EER trong `src/03_retrieval.py`.
- Phan cum KMeans trong `src/04_cluster.py`.
- Truc quan hoa PCA, t-SNE va heatmap trong `src/05_visualize.py`.

---

## 2. Ket qua moi nhat

Sau khi cap nhat Face Alignment va embed lai custom dataset:

| Hang muc | Gia tri |
|---|---:|
| Tong file anh trong `custom_dataset/` | 31,662 |
| Anh embed thanh cong | 31,480 |
| Anh bo qua do khong detect duoc mat | 182 |
| So nhan/identity | 1,244 |
| Embedding shape | `(31480, 512)` |
| Labels shape | `(31480,)` |
| Images shape | `(31480, 62, 62, 3)` |
| AUC | 0.9914 |
| EER | 0.0518 |
| Threshold tai EER | 0.4968 |
| KMeans tot nhat trong range 3-15 | k = 3 |
| Silhouette tai k=3 | 0.0837 |
| PCA 2D explained variance | 19.1% |
| PCA components dat 90% variance | 36 |
| PCA components dat 95% variance | 41 |

Ghi chu ve "nhan/identity": moi thu muc nguoi/ID trong dataset duoc xem la mot nhan. Vi du `VN-celeb/558/` la mot nhan, `Ca si/Hoang Thuy Linh/` la mot nhan. Mot nhan co the co nhieu anh cua cung mot nguoi/ID.

---

## 3. Cai dat

```powershell
cd D:\AP\face_similarity_project

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Thu vien chinh:

- `facenet-pytorch`: MTCNN va InceptionResnetV1.
- `torch`, `torchvision`: backend deep learning.
- `scikit-learn`: LFW loader, cosine similarity, ROC/AUC, KMeans, PCA, t-SNE.
- `numpy`: xu ly embedding va file `.npy`.
- `Pillow`: doc anh, convert RGB, rotate alignment.
- `opencv-python`: webcam realtime.
- `matplotlib`: ve bieu do.
- `streamlit`: Web UI.
- `tqdm`: progress bar khi embed.

---

## 4. Dataset

### 4.1. Custom dataset nguoi Viet

Day la luong chinh cua project. Thu muc du lieu:

```text
custom_dataset/
|-- Category/
|   |-- Person_A/
|   |   |-- image1.jpg
|   |   `-- image2.jpg
|   `-- Person_B/
`-- VN-celeb/
    |-- 1/
    |-- 2/
    `-- ...
```

`src/02b_embed_custom.py` scan cac file anh co duoi:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

Thong ke custom dataset hien tai:

- Tong file anh: 31,662.
- Anh embed thanh cong: 31,480.
- Anh bi bo qua: 182.
- So nhan/identity: 1,244.
- Vietnamese Celebrity Faces: 8,557 anh, 224 nguoi.
- VN-Celeb: 23,105 anh, 1,020 ID.

### 4.2. LFW dataset

LFW la luong phu de test nhanh pipeline hoc thuat:

- Tai qua `sklearn.datasets.fetch_lfw_people`.
- Cau hinh `min_faces_per_person=20`.
- So anh cache hien tai: 3,023.
- So nguoi: 62.
- Anh LFW da crop san, nen `src/02_embed.py` resize truc tiep ve 160x160 va dua vao FaceNet, khong dung MTCNN.

> Luu y: `run_pipeline.py` chay luong LFW va co the ghi de `embeddings/*.npy`. Voi custom dataset, nen dung `run_custom_pipeline.py`.

---

## 5. Chay pipeline custom dataset

### 5.1. Da co embeddings, chi chay tiep phan tich

Lenh an toan sau khi da embed custom dataset:

```powershell
python run_custom_pipeline.py
```

Lenh nay khong chay lai `src/02b_embed_custom.py`, nen khong ghi de embedding custom hien tai. Cac buoc duoc chay:

```powershell
python src/03_retrieval.py
python src/04_cluster.py
python src/05_visualize.py
```

### 5.2. Co y embed lai toan bo custom dataset

Chi dung khi da them/sua du lieu hoac thay doi tien xu ly:

```powershell
python run_custom_pipeline.py --with-embed
```

Lenh nay se chay:

```powershell
python src/02b_embed_custom.py
python src/03_retrieval.py
python src/04_cluster.py
python src/05_visualize.py
```

`src/02b_embed_custom.py` ton thoi gian vi phai detect/align/crop/embed toan bo anh. Khi embed lai, `save_embeddings()` tu xoa `embeddings/cluster_labels.npy` cu neu co de tranh lech so dong voi embedding moi.

---

## 6. Chay LFW pipeline phu

```powershell
python run_pipeline.py
```

Hoac chay tung buoc:

```powershell
python src/01_preprocess.py
python src/02_embed.py
python src/03_retrieval.py
python src/04_cluster.py
python src/05_visualize.py
```

`run_pipeline.py` chi phu hop voi LFW/test nhanh. Neu dang bao ve ket qua custom dataset, khong nen chay lenh nay vi no co the ghi de embedding custom.

---

## 7. Demo san pham

### 7.1. Streamlit Web UI

```powershell
streamlit run app.py
```

Mo trinh duyet:

```text
http://localhost:8501
```

Web UI gom cac tab:

| Tab | Chuc nang |
|---|---|
| Query Upload | Upload anh, detect/align/embed, hien thi Top-K |
| Dataset Info | Thong ke dataset va anh mau |
| Ket qua & Bieu do | Xem ROC, PCA, t-SNE, heatmap, retrieval |
| Clustering | Xem Elbow, Silhouette, anh mau cum |
| Pipeline & Giai thich | Giai thich pipeline va y nghia ket qua |

Nguong hien thi similarity trong UI:

| Khoang score | Dien giai |
|---|---|
| `score >= 0.70` | Rat giong / du doan cung nguoi |
| `0.60 <= score < 0.70` | Kha giong, ung vien tuong dong trong Top-K |
| `0.4968 <= score < 0.60` | Co tuong dong nhe theo nguong EER |
| `score < 0.4968` | Khong chac chan |

`0.70` duoc dong bo voi `SAME_PERSON_THRESH` trong `src/03_retrieval.py`. `0.4968` la threshold tai EER tu ket qua thuc nghiem moi nhat.

### 7.2. Query anh ngoai bang CLI

```powershell
python query_external.py "duong_dan_anh.jpg"
python query_external.py "duong_dan_anh.jpg" --topk 8
```

Ket qua duoc luu sau khi chay lenh:

```text
results/query_external_result.png
```

### 7.3. Webcam realtime

```powershell
python webcam_query.py
python webcam_query.py --topk 5
python webcam_query.py --camera 1
```

Cau hinh chinh:

- `FRAME_SKIP = 4`: chi embed moi 4 frame de tang FPS.
- `CONF_THRESH = 0.85`: chi xu ly mat co confidence tu 0.85 tro len.
- `MATCH_THRESH = 0.70`: nguong Match.
- `SIMILAR_THRESH = 0.60`: nguong Similar.

Phim tat:

| Phim | Chuc nang |
|---|---|
| `Q` / `ESC` | Thoat |
| `S` | Chup snapshot vao `results/webcam_snapshot.png` |
| `SPACE` | Tam dung / tiep tuc |

---

## 8. Cau truc project

```text
face_similarity_project/
|-- src/
|   |-- utils.py               # Path, save/load, face alignment
|   |-- 01_preprocess.py       # Load/thong ke LFW
|   |-- 02_embed.py            # Embed LFW
|   |-- 02b_embed_custom.py    # Embed custom dataset nguoi Viet
|   |-- 03_retrieval.py        # Top-K, similarity distribution, ROC/AUC/EER
|   |-- 04_cluster.py          # KMeans, Elbow, Silhouette
|   `-- 05_visualize.py        # PCA, t-SNE, heatmap
|-- app.py                     # Streamlit Web UI
|-- query_external.py          # Query anh ngoai bang CLI
|-- webcam_query.py            # Webcam realtime
|-- run_custom_pipeline.py     # Pipeline custom an toan, mac dinh bo qua embed
|-- run_pipeline.py            # Pipeline LFW phu
|-- custom_dataset/            # Dataset nguoi Viet
|-- embeddings/                # embeddings.npy, labels.npy, images.npy, cluster_labels.npy
|-- results/                   # Hinh ket qua pipeline
|-- docs/
|-- requirements.txt
`-- README.md
```

---

## 9. File ket qua quan trong

| File | Y nghia |
|---|---|
| `embeddings/embeddings.npy` | Embedding 512D cua database |
| `embeddings/labels.npy` | Nhan/identity tuong ung moi embedding |
| `embeddings/images.npy` | Thumbnail 62x62 de hien thi, khong phai tensor FaceNet |
| `embeddings/cluster_labels.npy` | Nhan cum KMeans moi nhat |
| `results/03_retrieval_query*.png` | Top-K retrieval mau |
| `results/03_similarity_distribution.png` | Phan phoi similarity |
| `results/03_roc_curve.png` | ROC Curve, AUC, EER |
| `results/04_elbow_silhouette.png` | Elbow + Silhouette |
| `results/04_cluster_samples_k3.png` | Anh mau cum KMeans |
| `results/05_pca_*.png` | PCA visualization |
| `results/05_tsne_*.png` | t-SNE visualization |
| `results/05_similarity_heatmap.png` | Heatmap similarity |
| `results/webcam_snapshot.png` | Anh minh chung webcam neu da chup snapshot |
| `results/query_external_result.png` | Anh minh chung CLI sau khi chay `query_external.py` |

---

## 10. Ghi chu ky thuat

- Custom dataset nen chay bang `run_custom_pipeline.py`, khong chay `run_pipeline.py`.
- `run_custom_pipeline.py` mac dinh bo qua embed va chi chay `03_retrieval.py`, `04_cluster.py`, `05_visualize.py`.
- Chi dung `python run_custom_pipeline.py --with-embed` khi that su muon embed lai toan bo custom dataset.
- Neu sua tien xu ly nhu Face Alignment thi phai embed lai de embedding phan anh code moi.
- Neu embed lai, phai chay lai retrieval, clustering va visualization.
- `cluster_labels.npy` phai co cung so dong voi `embeddings.npy`.
- Threshold `0.70` la nguong tham chieu de du doan cung nguoi trong code danh gia va UI.
- Threshold tai EER la `0.4968`, dung de tham khao diem can bang FPR/FNR tren mau ROC.
- KMeans co Silhouette thap vi dang ep hon 1,200 identity vao vai cum lon, khong co nghia la retrieval kem.
- Bai toan chinh cua project la Top-K face similarity retrieval, khong phai face verification tuyet doi.

---

## 11. Tai lieu lien quan

- `BAO_CAO_BTL.md`: bao cao chinh.
- `PHAN_CONG_NHIEM_VU.md`: phan cong va on tap bao ve.
- `TECHNICAL_GUIDE.md`: giai thich ky thuat.
- `docs/TAI_LIEU_TONG_THE_PIPELINE_CHI_TIET.md`: tai lieu pipeline chi tiet.
