# Project Handoff Context: Clickstream Behavior Intelligence App

Dokumen ini berisi konteks lengkap agar proyek dapat dilanjutkan di akun ChatGPT lain, laptop lain, atau oleh reviewer/dosen/lektor.

---

## 1. Ringkasan Proyek

**Nama proyek:** Clickstream Behavior Intelligence App  
**Bidang:** Machine Learning, Classification, Clustering, Behavioral Analytics  
**Arah unik proyek:** E-commerce clickstream analysis sebagai jembatan menuju cyber-inspired behavioral log analysis.

Proyek ini bertujuan membangun aplikasi machine learning untuk menganalisis perilaku pengguna berdasarkan data clickstream dari website e-commerce. Data clickstream diperlakukan sebagai bentuk behavioral log, sehingga pendekatan metodologinya dapat dikaitkan dengan analisis log pada sistem cyber, seperti session behavior, sequence event, pola klik, dan deteksi perilaku tidak biasa.

---

## 2. Dataset

**Nama dataset:** Clickstream Data for Online Shopping  
**Sumber:** UCI Machine Learning Repository  
**Link dataset:** https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping

Dataset ini berisi data aktivitas klik pengguna pada toko online pakaian. Setiap baris merepresentasikan aktivitas klik dalam sebuah sesi pengguna.

### File utama

File yang perlu digunakan:

```text
e-shop clothing 2008.csv
```

Setelah diunduh, simpan file tersebut ke:

```text
data/raw/clickstream_raw.csv
```

### Catatan penting

Awalnya dicoba menggunakan package `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
clickstream = fetch_ucirepo(id=553)
```

Namun muncul error:

```text
DatasetNotFoundError:
"Clickstream Data for Online Shopping" dataset (id=553) exists in the repository,
but is not available for import.
```

Kesimpulan: dataset ada di UCI, tetapi tidak tersedia untuk import otomatis melalui `ucimlrepo`. Karena itu, pendekatan yang dipilih adalah menggunakan url tidak menggunakan library `ucimlrepo`
---

## 3. Tujuan Machine Learning

Proyek ini punya dua tugas utama:

### 3.1 Classification

Classification digunakan untuk memprediksi label perilaku pengguna berdasarkan fitur session-level.

Target utama:

```text
high_engagement
```

Definisi:

- `1` = sesi dengan engagement tinggi.
- `0` = sesi dengan engagement biasa/rendah.

Target dibuat dari jumlah klik per sesi. Contoh definisi:

```python
threshold = session_df["total_clicks"].quantile(0.75)
session_df["high_engagement"] = (session_df["total_clicks"] > threshold).astype(int)
```

Target tambahan:

```text
premium_interest
```

Definisi:

- `1` = sesi yang cenderung melihat produk premium.
- `0` = sesi yang tidak dominan melihat produk premium.

Contoh definisi:

```python
session_df["premium_interest"] = (
    session_df["premium_click_ratio"] >= 0.5
).astype(int)
```

Catatan metodologis penting:

Jika target dibuat dari fitur tertentu, fitur pembentuk target tidak boleh digunakan sebagai input model, agar tidak terjadi data leakage.

Contoh: jika `premium_interest` dibuat dari `premium_click_ratio`, maka `premium_click_ratio` tidak boleh dipakai sebagai fitur input saat melatih model untuk target `premium_interest`.

---

### 3.2 Clustering

Clustering digunakan untuk mengelompokkan sesi pengguna berdasarkan pola perilaku.

Contoh cluster yang diharapkan:

1. **Low Activity Visitors**: pengguna dengan sedikit klik dan eksplorasi rendah.
2. **Product Explorers**: pengguna dengan banyak klik dan banyak produk unik.
3. **Premium-Oriented Visitors**: pengguna yang sering melihat produk dengan harga tinggi.
4. **Focused Browsers**: pengguna yang fokus pada sedikit kategori atau produk.
5. **Multi-Category Explorers**: pengguna yang berpindah-pindah kategori.

---

## 4. Analogi dengan Cyber Log Analysis

Proyek ini sengaja diarahkan agar tidak sekadar menjadi e-commerce analytics biasa.

| Clickstream | Cyber/System Log |
|---|---|
| session ID | user session, login session, token session |
| order | urutan event |
| page click | endpoint access, URL hit, API request |
| country | geolocation IP |
| product category | jenis resource yang diakses |
| repeated clicks | repeated access, scanning pattern |
| session length | durasi atau panjang aktivitas |
| unusual click pattern | behavioral anomaly |

Kesamaan pendekatan:

1. Sama-sama berbasis event log.
2. Sama-sama membutuhkan agregasi per sesi.
3. Sama-sama bisa dianalisis dengan classification.
4. Sama-sama bisa dianalisis dengan clustering.
5. Bisa dikembangkan menuju anomaly detection.

---

## 5. Fase Pengerjaan Proyek

### Phase 0 — Project Setup

Tujuan:

- Membuat struktur folder.
- Menyiapkan environment.
- Menyiapkan dependency.
- Menyiapkan dataset.

Deliverable:

- Folder proyek.
- `requirements.txt`
- `README.md`
- `WEEKLY_PROGRESS.md`
- `Dockerfile`
- `.gitignore`
- `.dockerignore`

### Phase 1 — EDA

Tujuan: memahami struktur dan karakteristik dataset.

Analisis yang perlu dilakukan:

1. Jumlah baris dan kolom.
2. Nama kolom.
3. Tipe data.
4. Missing value.
5. Duplikasi.
6. Nilai unik setiap kolom.
7. Distribusi harga.
8. Distribusi negara.
9. Distribusi kategori produk.
10. Distribusi klik per sesi.
11. Aktivitas berdasarkan bulan/hari.

Deliverable:

- Notebook atau script EDA.
- Tabel missing value.
- Tabel unique value.
- Insight awal data.

### Phase 2 — Preprocessing

Tujuan: membersihkan dan menyiapkan data sebelum feature engineering/modeling.

Langkah:

1. Rename kolom.
2. Cek missing value.
3. Cek duplikasi.
4. Konversi tipe data.
5. Simpan data bersih.

Contoh rename kolom:

```python
RENAME_COLS = {
    "session ID": "session_id",
    "page 1 (main category)": "main_category",
    "page 2 (clothing model)": "clothing_model",
    "model photography": "model_photography",
    "price 2": "price_above_avg",
}
```

### Phase 3 — Feature Engineering

Tujuan: mengubah data event-level menjadi session-level dataset.

Fitur session-level yang dibuat:

| Fitur | Makna |
|---|---|
| `total_clicks` | jumlah klik dalam satu sesi |
| `max_order` | order tertinggi dalam sesi |
| `unique_categories` | jumlah kategori unik |
| `unique_products` | jumlah produk unik |
| `unique_colours` | jumlah warna unik |
| `unique_locations` | jumlah lokasi foto unik |
| `unique_pages` | jumlah halaman unik |
| `avg_price` | rata-rata harga produk |
| `max_price` | harga maksimum |
| `min_price` | harga minimum |
| `std_price` | variasi harga |
| `premium_click_ratio` | proporsi klik produk harga tinggi |
| `dominant_category` | kategori paling sering muncul |
| `dominant_colour` | warna paling sering muncul |
| `dominant_page` | halaman paling sering muncul |
| `dominant_country` | negara paling sering muncul |
| `category_diversity_ratio` | variasi kategori dibanding total klik |
| `product_diversity_ratio` | variasi produk dibanding total klik |
| `colour_diversity_ratio` | variasi warna dibanding total klik |
| `page_diversity_ratio` | variasi halaman dibanding total klik |

Deliverable:

```text
data/processed/session_level_dataset.csv
```

### Phase 4 — Modeling

#### Classification models yang akan dicoba

1. Dummy Classifier
2. Logistic Regression
3. Naive Bayes
4. K-Nearest Neighbors
5. Support Vector Machine Linear
6. Support Vector Machine RBF
7. Decision Tree
8. Random Forest
9. Extra Trees
10. AdaBoost
11. Gradient Boosting
12. XGBoost
13. LightGBM
14. CatBoost
15. MLP Classifier

Metrik classification:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. PR-AUC jika data imbalanced
7. Confusion Matrix
8. Training Time

#### Clustering models yang akan dicoba

1. K-Means
2. MiniBatch K-Means
3. Agglomerative Clustering
4. DBSCAN
5. Gaussian Mixture Model
6. Spectral Clustering, opsional

Metrik clustering:

1. Silhouette Score
2. Davies-Bouldin Index
3. Calinski-Harabasz Index
4. Cluster size distribution
5. Interpretasi profil cluster

### Phase 5 — Hyperparameter Tuning & Feature Selection

Model yang dituning terlebih dahulu:

1. Random Forest
2. XGBoost atau LightGBM
3. SVM
4. KNN
5. Logistic Regression

Metode tuning:

1. RandomizedSearchCV
2. GridSearchCV
3. Cross-validation

Feature selection:

1. Correlation analysis
2. Mutual information
3. Recursive Feature Elimination
4. Random Forest feature importance
5. XGBoost/LightGBM feature importance
6. SHAP analysis

### Phase 6 — Streamlit App

Tujuan: membuat dashboard/app sederhana untuk presentasi dan review.

Halaman app yang direncanakan:

1. Project Overview
2. Dataset Preview
3. EDA Dashboard
4. Classification Result
5. Clustering Result
6. Feature Importance
7. Cyber Log Analysis Analogy
8. Final Insight

### Phase 7 — Docker & Deployment

Tujuan: agar project mudah dijalankan di laptop lain atau direview oleh dosen/lektor.

Strategi:

1. GitHub repository untuk source code.
2. `requirements.txt` untuk dependency.
3. Docker untuk reproducibility.
4. Google Cloud Run untuk deployment online.

---

## 6. Struktur Folder Proyek

```text
clickstream-behavior-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_initial_eda.py
│   └── 02_feature_engineering.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── evaluation.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── WEEKLY_PROGRESS.md
└── README.md
```

---

## 7. Environment dan Versi Python

Pada laptop pengguna ditemukan:

```text
Python 3.13.0
pip 24.2
git version 2.53.0.windows.1
```

Namun Python 3.13 berpotensi menimbulkan masalah compatibility dengan beberapa library ML. Kemudian dicek dengan:

```bash
py -0p
```

Hasil menunjukkan Python 3.11 tersedia:

```text
-V:3.11[-64] C:\Users\danur\AppData\Local\Python\pythoncore-3.11-64\python.exe
```

Keputusan: gunakan **Python 3.11** untuk project ini.

Langkah yang disarankan:

```powershell
cd "C:\All Documents\Kuliah\Semester 6\Bengkel Koding\Projek\clickstream-behavior-ml"
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version
```

Target:

```text
Python 3.11.x
```

Lalu install dependency:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Requirements

Isi `requirements.txt` awal:

```text
pandas==2.2.3
numpy==2.1.3
matplotlib==3.9.2
scikit-learn==1.5.2
scipy==1.14.1
plotly==5.24.1
streamlit==1.40.1
ucimlrepo==0.0.7
joblib==1.4.2
```

Catatan: karena `ucimlrepo` gagal mengambil dataset ini, dependency `ucimlrepo` boleh tetap dibiarkan, tetapi proses utama saat ini menggunakan manual download dataset.

---

## 9. Cara Menjalankan Proyek Saat Ini

### Step 1 — Aktifkan virtual environment

```powershell
.venv\Scripts\activate
```

### Step 2 — Pastikan dataset ada

Dataset harus berada di:

```text
data/raw/clickstream_raw.csv
```

Jika belum ada:

1. Buka link UCI: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
2. Download file `e-shop clothing 2008.csv`.
3. Rename menjadi `clickstream_raw.csv`.
4. Simpan ke `data/raw/clickstream_raw.csv`.

### Step 3 — Jalankan EDA awal

```powershell
python scripts/01_initial_eda.py
```

Output yang dihasilkan:

```text
reports/tables/missing_values.csv
reports/tables/unique_values.csv
```

### Step 4 — Buat session-level dataset

```powershell
python scripts/02_feature_engineering.py
```

Output yang dihasilkan:

```text
data/processed/session_level_dataset.csv
```

### Step 5 — Jalankan Streamlit

```powershell
streamlit run app/streamlit_app.py
```

Buka browser:

```text
http://localhost:8501
```

---

## 10. Docker Plan

Docker digunakan agar project dapat dijalankan di laptop reviewer/lektor tanpa konflik dependency.

Isi `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
```

Build image:

```bash
docker build -t clickstream-behavior-ml .
```

Run container:

```bash
docker run -p 8080:8080 clickstream-behavior-ml
```

Buka:

```text
http://localhost:8080
```

Catatan: untuk weekly progress saat ini, Docker belum wajib dijalankan. GitHub repository dan local Streamlit sudah cukup sebagai progress awal.

---

## 11. GitHub Plan

Target weekly submission:

1. Project folder sudah rapi.
2. README tersedia.
3. WEEKLY_PROGRESS tersedia.
4. Script awal tersedia.
5. Link dataset dicantumkan.
6. Repo GitHub bisa dishare ke lektor.

Command upload ke GitHub:

```bash
git init
git add .
git commit -m "Initial weekly progress: setup clickstream behavior ML project"
git branch -M main
git remote add origin https://github.com/USERNAME/clickstream-behavior-ml.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub.

---

## 12. Isi Weekly Progress

File `WEEKLY_PROGRESS.md` berisi:

1. Judul proyek.
2. Fokus student/project.
3. Dataset yang dipakai.
4. Pekerjaan yang sudah dilakukan minggu ini.
5. Status proyek saat ini.
6. Rencana minggu depan.
7. Expected final output.
8. Short explanation for weekly submission.

Poin progress minggu ini:

- Topic selection selesai.
- Dataset selection selesai.
- Project scope selesai.
- Folder structure dibuat.
- Reproducibility setup awal dibuat.
- Initial data script dibuat.
- Initial EDA script dibuat.
- Initial feature engineering script dibuat.
- Starter Streamlit app dibuat.
- Rencana Docker dan deployment disiapkan.

---

## 13. Narasi Singkat untuk Submit Weekly Progress

Gunakan narasi ini saat submit:

```text
This week, I started the Clickstream Behavior Intelligence App project. The project focuses on machine learning for user behavior analysis using classification and clustering. I selected the Clickstream Data for Online Shopping dataset from the UCI Machine Learning Repository because it contains behavioral clickstream data and is suitable for classification and clustering experiments.

The current progress includes project topic selection, dataset selection, project scope definition, repository structure preparation, reproducibility setup using Docker, initial EDA script, initial feature engineering script, and a starter Streamlit dashboard. The project is designed to be developed further into a cyber-inspired behavioral log analysis system.

For the next phase, I will focus on completing EDA, preprocessing, session-level feature engineering, baseline classification models, baseline clustering models, and model comparison.
```

---

## 14. Current Status Terakhir

Status terakhir percakapan:

1. Starter project sudah dibuat dalam bentuk ZIP.
2. User ingin membuat project langkah per langkah.
3. Python 3.11 sudah tersedia.
4. Virtual environment perlu diarahkan ke Python 3.11.
5. `ucimlrepo` gagal mengambil dataset.
6. Keputusan terbaru: pakai dataset file langsung/manual download.
7. File dataset harus disimpan sebagai `data/raw/clickstream_raw.csv`.
8. Langkah berikutnya adalah menjalankan:

```powershell
python scripts/01_initial_eda.py
python scripts/02_feature_engineering.py
streamlit run app/streamlit_app.py
```

---

## 15. Prompt untuk Melanjutkan di Akun ChatGPT Lain

Jika ingin melanjutkan project ini di akun ChatGPT lain, kirim prompt berikut:

```text
Saya sedang membuat project machine learning bernama Clickstream Behavior Intelligence App. Project ini memakai dataset Clickstream Data for Online Shopping dari UCI: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping

Fokus project adalah classification dan clustering untuk analisis perilaku pengguna berbasis clickstream. Arah uniknya adalah cyber-inspired behavioral log analysis, karena clickstream dianalogikan sebagai event/session log.

Status project:
- Struktur folder sudah dibuat.
- README.md dan WEEKLY_PROGRESS.md sudah dibuat.
- Streamlit starter app sudah dibuat.
- Dataset harus didownload manual dari UCI, file e-shop clothing 2008.csv, lalu disimpan sebagai data/raw/clickstream_raw.csv.
- ucimlrepo tidak bisa digunakan karena dataset id 553 tidak tersedia untuk import otomatis.
- Python yang disarankan adalah Python 3.11.
- Virtual environment menggunakan .venv.
- Script yang tersedia:
  - scripts/01_initial_eda.py
  - scripts/02_feature_engineering.py
  - app/streamlit_app.py
- Target classification utama adalah high_engagement.
- Target tambahan adalah premium_interest.
- Clustering dilakukan pada session-level dataset.

Bantu saya melanjutkan project ini step by step mulai dari EDA, preprocessing, feature engineering, modeling classification, modeling clustering, hyperparameter tuning, feature selection, Streamlit dashboard, Docker, dan GitHub deployment.
```

---

## 16. Prioritas Langkah Berikutnya

Langkah teknis selanjutnya:

1. Pastikan dataset manual sudah ada:

```text
data/raw/clickstream_raw.csv
```

2. Jalankan:

```powershell
python scripts/01_initial_eda.py
```

3. Jika berhasil, jalankan:

```powershell
python scripts/02_feature_engineering.py
```

4. Jika berhasil, jalankan:

```powershell
streamlit run app/streamlit_app.py
```

5. Setelah app berhasil tampil, push ke GitHub.
6. Share link GitHub sebagai weekly progression.

---

## 17. Catatan untuk Pengembangan Selanjutnya

Untuk pengembangan modeling, jangan langsung training model di Streamlit app. Training model harus dilakukan di notebook atau script terpisah, lalu model disimpan menggunakan `joblib`.

Contoh:

```python
import joblib
joblib.dump(model, "models/final_classification_model.joblib")
```

Di Streamlit:

```python
import joblib
model = joblib.load("models/final_classification_model.joblib")
```

Alasan:

1. App lebih ringan.
2. Tidak timeout saat demo.
3. Lebih aman saat dijalankan di Cloud Run.
4. Reviewer tidak perlu menunggu training ulang.
5. Project lebih profesional.

---

## 18. Batasan Proyek

Batasan yang perlu ditulis di laporan:

1. Dataset tidak memiliki label pembelian eksplisit.
2. Target classification dibuat melalui engineered label.
3. Dataset berasal dari e-commerce tahun 2008.
4. Hasil tidak boleh digeneralisasi langsung ke semua website e-commerce modern.
5. Hubungan dengan cyber masih bersifat metodologis, bukan dataset cyber asli.
6. Model awal digunakan untuk pembelajaran dan perbandingan, bukan sistem produksi.

---

## 19. Arah Future Work

Pengembangan lanjutan yang dapat ditulis:

1. Menggunakan dataset log cyber asli.
2. Menambahkan anomaly detection.
3. Menambahkan sequence modeling.
4. Menggunakan LSTM/Transformer untuk session sequence.
5. Menggunakan graph-based user behavior analysis.
6. Deploy ke Google Cloud Run.
7. Menambahkan monitoring model.
8. Menambahkan explainable AI menggunakan SHAP.

---

## 20. Keputusan Teknis Final Saat Ini

| Komponen | Keputusan |
|---|---|
| Bahasa | Python |
| Versi Python | 3.11 |
| App | Streamlit |
| Dataset | Clickstream Data for Online Shopping |
| Dataset source | UCI Machine Learning Repository |
| Data loading | Manual CSV download |
| Classification target utama | high_engagement |
| Classification target tambahan | premium_interest |
| Clustering level | session-level |
| Version control | GitHub |
| Reproducibility | Docker |
| Cloud demo rencana | Google Cloud Run |
| Weekly submission | GitHub repo + explanation |
