# Panduan Pemula: Clickstream Behavior Intelligence App

Dokumen ini adalah panduan lengkap dan jelas untuk pemula agar bisa memahami proyek ini, cara menjalankannya, serta memahami output yang dihasilkan.

---

## 1. Ringkasan Proyek (Tujuan Dunia Nyata)

Proyek ini tidak hanya membandingkan model. Tujuan akhirnya adalah **memberikan insight yang bisa dipakai tim product/UX** untuk:

1. Mengetahui perilaku sesi dengan engagement tinggi.
2. Menemukan segmen pengguna berdasarkan pola klik (clustering).
3. Memberi rekomendasi tindakan berdasarkan perilaku pengguna.
4. Menyediakan daftar sesi prioritas (action list).

---

## 2. Dataset

**Nama dataset:** Clickstream Data for Online Shopping (UCI)

**Link:**
https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping

**Catatan penting:**
- File CSV dipisahkan dengan **semicolon (;)**, bukan koma (,).
- Dataset harus disimpan sebagai:

```
data/raw/clickstream_raw.csv
```

---

## 3. Struktur Folder (Ringkas)

```
clickstream-behavior-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── markdown/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_initial_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_full_eda.py
│   ├── 04_baseline_classification.py
│   ├── 05_baseline_clustering.py
│   └── 06_build_insight_assets.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

---

## 4. Setup Environment

### 4.1 Buat dan aktifkan environment

```
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 4.2 Install dependencies

```
pip install -r requirements.txt
```

---

## 5. Alur Eksekusi Lengkap (Step by Step)

Ikuti urutan ini dari awal sampai akhir.

### Step 1 - Pastikan dataset sudah ada

Pastikan file ini ada:

```
data/raw/clickstream_raw.csv
```

### Step 2 - EDA awal

```
python scripts/01_initial_eda.py
```

**Output:**
- reports/tables/missing_values.csv
- reports/tables/unique_values.csv

### Step 3 - Feature engineering (session-level dataset)

```
python scripts/02_feature_engineering.py
```

**Output:**
- data/processed/session_level_dataset.csv

### Step 4 - Full EDA dengan visualisasi

```
python scripts/03_full_eda.py
```

**Output utama:**
- reports/figures/event_*.png
- reports/figures/session_*.png
- reports/tables/top_*.csv
- reports/tables/session_feature_summary.csv
- reports/tables/session_feature_correlation.csv

### Step 5 - Baseline classification

```
python scripts/04_baseline_classification.py
```

**Output:**
- reports/tables/classification_baseline_results.csv

Catatan: jika target hanya memiliki satu kelas (semua 1 atau semua 0), target tersebut akan dilewati.

### Step 6 - Baseline clustering

```
python scripts/05_baseline_clustering.py
```

**Output:**
- reports/tables/clustering_baseline_metrics.csv
- reports/tables/clustering_cluster_sizes.csv
- reports/figures/clustering_kmeans_silhouette.png

### Step 7 - Build insight assets (untuk aplikasi UX)

```
python scripts/06_build_insight_assets.py
```

**Output:**
- data/processed/session_level_with_clusters.csv
- reports/tables/cluster_profiles_kmeans_k4.csv
- reports/tables/cluster_sizes_kmeans_k4.csv
- reports/tables/session_action_list.csv

### Step 8 - Jalankan dashboard

```
streamlit run app/streamlit_app.py
```

Buka di browser:

```
http://localhost:8501
```

---

## 6. Cara Membaca Output

### 6.1 EDA Figures

- **event_price_distribution.png**: distribusi harga di level event.
- **event_top_categories.png**: kategori produk teratas berdasarkan klik.
- **event_top_countries.png**: negara teratas berdasarkan klik.
- **session_total_clicks_distribution.png**: distribusi jumlah klik per sesi.
- **session_correlation_heatmap.png**: korelasi antar fitur session-level.

### 6.2 Baseline Classification

File `classification_baseline_results.csv` berisi metrik:
- accuracy
- precision
- recall
- f1_score
- roc_auc

Gunakan tabel ini untuk melihat model baseline terbaik.

### 6.3 Baseline Clustering

File `clustering_baseline_metrics.csv` berisi:
- silhouette
- calinski_harabasz
- davies_bouldin

Tabel ini membantu memilih model clustering terbaik.

### 6.4 Cluster Profiling (UX Insight)

File `cluster_profiles_kmeans_k4.csv` berisi rata-rata fitur per cluster. Ini yang dipakai untuk memberi label segmen, misalnya:
- Cluster dengan click tinggi + produk unik tinggi -> "Explorer"
- Cluster dengan click rendah -> "Low Activity"

### 6.5 Action List

File `session_action_list.csv` berisi daftar sesi dengan skor engagement tertinggi. Ini bisa dianggap sebagai daftar prioritas untuk aksi bisnis.

---

## 7. Penjelasan Target dan Leakage

### Target Classification

1. **high_engagement**
   - Dihitung dari total_clicks (di atas persentil 75).

2. **premium_interest**
   - Dihitung dari premium_click_ratio.

### Data Leakage (penting)

Jika target dibuat dari fitur tertentu, fitur tersebut **tidak boleh** dipakai sebagai input model.

Contoh:
- Target `premium_interest` dibuat dari `premium_click_ratio`, maka fitur itu harus dibuang saat training.

---

## 8. Output Real World (UX/Product)

Contoh dampak yang bisa dijelaskan ke dosen:

1. **UX Improvement**
   - Halaman dengan klik tertinggi dapat menjadi prioritas optimasi.

2. **Segmentasi Pengguna**
   - Cluster membantu memahami tipe pengunjung.

3. **Actionable Insight**
   - Daftar sesi prioritas bisa dipakai untuk retargeting atau eksperimen A/B.

---

## 9. Troubleshooting

### Error: session_id not found

Penyebab: file CSV dibaca sebagai satu kolom karena separator salah.

Solusi:
Pastikan baca file dengan `sep=';'` di semua script.

### Error: premium_interest hanya satu kelas

Penyebab: premium_click_ratio menghasilkan semua 1.

Solusi:
- Ubah threshold premium_interest.
- Atau skip target tersebut (sudah otomatis di script).

---

## 10. Checklist Output Akhir

Pastikan file berikut sudah ada:

- data/processed/session_level_dataset.csv
- reports/figures/*.png
- reports/tables/*.csv
- reports/tables/classification_baseline_results.csv
- reports/tables/clustering_baseline_metrics.csv
- reports/tables/cluster_profiles_kmeans_k4.csv
- reports/tables/session_action_list.csv

---

## 11. Next Improvement (Opsional)

1. Beri nama cluster otomatis berdasarkan aturan sederhana.
2. Tambahkan dashboard untuk menampilkan insight naratif.
3. Tambahkan export CSV di Streamlit.
4. Tambahkan hyperparameter tuning.

---

## 12. Kesimpulan

Dengan pipeline ini, proyek sudah bisa dipakai untuk studi perilaku pengguna secara nyata. Bukan hanya membandingkan model, tetapi menghasilkan **insight yang bisa dijadikan keputusan product/UX**.

Jika kamu butuh dokumentasi tambahan atau ingin versi bahasa Inggris, bilang saja.
