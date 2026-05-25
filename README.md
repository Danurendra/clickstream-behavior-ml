# Clickstream Behavior Intelligence App

Clickstream Behavior Intelligence App adalah proyek machine learning yang berfokus pada analisis perilaku pengguna berdasarkan data clickstream dari website e-commerce. Proyek ini menggunakan dataset **Clickstream Data for Online Shopping** dari UCI Machine Learning Repository untuk membangun fitur berbasis sesi pengguna dan menganalisis pola interaksi pengguna selama menjelajahi toko online.

Proyek ini mencakup proses exploratory data analysis, preprocessing data, feature engineering, serta pengembangan model machine learning untuk dua pendekatan utama, yaitu **classification** dan **clustering**. Pada task classification, model digunakan untuk memprediksi perilaku pengguna seperti tingkat engagement tinggi dan ketertarikan terhadap produk premium. Sementara itu, clustering digunakan untuk mengelompokkan sesi pengguna berdasarkan pola klik, jumlah produk yang dilihat, variasi kategori, halaman yang dikunjungi, rata-rata harga produk, dan rasio interaksi terhadap produk premium.

Selain pengembangan model, proyek ini juga dilengkapi dengan dashboard sederhana berbasis **Streamlit** untuk menampilkan hasil analisis secara interaktif. Struktur proyek dirancang agar mudah dikembangkan, terdokumentasi, dan reproducible melalui penggunaan script modular, penyimpanan dataset terstruktur, serta dukungan deployment menggunakan **Docker**.

Secara umum, proyek ini bertujuan untuk memahami pola perilaku pengguna e-commerce melalui data klik, sekaligus menjadi dasar pengembangan analisis perilaku digital yang dapat diperluas ke arah **cyber-inspired behavioral log analysis**.

## Langkah Inisialisasi Projek

Ikuti langkah ini dari awal agar proyek siap dijalankan.

### 1. Buat dan aktifkan virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Siapkan dataset

1. Download file `e-shop clothing 2008.csv` dari UCI.
2. Simpan sebagai:

```text
data/raw/clickstream_raw.csv
```

Catatan: file CSV dipisahkan dengan `;` (semicolon), bukan koma.

### 4. Jalankan preprocessing dan EDA dasar

```powershell
python scripts/01_initial_eda.py
python scripts/02_feature_engineering.py
```

### 5. Jalankan dashboard

```powershell
streamlit run app/streamlit_app.py
```

Dashboard akan terbuka di:

```text
http://localhost:8501
```
