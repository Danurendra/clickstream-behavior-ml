# Dokumentasi Lengkap Proyek dari Nol
# Clickstream Behavior Intelligence App
## Classification dan Clustering untuk Analisis Perilaku Pengguna Berbasis Clickstream

---

## 0. Ringkasan Singkat

Dokumen ini adalah panduan teknis lengkap untuk membangun proyek machine learning **dari nol** tanpa menggunakan starter project.

Proyek yang dibuat:

```text
Clickstream Behavior Intelligence App
```

Fokus utama:

1. **Classification**
2. **Clustering**
3. **Exploratory Data Analysis**
4. **Preprocessing**
5. **Feature Engineering**
6. **Streamlit Dashboard**
7. **GitHub Repository**
8. **Docker Preparation**
9. **Prediction + UX/Product Insight**

Dataset:

```text
Clickstream Data for Online Shopping
```

Sumber dataset:

```text
https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
```

Dataset ini berisi data perilaku klik pengguna pada website e-commerce. Setiap baris data merepresentasikan satu aktivitas klik dalam satu sesi pengguna.

---

# BAGIAN A
# Konsep Proyek

---

## 1. Latar Belakang Proyek

Website modern menghasilkan data perilaku pengguna dalam jumlah besar. Salah satu bentuk data perilaku tersebut adalah **clickstream data**, yaitu rekaman aktivitas klik pengguna selama menjelajah website.

Clickstream dapat dianalisis untuk memahami:

1. Seberapa aktif pengguna dalam satu sesi.
2. Produk atau kategori apa yang paling sering dilihat.
3. Apakah pengguna cenderung mengeksplorasi banyak produk.
4. Apakah pengguna tertarik pada produk premium.
5. Pola perilaku apa saja yang muncul dari sesi pengguna.

Proyek ini menggunakan clickstream dari website e-commerce, tetapi pendekatannya juga relevan dengan dunia cyber karena clickstream mirip dengan **log aktivitas sistem**.

Contoh analogi:

| Clickstream E-Commerce | Cyber/System Log |
|---|---|
| session ID | user session/login session |
| urutan klik | urutan event log |
| halaman produk | endpoint/API/URL yang diakses |
| negara pengguna | geolocation IP |
| repeated clicks | repeated request/scanning behavior |
| panjang sesi | session duration/activity depth |
| pola klik tidak biasa | behavioral anomaly |

Jadi proyek ini tidak hanya membahas e-commerce, tetapi juga menjadi dasar untuk memahami **behavioral log analysis**.

---

## 2. Tujuan Proyek

Tujuan proyek ini adalah membangun pipeline machine learning end-to-end untuk menganalisis clickstream pengguna.

Tujuan teknis:

1. Mengunduh dan menyiapkan dataset clickstream.
2. Melakukan EDA untuk memahami struktur data.
3. Membersihkan dan menstandarkan dataset.
4. Mengubah data event-level menjadi session-level.
5. Membuat engineered target untuk classification.
6. Melakukan clustering untuk segmentasi perilaku pengguna.
7. Membuat dashboard menggunakan Streamlit.
8. Menyiapkan repository GitHub untuk weekly progress.
9. Menyiapkan Docker agar proyek lebih mudah dijalankan di komputer lain.

---

## 3. Machine Learning Task

### 3.1 Classification

Karena dataset ini tidak memiliki label pembelian eksplisit, maka target classification dibuat melalui **engineered label**.

Target utama:

```text
high_engagement
```

Definisi:

- `1` = sesi dengan engagement tinggi.
- `0` = sesi dengan engagement biasa/rendah.

Cara membuat target:

```python
threshold = session_df["total_clicks"].quantile(0.75)
session_df["high_engagement"] = (session_df["total_clicks"] > threshold).astype(int)
```

Artinya, sesi dengan jumlah klik di atas persentil 75 dianggap sebagai sesi dengan engagement tinggi.

Target tambahan:

```text
premium_interest
```

Definisi:

- `1` = sesi yang dominan melihat produk premium.
- `0` = sesi yang tidak dominan melihat produk premium.

Cara membuat target:

```python
session_df["premium_interest"] = (
    session_df["premium_click_ratio"] >= 0.5
).astype(int)
```

### 3.2 Clustering

Clustering digunakan untuk mengelompokkan sesi pengguna berdasarkan perilaku.

Contoh kemungkinan cluster:

1. **Low Activity Visitors**: pengguna dengan sedikit klik dan eksplorasi rendah.
2. **Product Explorers**: pengguna dengan banyak klik dan banyak produk unik.
3. **Premium-Oriented Visitors**: pengguna yang sering melihat produk mahal/premium.
4. **Focused Browsers**: pengguna yang fokus pada sedikit produk/kategori.
5. **Multi-Category Explorers**: pengguna yang berpindah-pindah kategori.

---

## 4. Fase Pengerjaan

Proyek dibagi menjadi beberapa fase:

```text
Phase 0  : Project Setup
Phase 1  : Dataset Preparation
Phase 2  : Exploratory Data Analysis
Phase 3  : Preprocessing
Phase 4  : Feature Engineering
Phase 5  : Classification Modeling
Phase 6  : Clustering Modeling
Phase 7  : Hyperparameter Tuning & Feature Selection
Phase 8  : Streamlit Dashboard
Phase 9  : GitHub Submission
Phase 10 : Docker & Deployment Preparation
```

Untuk weekly progress awal, minimal yang harus selesai:

1. Struktur folder proyek.
2. Dataset masuk ke folder `data/raw`.
3. Script EDA awal.
4. Script feature engineering awal.
5. Dataset session-level terbentuk.
6. Streamlit app awal bisa berjalan.
7. README dan progress report tersedia.
8. Repo GitHub siap dishare.

---

# BAGIAN B
# Setup Environment dari Nol

---

## 5. Prasyarat Tools

Pastikan komputer memiliki:

1. Python 3.11
2. pip
3. Git
4. VS Code
5. Browser
6. Docker Desktop, opsional untuk tahap awal

### 5.1 Cek Python

Buka PowerShell, lalu jalankan:

```powershell
python --version
```

Jika muncul Python 3.13, tetap bisa terbaca, tetapi untuk proyek ini lebih disarankan Python 3.11.

Cek semua versi Python yang terinstall:

```powershell
py -0p
```

Jika muncul seperti ini:

```text
-V:3.11[-64] C:\Users\...\python.exe
```

Berarti Python 3.11 tersedia.

### 5.2 Cek pip

```powershell
pip --version
```

### 5.3 Cek Git

```powershell
git --version
```

---

## 6. Buat Folder Proyek dari Nol

Pilih folder tempat menyimpan proyek, misalnya:

```text
C:\All Documents\Kuliah\Semester 6\Bengkel Koding\Projek
```

Masuk ke folder tersebut:

```powershell
cd "C:\All Documents\Kuliah\Semester 6\Bengkel Koding\Projek"
```

Buat folder proyek:

```powershell
mkdir clickstream-behavior-ml
cd clickstream-behavior-ml
```

---

## 7. Buat Virtual Environment

Gunakan Python 3.11:

```powershell
py -3.11 -m venv .venv
```

Aktifkan virtual environment:

```powershell
.venv\Scripts\activate
```

Jika berhasil, terminal akan menampilkan tanda:

```text
(.venv)
```

Cek versi Python di dalam environment:

```powershell
python --version
```

Target:

```text
Python 3.11.x
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

# BAGIAN C
# Membuat Struktur Folder

---

## 8. Struktur Folder yang Digunakan

Buat folder berikut:

```powershell
mkdir app
mkdir data
mkdir data\raw
mkdir data\interim
mkdir data\processed
mkdir models
mkdir notebooks
mkdir reports
mkdir reports\figures
mkdir reports\tables
mkdir scripts
mkdir src
```

Struktur akhirnya:

```text
clickstream-behavior-ml/
│
├── app/
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
│
└── src/
```

---

# BAGIAN D
# Membuat File Konfigurasi Dasar

---

## 9. Buat requirements.txt

Di root project, buat file:

```text
requirements.txt
```

Isi:

```text
pandas==2.2.3
numpy==2.1.3
matplotlib==3.9.2
scikit-learn==1.5.2
scipy==1.14.1
plotly==5.24.1
streamlit==1.40.1
joblib==1.4.2
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Catatan:

Package `ucimlrepo` tidak digunakan sebagai metode utama karena dataset UCI id 553 tidak tersedia untuk import otomatis melalui package tersebut. Dataset akan dipakai melalui download manual.

---

## 10. Buat .gitignore

Buat file:

```text
.gitignore
```

Isi:

```text
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.DS_Store

data/raw/*.csv
data/interim/*.csv
data/processed/*.csv

models/*.joblib
models/*.pkl

reports/figures/*.png
reports/tables/*.csv
```

Penjelasan:

- `.venv/` tidak perlu masuk GitHub.
- Dataset CSV tidak perlu dipush jika ukurannya besar.
- Model hasil training tidak perlu dipush pada tahap awal.
- File hasil report bisa digenerate ulang.

Jika dosen meminta dataset ikut disertakan, aturan `.gitignore` bisa diubah nanti.

---

## 11. Buat .dockerignore

Buat file:

```text
.dockerignore
```

Isi:

```text
.git
.venv
__pycache__
*.pyc
.ipynb_checkpoints
.env
.DS_Store
reports/figures/*
reports/tables/*
```

---

# BAGIAN E
# Dataset Preparation

---

## 12. Download Dataset Manual

Buka link:

```text
https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
```

Download file:

```text
e-shop clothing 2008.csv
```

Simpan ke folder:

```text
data/raw/
```

Rename file menjadi:

```text
clickstream_raw.csv
```

Path akhir harus seperti ini:

```text
data/raw/clickstream_raw.csv
```

Catatan penting:
- File CSV dipisahkan dengan **semicolon (;)**, bukan koma (,).
- Semua script membaca file dengan `sep=";"`.

Cek file via PowerShell:

```powershell
dir data\raw
```

Harus muncul:

```text
clickstream_raw.csv
```

---

## 13. Kenapa Tidak Pakai ucimlrepo?

Awalnya dicoba:

```python
from ucimlrepo import fetch_ucirepo
clickstream = fetch_ucirepo(id=553)
```

Tetapi muncul error:

```text
DatasetNotFoundError:
"Clickstream Data for Online Shopping" dataset (id=553) exists in the repository,
but is not available for import.
```

Artinya:

1. Dataset memang ada di UCI.
2. Namun dataset tersebut belum tersedia untuk import otomatis melalui package `ucimlrepo`.
3. Solusi paling stabil adalah download file CSV secara manual.

Keputusan teknis:

```text
Gunakan manual CSV download.
```

Ini juga lebih mudah untuk weekly progress karena tidak tergantung pada API/package UCI.

---

# BAGIAN F
# Membuat Script EDA Awal

---

## 14. Buat File scripts/01_initial_eda.py

Buat file:

```text
scripts/01_initial_eda.py
```

Isi kode berikut:

```python
"""
Initial EDA script for Clickstream Behavior Intelligence App.

This script performs basic data inspection:
1. dataset shape
2. column names
3. data types
4. missing values
5. duplicate rows
6. numeric description
7. unique values

Run:
    python scripts/01_initial_eda.py
"""

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/clickstream_raw.csv")
REPORT_TABLE_DIR = Path("reports/tables")


def main() -> None:
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please place the dataset at: "
            "data/raw/clickstream_raw.csv"
        )

    df = pd.read_csv(RAW_PATH, sep=";")

    print("=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing)

    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())

    print("\n=== Numeric Description ===")
    print(df.describe(include="all"))

    print("\n=== Unique Values per Column ===")
    unique_values = df.nunique().sort_values(ascending=False)
    print(unique_values)

    missing.to_csv(REPORT_TABLE_DIR / "missing_values.csv")
    unique_values.to_csv(REPORT_TABLE_DIR / "unique_values.csv")

    print("\nEDA summary tables saved to reports/tables/")


if __name__ == "__main__":
    main()
```

Jalankan:

```powershell
python scripts/01_initial_eda.py
```

Output yang diharapkan:

1. Shape dataset.
2. Daftar kolom.
3. Tipe data.
4. Missing values.
5. Duplicate rows.
6. Statistik deskriptif.
7. Unique values.

File yang terbentuk:

```text
reports/tables/missing_values.csv
reports/tables/unique_values.csv
```

---

# BAGIAN G
# Membuat Preprocessing Utility

---

## 15. Buat File src/preprocessing.py

Buat file:

```text
src/preprocessing.py
```

Isi:

```python
"""
Preprocessing utilities for clickstream data.
"""

import pandas as pd


RENAME_COLS = {
    "session ID": "session_id",
    "page 1 (main category)": "main_category",
    "page 2 (clothing model)": "clothing_model",
    "model photography": "model_photography",
    "price 2": "price_above_avg",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw dataset columns into cleaner column names.
    """
    return df.rename(columns=RENAME_COLS).copy()


def basic_quality_report(df: pd.DataFrame) -> dict:
    """
    Create basic quality report from dataframe.
    """
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": df.nunique().to_dict(),
    }
```

---

# BAGIAN H
# Membuat Feature Engineering Utility

---

## 16. Buat File src/feature_engineering.py

Buat file:

```text
src/feature_engineering.py
```

Isi:

```python
"""
Feature engineering utilities for session-level clickstream analysis.
"""

import pandas as pd


def safe_mode(series: pd.Series):
    """
    Return the most frequent value from a pandas Series.
    If the series has no mode, return None.
    """
    mode_values = series.mode(dropna=True)
    if len(mode_values) == 0:
        return None
    return mode_values.iloc[0]


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build session-level features from standardized clickstream data.
    """

    session_df = df.groupby("session_id").agg(
        total_clicks=("order", "count"),
        max_order=("order", "max"),
        unique_categories=("main_category", "nunique"),
        unique_products=("clothing_model", "nunique"),
        unique_colours=("colour", "nunique"),
        unique_locations=("location", "nunique"),
        unique_pages=("page", "nunique"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        std_price=("price", "std"),
        premium_click_ratio=("price_above_avg", "mean"),
        dominant_category=("main_category", safe_mode),
        dominant_colour=("colour", safe_mode),
        dominant_page=("page", safe_mode),
        dominant_country=("country", safe_mode),
        month=("month", safe_mode),
        day=("day", safe_mode),
    ).reset_index()

    session_df["std_price"] = session_df["std_price"].fillna(0)

    session_df["category_diversity_ratio"] = (
        session_df["unique_categories"] / session_df["total_clicks"]
    )

    session_df["product_diversity_ratio"] = (
        session_df["unique_products"] / session_df["total_clicks"]
    )

    session_df["colour_diversity_ratio"] = (
        session_df["unique_colours"] / session_df["total_clicks"]
    )

    session_df["page_diversity_ratio"] = (
        session_df["unique_pages"] / session_df["total_clicks"]
    )

    return session_df


def add_engineered_targets(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered classification targets:
    1. high_engagement
    2. premium_interest
    """

    session_df = session_df.copy()

    engagement_threshold = session_df["total_clicks"].quantile(0.75)
    session_df["high_engagement"] = (
        session_df["total_clicks"] > engagement_threshold
    ).astype(int)

    session_df["premium_interest"] = (
        session_df["premium_click_ratio"] >= 0.5
    ).astype(int)

    return session_df
```

---

# BAGIAN I
# Membuat Script Feature Engineering

---

## 17. Buat File scripts/02_feature_engineering.py

Buat file:

```text
scripts/02_feature_engineering.py
```

Isi:

```python
"""
Create session-level behavioral dataset from raw clickstream data.

Run:
    python scripts/02_feature_engineering.py
"""

from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import standardize_columns
from src.feature_engineering import build_session_features, add_engineered_targets


RAW_PATH = Path("data/raw/clickstream_raw.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "session_level_dataset.csv"


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "session_id",
        "order",
        "main_category",
        "clothing_model",
        "colour",
        "location",
        "model_photography",
        "price",
        "price_above_avg",
        "page",
        "country",
        "month",
        "day",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please place the dataset at: "
            "data/raw/clickstream_raw.csv"
        )

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH, sep=";")

    print("Standardizing column names...")
    df = standardize_columns(df)

    print("Validating required columns...")
    validate_required_columns(df)

    print("Building session-level features...")
    session_df = build_session_features(df)

    print("Adding engineered targets...")
    session_df = add_engineered_targets(session_df)

    print("Saving processed dataset...")
    session_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Session-level dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {session_df.shape}")

    print("\nTarget distribution: high_engagement")
    print(session_df["high_engagement"].value_counts(normalize=True))

    print("\nTarget distribution: premium_interest")
    print(session_df["premium_interest"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
```

Jalankan:

```powershell
python scripts/02_feature_engineering.py
```

Jika berhasil, akan muncul file:

```text
data/processed/session_level_dataset.csv
```

Cek:

```powershell
dir data\processed
```

---

# BAGIAN J
# Membuat Data Loader Utility

---

## 18. Buat File src/data_loader.py

Buat file:

```text
src/data_loader.py
```

Isi:

```python
"""
Data loading utilities.
"""

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load CSV file from path.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)
```

---

# BAGIAN K
# Membuat Evaluation Utility

---

## 19. Buat File src/evaluation.py

Buat file:

```text
src/evaluation.py
```

Isi:

```python
"""
Evaluation utilities for classification and clustering.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_binary_classifier(y_true, y_pred, y_proba=None) -> dict:
    """
    Evaluate binary classification model.
    """

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)

    return result


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert list of result dictionaries into dataframe.
    """
    return pd.DataFrame(results)
```

---

# BAGIAN L
# Membuat Streamlit App Awal

---

## 20. Buat File app/streamlit_app.py

Buat file:

```text
app/streamlit_app.py
```

Isi:

```python
"""
Streamlit app for Clickstream Behavior Intelligence App.

Run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Clickstream Behavior Intelligence App",
    page_icon="📊",
    layout="wide",
)


PROCESSED_PATH = Path("data/processed/session_level_dataset.csv")


st.title("Clickstream Behavior Intelligence App")
st.caption("Classification and Clustering for User Behavior Analysis")

st.markdown(
    """
    This dashboard analyzes user behavior using clickstream data from an online shopping website.
    The project focuses on two machine learning tasks: **classification** and **clustering**.
    """
)

st.subheader("Project Scope")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Main Task 1", "Classification")

with col2:
    st.metric("Main Task 2", "Clustering")

with col3:
    st.metric("App Framework", "Streamlit")

st.subheader("Dataset Source")

st.markdown(
    """
    **Clickstream Data for Online Shopping**  
    Source: UCI Machine Learning Repository  
    https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
    """
)

st.divider()

if PROCESSED_PATH.exists():
    df = pd.read_csv(PROCESSED_PATH)

    st.subheader("Session-Level Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic Behavioral Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sessions", f"{df['session_id'].nunique():,}")
    c2.metric("Avg Clicks / Session", f"{df['total_clicks'].mean():.2f}")
    c3.metric("Avg Unique Products", f"{df['unique_products'].mean():.2f}")
    c4.metric("Avg Price", f"{df['avg_price'].mean():.2f}")

    st.subheader("Target Distribution")

    if "high_engagement" in df.columns:
        st.write("High Engagement")
        st.bar_chart(df["high_engagement"].value_counts().sort_index())

    if "premium_interest" in df.columns:
        st.write("Premium Interest")
        st.bar_chart(df["premium_interest"].value_counts().sort_index())

    st.subheader("Session-Level Feature Summary")
    numeric_cols = df.select_dtypes(include="number").columns
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

else:
    st.warning(
        """
        Processed dataset not found.

        Please run:

        1. `python scripts/01_initial_eda.py`
        2. `python scripts/02_feature_engineering.py`
        """
    )

st.divider()

st.subheader("Next Development Plan")

st.markdown(
    """
    1. Complete EDA visualizations.
    2. Add classification model comparison.
    3. Add clustering analysis.
    4. Add feature importance.
    5. Add Docker and deployment workflow.
    """
)
```

Jalankan app:

```powershell
streamlit run app/streamlit_app.py
```

Buka browser:

```text
http://localhost:8501
```

---

## 20A. Update Dashboard Terkini (EDA + Insight + Prediksi)

Dashboard saat ini tidak hanya menampilkan ringkasan data, tetapi juga:

1. **EDA Highlights** (grafik event-level dan session-level).
2. **Baseline Classification & Clustering** (tabel metrik).
3. **Cluster Profiling** (rata-rata fitur per cluster).
4. **Action List** (daftar sesi dengan skor engagement tertinggi).
5. **Prediction High Engagement** (hasil model + form input manual).

Untuk mengaktifkan semua panel di dashboard, jalankan urutan ini:

```powershell
python scripts/03_full_eda.py
python scripts/04_baseline_classification.py
python scripts/05_baseline_clustering.py
python scripts/06_build_insight_assets.py
python scripts/07_train_high_engagement_model.py
python scripts/08_predict_high_engagement.py
streamlit run app/streamlit_app.py
```

Di dashboard, buka tab **Prediction: High Engagement → Manual Input** untuk mengisi fitur sendiri dan melihat hasil prediksi.

---

# BAGIAN M
# Membuat README.md

---

## 21. Buat File README.md

Buat file:

```text
README.md
```

Isi:

```markdown
# Clickstream Behavior Intelligence App

## Project Overview

Clickstream Behavior Intelligence App adalah proyek machine learning untuk menganalisis perilaku pengguna berdasarkan data clickstream dari website e-commerce.

Proyek ini berfokus pada:

1. Classification
2. Clustering
3. Behavioral feature engineering
4. Streamlit dashboard
5. Reproducible machine learning workflow

## Dataset

Dataset yang digunakan:

Clickstream Data for Online Shopping  
Source: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping

File dataset yang digunakan:

```text
e-shop clothing 2008.csv
```

Setelah didownload, file disimpan sebagai:

```text
data/raw/clickstream_raw.csv
```

## Project Structure

```text
clickstream-behavior-ml/
│
├── app/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Create virtual environment

```bash
py -3.11 -m venv .venv
```

### 2. Activate virtual environment

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place dataset

Download dataset manually from UCI and place it at:

```text
data/raw/clickstream_raw.csv
```

### 5. Run EDA

```bash
python scripts/01_initial_eda.py
```

### 6. Build session-level dataset

```bash
python scripts/02_feature_engineering.py
```

### 7. Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Current Status

Current weekly progress:

1. Project topic selected.
2. Dataset selected.
3. Folder structure created.
4. Initial EDA script created.
5. Initial feature engineering script created.
6. Session-level dataset generation prepared.
7. Streamlit starter dashboard created.

## Next Steps

1. Complete full EDA.
2. Build preprocessing pipeline.
3. Train classification models.
4. Train clustering models.
5. Compare models.
6. Add hyperparameter tuning.
7. Add feature selection.
8. Prepare Docker deployment.
```

---

# BAGIAN N
# Membuat WEEKLY_PROGRESS.md

---

## 22. Buat File WEEKLY_PROGRESS.md

Buat file:

```text
WEEKLY_PROGRESS.md
```

Isi:

```markdown
# Weekly Progress Report

## Project Title

Clickstream Behavior Intelligence App: Classification and Clustering for User Behavior Analysis

## Dataset

Dataset used:

Clickstream Data for Online Shopping  
Source: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping

The dataset contains user clickstream behavior from an online shopping website.

## Work Completed This Week

### 1. Project Topic Selection

The project topic has been selected as:

Clickstream Behavior Intelligence App

The project focuses on user behavior analysis using classification and clustering.

### 2. Dataset Selection

The selected dataset is Clickstream Data for Online Shopping from UCI Machine Learning Repository.

Reasons for selecting this dataset:

1. It contains behavioral data.
2. It is suitable for classification and clustering.
3. It can be connected to log analysis methodology.
4. It is relevant for learning machine learning pipelines.

### 3. Project Scope Definition

The project scope includes:

1. EDA
2. Preprocessing
3. Feature engineering
4. Classification modeling
5. Clustering modeling
6. Hyperparameter tuning
7. Feature selection
8. Streamlit dashboard
9. Docker preparation

### 4. Project Structure Preparation

The folder structure has been prepared:

- app
- data
- models
- notebooks
- reports
- scripts
- src

### 5. Environment Preparation

Python virtual environment has been prepared using Python 3.11.

### 6. Initial Scripts Created

Initial scripts prepared:

1. scripts/01_initial_eda.py
2. scripts/02_feature_engineering.py

### 7. Initial Streamlit App Created

A basic Streamlit dashboard has been created to display:

1. project overview,
2. dataset source,
3. session-level dataset preview,
4. basic statistics,
5. target distribution.

## Current Status

The project is currently in the initial implementation phase.

Completed:

1. Topic selection.
2. Dataset selection.
3. Project structure.
4. Environment setup.
5. Initial EDA script.
6. Initial feature engineering script.
7. Starter dashboard.

In progress:

1. Full EDA.
2. Preprocessing.
3. Classification modeling.
4. Clustering modeling.

## Next Week Plan

Next week, the project will focus on:

1. Completing EDA visualizations.
2. Cleaning and preprocessing data.
3. Building session-level features.
4. Creating classification targets.
5. Training baseline classification models.
6. Training baseline clustering models.
7. Updating Streamlit dashboard.

## Short Weekly Submission Explanation

This week, I started the Clickstream Behavior Intelligence App project. The project focuses on machine learning for user behavior analysis using classification and clustering. I selected the Clickstream Data for Online Shopping dataset from the UCI Machine Learning Repository because it contains behavioral clickstream data and is suitable for classification and clustering experiments.

The current progress includes project topic selection, dataset selection, scope definition, project structure preparation, environment setup, initial EDA script, initial feature engineering script, and a starter Streamlit dashboard.
```

---

# BAGIAN O
# Docker Preparation

---

## 23. Buat Dockerfile

Buat file:

```text
Dockerfile
```

Isi:

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

Build Docker image:

```powershell
docker build -t clickstream-behavior-ml .
```

Run container:

```powershell
docker run -p 8080:8080 clickstream-behavior-ml
```

Buka:

```text
http://localhost:8080
```

Catatan:

Docker belum wajib untuk weekly progress awal. Prioritas saat ini adalah app jalan secara lokal dan repo GitHub siap dishare.

---

# BAGIAN P
# GitHub Submission

---

## 24. Inisialisasi Git

Jalankan di root project:

```powershell
git init
git add .
git commit -m "Initial weekly progress: setup clickstream behavior ML project"
```

Buat repository baru di GitHub:

```text
clickstream-behavior-ml
```

Hubungkan local repo ke GitHub:

```powershell
git branch -M main
git remote add origin https://github.com/USERNAME/clickstream-behavior-ml.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub.

---

## 25. Apa yang Dikirim ke Lektor?

Kirim:

1. Link GitHub repository.
2. Penjelasan singkat progress.
3. Link dataset UCI.
4. Screenshot Streamlit app jika perlu.

Contoh penjelasan:

```text
This week, I developed the initial version of the Clickstream Behavior Intelligence App. The project uses the Clickstream Data for Online Shopping dataset from UCI Machine Learning Repository. The project focuses on classification and clustering for user behavior analysis.

The current progress includes project setup, dataset preparation, folder structure, Python environment setup, initial EDA script, feature engineering script, session-level dataset preparation, Streamlit dashboard, README documentation, and weekly progress documentation.

The next development phase will focus on full EDA visualization, preprocessing, classification model comparison, clustering model comparison, hyperparameter tuning, and feature selection.
```

---

# BAGIAN Q
# Urutan Eksekusi Lengkap dari Nol

---

## 26. Checklist Command Lengkap

Jalankan dari awal:

```powershell
cd "C:\All Documents\Kuliah\Semester 6\Bengkel Koding\Projek"

mkdir clickstream-behavior-ml
cd clickstream-behavior-ml

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
```

Buat folder:

```powershell
mkdir app
mkdir data
mkdir data\raw
mkdir data\interim
mkdir data\processed
mkdir models
mkdir notebooks
mkdir reports
mkdir reports\figures
mkdir reports\tables
mkdir scripts
mkdir src
```

Buat file:

```text
requirements.txt
.gitignore
.dockerignore
README.md
WEEKLY_PROGRESS.md
Dockerfile
scripts/01_initial_eda.py
scripts/02_feature_engineering.py
src/preprocessing.py
src/feature_engineering.py
src/data_loader.py
src/evaluation.py
app/streamlit_app.py
```

Install library:

```powershell
pip install -r requirements.txt
```

Download dataset manual dari UCI, lalu simpan sebagai:

```text
data/raw/clickstream_raw.csv
```

Jalankan EDA:

```powershell
python scripts/01_initial_eda.py
```

Jalankan feature engineering:

```powershell
python scripts/02_feature_engineering.py
```

Jalankan full EDA + baseline modeling:

```powershell
python scripts/03_full_eda.py
python scripts/04_baseline_classification.py
python scripts/05_baseline_clustering.py
```

Bangun insight assets (cluster profiling + action list):

```powershell
python scripts/06_build_insight_assets.py
```

Latih model prediksi high_engagement dan buat output prediksi:

```powershell
python scripts/07_train_high_engagement_model.py
python scripts/08_predict_high_engagement.py
```

Jalankan Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

Push ke GitHub:

```powershell
git init
git add .
git commit -m "Initial weekly progress: setup clickstream behavior ML project"
git branch -M main
git remote add origin https://github.com/USERNAME/clickstream-behavior-ml.git
git push -u origin main
```

---

# BAGIAN R
# Troubleshooting

---

## 27. Error: Python Versi 3.13

Masalah:

Beberapa library ML belum tentu stabil di Python 3.13.

Solusi:

Gunakan Python 3.11:

```powershell
py -3.11 -m venv .venv
```

---

## 28. Error: Dataset Tidak Ditemukan

Error:

```text
FileNotFoundError: Raw dataset not found.
```

Solusi:

Pastikan file berada di:

```text
data/raw/clickstream_raw.csv
```

Pastikan nama file tidak salah, misalnya jangan:

```text
clickstream_raw.csv.csv
```

---

## 29. Error: Kolom Tidak Ditemukan

Error:

```text
Missing required columns
```

Penyebab:

Nama kolom dataset berbeda atau file salah.

Solusi:

Jalankan EDA:

```powershell
python scripts/01_initial_eda.py
```

Lihat nama kolom pada output terminal. Pastikan file yang digunakan adalah `e-shop clothing 2008.csv` dari UCI.

---

## 30. Error: Streamlit Tidak Dikenali

Error:

```text
streamlit is not recognized
```

Solusi:

Pastikan virtual environment aktif:

```powershell
.venv\Scripts\activate
```

Install ulang:

```powershell
pip install streamlit
```

Jalankan:

```powershell
python -m streamlit run app/streamlit_app.py
```

---

## 31. Error: Git Push Ditolak

Solusi umum:

Cek remote:

```powershell
git remote -v
```

Jika salah:

```powershell
git remote remove origin
git remote add origin https://github.com/USERNAME/clickstream-behavior-ml.git
```

Push ulang:

```powershell
git push -u origin main
```

---

# BAGIAN S
# Roadmap Setelah Weekly Progress

---

## 32. Phase Berikutnya

Setelah weekly progress terkirim, lanjutkan fase berikut:

### Phase 2 — EDA Lengkap

Tambahkan visualisasi:

1. Distribusi total klik per sesi.
2. Top country.
3. Top product category.
4. Distribusi harga.
5. Jumlah klik per bulan.
6. Jumlah klik per halaman.
7. Korelasi fitur numerik.

### Phase 3 — Modeling Classification

Model awal:

1. Dummy Classifier
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. KNN
6. SVM
7. Gradient Boosting

### Phase 4 — Modeling Clustering

Model awal:

1. K-Means
2. MiniBatch K-Means
3. Agglomerative Clustering
4. DBSCAN
5. Gaussian Mixture Model

### Phase 5 — Evaluation

Classification metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. Confusion matrix

Clustering metrics:

1. Silhouette Score
2. Davies-Bouldin Index
3. Calinski-Harabasz Index
4. Cluster profile

### Phase 6 — Streamlit Enhancement

Tambahkan:

1. Grafik EDA.
2. Pilihan target classification.
3. Hasil model.
4. Visualisasi clustering.
5. Interpretasi cluster.
6. Feature importance.

---

# BAGIAN T
# Prinsip Teknis Penting

---

## 33. Jangan Training Model Berat di Streamlit

Prinsip penting:

```text
Training dilakukan di script/notebook.
Streamlit hanya untuk loading hasil dan visualisasi.
```

Alasan:

1. App lebih ringan.
2. Tidak crash saat demo.
3. Tidak timeout.
4. Cocok untuk deployment.
5. Lebih profesional.

Contoh menyimpan model:

```python
import joblib

joblib.dump(model, "models/final_classification_model.joblib")
```

Contoh load model di Streamlit:

```python
import joblib

model = joblib.load("models/final_classification_model.joblib")
```

---

## 34. Hindari Data Leakage

Jika target dibuat dari suatu fitur, fitur tersebut tidak boleh digunakan sebagai input.

Contoh:

Target:

```text
premium_interest
```

Dibuat dari:

```text
premium_click_ratio
```

Maka saat training model untuk `premium_interest`, fitur `premium_click_ratio` harus dikeluarkan.

---

## 35. Gunakan Relative Path

Jangan gunakan path seperti:

```text
C:\Users\Nama\Desktop\file.csv
```

Gunakan:

```text
data/raw/clickstream_raw.csv
```

Agar proyek bisa dijalankan di laptop lain.

---

# BAGIAN U
# Kesimpulan Progress Saat Ini

---

## 36. Yang Sudah Dikerjakan

Dalam proyek ini, tahap awal yang dikerjakan adalah:

1. Menentukan topik proyek.
2. Memilih dataset.
3. Menentukan arah proyek classification dan clustering.
4. Menghubungkan proyek dengan cyber-inspired behavioral log analysis.
5. Menentukan Python 3.11 sebagai environment.
6. Menentukan manual dataset download sebagai solusi.
7. Menyusun struktur folder.
8. Menyusun script EDA awal.
9. Menyusun utility preprocessing.
10. Menyusun utility feature engineering.
11. Menyusun script pembuatan session-level dataset.
12. Menyusun app Streamlit awal.
13. Menyusun README.
14. Menyusun WEEKLY_PROGRESS.
15. Menyusun rencana Docker.
16. Menyusun rencana GitHub submission.

---

## 37. Target Weekly Submission

Minimal hasil yang ditunjukkan ke lektor:

1. Repository GitHub.
2. README jelas.
3. WEEKLY_PROGRESS jelas.
4. Struktur folder rapi.
5. Streamlit app bisa jalan.
6. Ada script EDA.
7. Ada script feature engineering.
8. Ada penjelasan dataset.
9. Ada rencana pengembangan berikutnya.

---

## 38. Prompt Lanjutan untuk ChatGPT

Jika ingin melanjutkan proyek ini di akun lain, gunakan prompt berikut:

```text
Saya sedang membuat proyek machine learning dari nol bernama Clickstream Behavior Intelligence App. Proyek ini memakai dataset Clickstream Data for Online Shopping dari UCI. Fokusnya adalah classification dan clustering untuk analisis perilaku pengguna berbasis clickstream. Dataset didownload manual dari UCI dan disimpan sebagai data/raw/clickstream_raw.csv.

Struktur folder sudah dibuat dengan app, data, scripts, src, reports, models, dan notebooks. Environment menggunakan Python 3.11. File awal yang dibuat adalah scripts/01_initial_eda.py, scripts/02_feature_engineering.py, src/preprocessing.py, src/feature_engineering.py, src/data_loader.py, src/evaluation.py, app/streamlit_app.py, README.md, WEEKLY_PROGRESS.md, requirements.txt, Dockerfile, .gitignore, dan .dockerignore.

Bantu saya melanjutkan step by step mulai dari EDA lengkap, preprocessing, classification modeling, clustering modeling, hyperparameter tuning, feature selection, Streamlit enhancement, Docker testing, dan GitHub deployment.
```

---

# Selesai

Dokumentasi ini dapat dijadikan panduan utama untuk membangun proyek dari nol, melanjutkan proyek di akun lain, atau menjelaskan progress kepada dosen/lektor.
