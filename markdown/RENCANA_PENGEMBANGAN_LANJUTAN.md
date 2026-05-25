# Rencana Pengembangan Lanjutan

Dokumen ini merangkum rencana pengembangan ke depan dan daftar pekerjaan yang belum selesai.

---

## 1. Tujuan Pengembangan Lanjutan

Fokus utama pengembangan berikutnya adalah menjadikan proyek lebih aplikatif untuk dunia nyata (UX/Product) dengan output yang jelas dan bisa digunakan oleh tim non-teknis.

Target hasil akhir:
1. Insight perilaku pengguna yang mudah dipahami.
2. Segmentasi pengguna yang dapat diberi label dan interpretasi.
3. Action list yang bisa langsung dipakai untuk eksperimen dan optimasi.
4. Dashboard yang menyatukan data, model, dan rekomendasi.

---

## 2. Rencana Pengembangan yang Belum Selesai

### 2.1 Preprocessing Lanjutan
- Validasi tipe data dan missing value secara lebih sistematis.
- Penanganan outlier untuk fitur harga dan click count.
- Konsistensi encoding untuk fitur kategori.

### 2.2 Modeling Classification (Baseline + Tuning)
- Jalankan baseline classification untuk semua target yang valid.
- Lakukan hyperparameter tuning pada model terbaik.
- Tambahkan metrik tambahan (ROC-AUC, PR-AUC jika imbalanced).

### 2.3 Modeling Clustering (Baseline + Profiling)
- Uji variasi jumlah cluster dan konfigurasi DBSCAN.
- Tambahkan interpretasi cluster otomatis (label cluster).
- Tambahkan visualisasi cluster menggunakan PCA atau t-SNE.

### 2.4 Insight dan Storytelling
- Tambahkan narasi insight yang lebih lengkap.
- Buat ringkasan eksekutif untuk non-teknis.
- Hubungkan temuan ke rekomendasi produk/UX.

### 2.5 Streamlit Enhancement
- Tambah menu navigasi (multi-page).
- Tambah section download CSV (action list dan cluster profile).
- Tambah visual interaktif (Plotly).

---

## 3. Rencana Pengembangan ke Depan (Future Roadmap)

### Phase A - Data Quality & Insight
1. Audit kualitas data otomatis.
2. EDA dengan grafik interaktif.
3. Validasi target engineered agar balance.

### Phase B - Model Improvement
1. Coba model tambahan (XGBoost/LightGBM jika diizinkan).
2. Feature selection untuk kurangi noise.
3. Simpan model terbaik ke folder models.

### Phase C - Segmentation & Actionability
1. Tetapkan label segmen cluster.
2. Tambah analisis perilaku spesifik per segmen.
3. Buat action plan per segmen.

### Phase D - Deployment Readiness
1. Docker build + run untuk demo.
2. Dokumentasi deployment sederhana.
3. Simulasi penggunaan oleh reviewer/dosen.

---

## 4. Checklist yang Belum Diselesaikan

- [ ] Hyperparameter tuning untuk classification
- [ ] Visualisasi cluster (PCA/t-SNE)
- [ ] Label cluster otomatis
- [ ] Export action list dari dashboard
- [ ] Insight narrative yang lebih lengkap
- [ ] Dokumentasi deployment

---

## 5. Output Akhir yang Diharapkan

1. Dashboard Streamlit lengkap (EDA + model + insight).
2. Action list siap pakai.
3. Segmentasi perilaku dengan label yang jelas.
4. Dokumentasi lengkap untuk pemula.
5. Reproducible pipeline (scripts + Docker).

---

## 6. Catatan

Jika target premium_interest masih satu kelas, revisi threshold atau definisi target agar usable untuk classification.
