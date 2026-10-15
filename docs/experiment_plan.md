# Rencana Eksperimen: Imbalance dan Generalisasi Antar Pasien, PTB-XL

## Pertanyaan penelitian
1. Strategi penanganan imbalance mana yang paling meningkatkan kinerja kelas jarang
   pada EKG 12-lead, tanpa merusak kelas mayoritas?
2. Seberapa stabil hasilnya pada pasien yang tidak pernah dilihat model?

## Pembagian tugas label
- **Tugas utama: `sub` dengan `--min-count 150`.** Dari 23 subclass, 15 kelas lolos
  (NORM, AMI, IMI, STTC, LVH, LAFB/LPFB, ISC_, IRBBB, NST_, _AVB, ISCA, IVCD, CRBBB,
  CLBBB, LAO/LAE). Rasio kelas terbesar (NORM, 3.872) terhadap terkecil (LAO/LAE, 157)
  sekitar 25:1, dan tiap kelas punya minimal sekitar 16 contoh di fold 10.
  Delapan kelas lain dibuang karena contohnya terlalu sedikit untuk dievaluasi
  (misalnya PMI punya 0 contoh uji, WPW dan ILBBB hanya 1).
- **Tugas pembanding: `super` (5 superclass).** Imbalance-nya ringan, sekitar 4:1
  (NORM 48,8% vs HYP 12%), jadi dipakai untuk melihat apakah strategi imbalance
  masih berguna saat ketidakseimbangannya kecil.
- Rekaman yang labelnya hanya berisi kelas yang dibuang ikut dibuang. Jumlahnya dicatat
  di `info.json` (`n_dropped_rare_only`).
- Distribusi akhir selalu dicek di `data/processed/.../info.json` sebelum masuk slide.

## Cakupan data (PTB-XL v1.0.1, sebagian)
Sinyal 100 Hz tersedia untuk 8.031 dari 21.837 rekaman (7.183 pasien). Fold resmi tetap
dipakai, jumlah rekaman tersedia per fold: 1:766, 2:762, 3:802, 4:810, 5:796, 6:770,
7:760, 8:734, 9:910, 10:921. Akibatnya:
- Uji (fold 10) sekitar 921 rekaman, jadi kelas jarang punya sedikit contoh uji.
  Ditangani dengan `--min-count 150` dan metrik gabungan lintas rotasi fold (lihat di bawah).
- Metrik kelas jarang berfluktuasi besar. Itu alasan memakai rotasi fold dan banyak seed.
- Nyatakan di slide 4 bahwa data adalah subset (v1.0.1), bukan seluruh PTB-XL.

## Split dan generalisasi antar pasien (slide 4)
- Fold resmi PTB-XL (`strat_fold`), dipisah per pasien: latih 1-8, validasi 9, uji 10.
- `make_split` memeriksa tidak ada pasien yang beririsan antar split.
- Evaluasi stabilitas: 10 rotasi fold (`rotating_splits`), laporkan rata-rata ± simpangan baku.
- **Metrik per kelas dihitung dari gabungan prediksi uji seluruh rotasi fold.** Tiap rekaman
  jadi data uji tepat sekali, sehingga per kelas ada ratusan sampai ribuan contoh, bukan
  hanya isi fold 10 (sekitar 920 rekaman). Fold validasi tetap terpisah untuk memilih threshold.
- Rotasi penuh tidak untuk semua strategi. Jalankan semua strategi di satu split (fold 10)
  dulu, lalu rotasi penuh dan multi-seed hanya untuk 2-3 strategi terbaik.
- Beberapa seed per konfigurasi akhir (minimal 3).
- Normalisasi dan threshold hanya dari fold latih/validasi. Fold 10 disentuh sekali di akhir.
- Analisis subkelompok pada uji: jenis kelamin dan kelompok usia.

## Preprocessing
- [ ] Muat 100 Hz, 12 lead, 1000 sampel
- [ ] Label multi-hot (super dan sub), buang rekaman tanpa label diagnostik
- [ ] Z-score per lead dari fold latih
- [ ] Opsional: band-pass 0.5-40 Hz (ablation)

## Model (secukupnya, fokus ada di strategi imbalance)
- [ ] Baseline: MLP / fitur sederhana
- [ ] 1D-CNN
- [ ] ResNet-1D (model utama untuk perbandingan strategi)
- [ ] Opsional: LSTM atau Transformer kecil

## Strategi imbalance (inti eksperimen)
| Kode | Strategi |
|---|---|
| B0 | BCE tanpa penanganan |
| B1 | BCE dengan pos_weight (rasio neg/pos dari data latih) |
| B2 | Focal loss |
| B3 | Asymmetric loss |
| B4 | Class-balanced loss |
| B5 | Oversampling kelas jarang (hanya data latih, per batch sampler) |
| B6 | Augmentasi (noise, scaling, time shift) + salah satu di atas |
| B7 | Threshold per kelas, dipilih di fold validasi |

Catatan: resampling pada data multi-label harus berbasis rekaman, bukan per label.
Semua penanganan hanya menyentuh data latih.

## Ablation lain
- [ ] Dengan vs tanpa band-pass
- [ ] Hyperparameter: filter, layer, kernel, dropout, learning rate
- [ ] Panjang sinyal dan jumlah lead (opsional, jika waktu ada)

## Metrik
Macro AUROC dan macro AUPRC (lebih informatif untuk kelas jarang), macro-F1 dengan
threshold dari validasi, sensitivitas dan spesifisitas per kelas, kinerja pada kelas
paling jarang. Akurasi tidak dipakai.

## Jadwal (sampai Jumat, 16 Okt 2026)
| Hari | Target |
|---|---|
| Senin 12 Okt | Dataset terunduh, check_data dan preprocess jalan, distribusi kelas terlihat |
| Selasa 13 Okt | train.py jadi, baseline dan ResNet-1D B0 terlatih |
| Rabu 14 Okt | Semua strategi B1-B7, rotasi fold untuk konfigurasi terbaik |
| Kamis 15 Okt | Analisis, subkelompok, slide, push repo |
| Jumat 16 Okt | Latihan presentasi, presentasi |

## Pemetaan slide
1 Identitas · 2 Jobdesc · 3 Masalah (imbalance + generalisasi) · 4 Data dan split ·
5 Arsitektur · 6 Strategi belajar dan tuning · 7 Hasil perbandingan strategi imbalance ·
8 Hasil per kelas dan stabilitas antar fold · 9 Analisis subkelompok dan error ·
10 Kesimpulan · 11 URL kode · 12 Referensi
