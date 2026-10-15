# Penanganan Ketidakseimbangan Kelas dan Generalisasi Antar Pasien pada Klasifikasi EKG 12-Lead (PTB-XL)

Group Mini Project PMML, Magister Ilmu Komputer UGM.
Presentasi: Jumat, 16 Oktober 2026.

## Masalah
Klasifikasi multi-label kelainan jantung dari EKG 12-lead 10 detik. Fokus proyek:
1. **Imbalance:** membandingkan strategi penanganan kelas tidak seimbang
   (class weight, focal loss, asymmetric loss, resampling, augmentasi).
2. **Generalisasi antar pasien:** evaluasi pada pasien yang tidak pernah dilihat model,
   memakai fold resmi PTB-XL yang dipisah per pasien.

Dua tugas label:
- `sub` (utama, `--min-count 150`): 15 subclass diagnostik dengan imbalance sekitar 25:1.
- `super` (pembanding): 5 superclass dengan imbalance ringan (sekitar 4:1).

## Dataset
PTB-XL versi **1.0.1** (PhysioNet), ada di `data/raw/ptb-xl/` (tidak ikut di-commit).
Yang dipakai: `ptbxl_database.csv`, `scp_statements.csv`, dan `records100/` (100 Hz).

Salinan ini **hanya sebagian**: file sinyal tersedia untuk 8.031 dari 21.837 rekaman
(ID 00001 sampai 08031). `preprocess.py` otomatis hanya memakai rekaman yang sinyalnya
lengkap. Semua file sinyal sudah dicocokkan dengan `SHA256SUMS.txt` resmi dan utuh.

```bash
python src/check_data.py
```

## Struktur
```
src/
  check_data.py    verifikasi data hasil unduhan manual
  preprocess.py    sinyal -> X.npy, label multi-hot, meta
  data_utils.py    muat data, split per fold, normalisasi, bobot kelas
  models.py        MLP, CNN1D, ResNet1D
  losses.py        BCE, pos_weight, focal, asymmetric, class-balanced
  metrics.py       AUROC/AUPRC per kelas, threshold per kelas
  train.py         training satu konfigurasi pada satu fold uji
  evaluate.py      metrik per kelas, rotasi fold, subkelompok, gambar
notebooks/         eksplorasi dan visualisasi
configs/           konfigurasi eksperimen (YAML)
data/              raw dan processed (di-ignore git)
results/           tabel, gambar, checkpoint
```

## Cara menjalankan
```bash
pip install -r requirements.txt
python src/check_data.py
python src/preprocess.py --task super                      # pembanding
python src/preprocess.py --task sub --min-count 150        # tugas utama
```

`--min-count N` membuang kelas dengan contoh kurang dari N (dihitung dari rekaman yang tersedia),
lalu membuang rekaman yang labelnya habis. Keluaran di `data/processed/ptbxl_<task>_<sr>hz[_minN]/`:
`X.npy` (N, 1000, 12), `meta.npz` (`y`, `patient_id`, `strat_fold`, `age`, `sex`, `ecg_id`),
dan `info.json` (kelas dan jumlah per kelas).

Normalisasi tidak dilakukan di preprocess. Statistik z-score dihitung dari fold latih saja
lewat `data_utils.fit_norm`, supaya tidak ada kebocoran.

## Training dan evaluasi
Satu run = satu konfigurasi (YAML di `configs/`) pada satu fold uji:

```bash
python src/train.py --config configs/b0_bce.yaml                    # B0, fold uji 10
python src/train.py --config configs/b1_pos_weight.yaml --test-fold 3 --seed 1
python src/evaluate.py results/runs/ptbxl_sub_100hz_min150/b0_bce_f10_s0
```

Config tersedia: `b0_bce`, `b1_pos_weight`, `b2_focal`, `b3_asl`, `b4_class_balanced`,
`b5_oversample`, `b6_augment`, `baseline_mlp`, `baseline_cnn1d`. Argumen CLI menimpa YAML.
B7 (threshold per kelas) dikerjakan di `evaluate.py`, dipilih dari fold validasi.

Cara paling mudah: jalankan semua strategi berurutan dengan satu perintah (lihat bagian berikut).
Bandingkan banyak run, atau gabungkan 10 rotasi fold (tiap rekaman diuji sekali):

```bash
python src/evaluate.py results/runs/*/b*_f10_s0 --name banding_strategi --no-aggregate
python src/evaluate.py results/runs/*/b0_bce_f*_s0 --pool --subgroups --plot --name b0_rotasi
```

## Menjalankan semua strategi
```bash
python src/run_all.py --dry-run                                  # lihat daftar run
python src/run_all.py                                            # 9 config, fold uji 10, seed 0
python src/run_all.py --configs b0_bce b3_asl --folds 1-10       # rotasi 10 fold, tabel perbandingan
python src/run_all.py --seeds 0 1 2 --configs b0_bce b3_asl      # beberapa seed
```
Run yang sudah selesai dilewati, jadi aman dihentikan (Ctrl-C) dan dijalankan ulang.
Nama run: `<config>_f<fold>_s<seed>`. Log di `results/logs/`, tabel di `results/tables/`.
Satu fold menghasilkan `perbandingan_f10_s0_summary.csv`. Banyak fold menghasilkan
`perbandingan_rotasi_s0.csv` (tiap rekaman diuji sekali).

Fold validasi untuk fold uji K adalah fold K-1 (uji 10 memakai validasi 9, uji 1 memakai validasi 10).
Early stopping dan pemilihan checkpoint memakai macro AUPRC validasi. Tabel ke
`results/tables/`, gambar ke `results/figures/`, `best.pt` di-ignore git.

Torch yang terpasang di laptop ini versi CPU. Satu epoch data latih penuh sekitar 17-60 detik
(epoch pertama lebih lambat), jadi satu run 30 epoch sekitar 10 menit. Untuk rotasi fold dan
banyak seed, pertimbangkan Colab GPU.

## Mengumpulkan hasil
```bash
python src/collect_results.py      # results/summary.json, results/tables/all_runs.csv, by_config_fold10.csv
```
Skrip ini membaca semua folder di `results/runs/` langsung (tidak bergantung pada tabel perbandingan
yang bisa tertimpa) dan menjadi sumber data untuk slide.

## Ringkasan hasil (fold uji 10, seed 0-2, subset PTB-XL v1.0.1)
- CNN1D mengungguli ResNet-1D di ketiga seed (macro AUPRC 0,592 vs 0,556 dengan BCE).
- Loss khusus imbalance tidak konsisten lebih baik dari BCE pada CNN1D (selisih -0,010 sampai +0,002).
- Threshold per kelas dari validasi menaikkan macro F1 sekitar 0,06 untuk loss tanpa koreksi bobot.
- Rotasi 10 fold (BCE dan pos_weight): macro AUPRC 0,608 ± 0,021 vs 0,611 ± 0,020; variasi antar fold
  lebih besar dari selisih antar strategi. AUROC lebih rendah pada usia >= 80.

## Split
Fold 1-8 latih, fold 9 validasi, fold 10 uji (fold resmi PTB-XL, dipisah per pasien).
`data_utils.make_split` memeriksa tidak ada pasien yang muncul di dua split.
`data_utils.rotating_splits` menyediakan 10 rotasi untuk melaporkan rata-rata dan simpangan.

## Tim
Lihat `docs/jobdesc.md`.

## Disclaimer
Hanya untuk keperluan akademik, bukan alat diagnosis medis.
