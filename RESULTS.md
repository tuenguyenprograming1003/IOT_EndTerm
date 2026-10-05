# Kết quả — C3 Task-driven Log-Mel Feature Compression (FSDD, 6-fold LOSO)

Nguồn số liệu: `results/summary/results_summary.csv` (mỗi dòng = method × fold × seed × d),
`results_summary_mean.csv`, `report_table_seed0.csv`, `results/predictions/all_predictions.csv`.
Hình: `results/figures/`. Notebook: `notebooks/07_result_analysis.ipynb`.

Thiết lập: đặc trưng `S = CMVN(log(P + 1e-6))` 64×32 (lệch khỏi đặc tả gốc `log1p(P)`, xem README),
normalization theo Train từng fold; classifier tham chiếu đóng băng, cùng cho mọi phương pháp trong
một fold/seed; checkpoint luôn chọn bằng Validation; Test chỉ dùng để báo cáo.

## 1. Classifier tham chiếu (A0, Original FP32 8196 B)

| test speaker | george | jackson | lucas | nicolas | theo | yweweler | mean ± std |
|---|---|---|---|---|---|---|---|
| A0 seed 0 | 51.4 | 56.2 | 24.0 | 27.0 | 43.8 | 51.6 | 42.3 ± 13.7 |
| A0 seeds 0–2 (18 runs) | | | | | | | 44.2 ± 12.3 |

Mức may rủi = 10 %. Bài toán unseen-speaker với 4 speaker train là khó; biến thiên giữa speaker lớn
(lucas, nicolas thấp). Mọi kết luận bên dưới là **tương đối** so với A0 này, không phải accuracy tuyệt đối cao.

## 2. Kết quả chính (Test, seed 0, mean ± std qua 6 fold)

| method | d | bytes | CR_INT8 | CR_FP32 | Accuracy | Macro-F1 | ΔA_pp | R_acc (%) | MSE | PRD_spec (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | – | 8196 | 0.25 | 1.0 | 0.423 ± 0.137 | 0.373 | 0.0 | 100.0 | 0 | 0 |
| INT8 | – | 2052 | 1.0 | 4.0 | 0.422 ± 0.138 | 0.372 | −0.1 | 99.7 | 0.0003 | 1.8 |
| **Task-AE** | 64 | 68 | 30.2 | 120.5 | **0.431 ± 0.141** | **0.388** | +0.8 | 103.3 | 0.173 | 38.0 |
| AE-MSE | 64 | 68 | 30.2 | 120.5 | 0.345 ± 0.122 | 0.292 | −7.9 | 80.7 | 0.106 | 29.5 |
| DCT | 64 | 68 | 30.2 | 120.5 | 0.318 ± 0.108 | 0.258 | −10.5 | 74.8 | 0.132 | 32.9 |
| Low-Mel | 64 | 68 | 30.2 | 120.5 | 0.100 ± 0.000 | 0.018 | −32.3 | 26.4 | 0.992 | 94.7 |
| **Task-AE** | 128 | 132 | 15.5 | 62.1 | **0.427 ± 0.147** | **0.388** | +0.4 | 100.3 | 0.124 | 31.9 |
| AE-MSE | 128 | 132 | 15.5 | 62.1 | 0.404 ± 0.134 | 0.359 | −1.9 | 95.1 | 0.068 | 23.8 |
| DCT | 128 | 132 | 15.5 | 62.1 | 0.391 ± 0.140 | 0.347 | −3.2 | 91.3 | 0.083 | 26.6 |
| Low-Mel | 128 | 132 | 15.5 | 62.1 | 0.102 ± 0.004 | 0.020 | −32.2 | 26.7 | 0.960 | 91.3 |
| **Task-AE** | 256 | 260 | 7.9 | 31.5 | **0.445 ± 0.144** | **0.406** | +2.2 | 105.4 | 0.111 | 30.7 |
| AE-MSE | 256 | 260 | 7.9 | 31.5 | 0.415 ± 0.143 | 0.368 | −0.8 | 97.3 | 0.047 | 20.1 |
| DCT | 256 | 260 | 7.9 | 31.5 | 0.416 ± 0.139 | 0.370 | −0.7 | 97.8 | 0.054 | 21.9 |
| Low-Mel | 256 | 260 | 7.9 | 31.5 | 0.106 ± 0.016 | 0.023 | −31.7 | 27.8 | 0.896 | 83.8 |

ΔA_pp tính bằng **điểm phần trăm** (Acomp − A0). CR theo byte thực: 2052/(d+4), 8196/(d+4).
PRD_spec là sai số biểu diễn log-mel (miền S), không phải chất lượng âm thanh.

Hình chính: `accuracy_vs_bytes.png`, `compression_vs_retention.png`.

## 3. Task-AE so với AE-MSE

* Theo speaker (`task_gain_by_speaker.png`), seed 0: gain trung bình +8.7 pp (d=64), +2.3 pp (d=128),
  +3.0 pp (d=256); Task-AE tốt hơn ở **5/6 fold cho mỗi d**. Ngoại lệ: george (d=64, 128), theo (d=256).
* Multi-seed d=128 (3 seed). **Đơn vị suy luận = test speaker**: lấy trung bình 3 seed cho mỗi speaker
  → 6 cặp Task-AE/AE-MSE (`task_vs_mse_d128_speaker_level.csv`):

  | speaker | george | jackson | lucas | nicolas | theo | yweweler | mean ± std |
  |---|---|---|---|---|---|---|---|
  | AE-MSE | 50.9 | 54.8 | 25.2 | 29.5 | 46.6 | 46.1 | 42.2 |
  | Task-AE | 50.2 | 60.7 | 27.5 | 31.3 | 46.9 | 51.6 | 44.7 |
  | gain (pp) | −0.7 | +5.9 | +2.3 | +1.8 | +0.3 | +5.5 | **+2.53 ± 2.69** |

  Task-AE cao hơn ở 5/6 speaker. Wilcoxon signed-rank chính xác hai phía (n = 6): **p = 0.094**; paired
  t-test: t(5) = 2.30, p = 0.069. **Chưa đạt mức ý nghĩa 0.05** — với chỉ 6 speaker, kiểm định có lực
  thấp (Wilcoxon n = 6 có p nhỏ nhất đạt được là 0.031). Kết quả được xem là xu hướng nhất quán,
  không phải khác biệt có ý nghĩa thống kê.
* Mô tả (không dùng để kiểm định, vì 3 seed của cùng một speaker không độc lập): trên 18 cặp
  (seed, speaker), gain trung bình +2.53 pp, Task-AE thắng 16/18 cặp. Accuracy trung bình d=128 qua
  3 seed: Task-AE 0.447, AE-MSE 0.422, DCT 0.423, Original 0.442.
* MSE vs accuracy (`mse_vs_accuracy.png`): ở **15/18** cặp (fold, d) của seed 0, Task-AE có **MSE cao
  hơn nhưng accuracy cao hơn** AE-MSE → trong thiết lập này, MSE thấp hơn không đồng nghĩa phân loại
  tốt hơn; phù hợp với giả thuyết nén định hướng nhiệm vụ.
* Accuracy retention của Task-AE (seed 0): **không quan sát thấy suy giảm accuracy trung bình so với
  đặc trưng gốc; retention ≈ 101–105 %**:

  | packet | Task-AE acc | A0 | Acc/A0 (tỷ số các trung bình) | trung bình R_acc theo fold |
  |---|---|---|---|---|
  | 68 B | 43.1 % | 42.3 % | 101.9 % | 103.3 % |
  | 132 B | 42.7 % | 42.3 % | 100.9 % | 100.3 % |
  | 260 B | 44.5 % | 42.3 % | 105.2 % | 105.4 % |

  Phần vượt 100 % nằm trong độ lệch chuẩn giữa fold. Không diễn giải là nén "cải thiện thông tin"
  theo nghĩa tuyệt đối; khả năng hợp lý là hiệu ứng regularization/denoising đối với classifier cố định
  (decoder học đưa x̂ về vùng mà classifier đó phân loại tốt).

## 4. Baselines

* Direct INT8 (2052 B): gần như không mất gì (ΔA = −0.1 pp) → lợi ích của codec đến từ giảm chiều,
  không phải từ INT8.
* DCT zig-zag ngang AE-MSE ở d=256, kém hơn ở d=64.
* **Low-Mel ≈ 10 % (mức may rủi) ở mọi d**: giữ m = d/32 = 2, 4, 8 hàng mel thấp nhất và đặt phần còn
  lại = 0 làm đầu vào lệch xa phân bố classifier được train → classifier cố định không dùng được. Kết quả
  này được giữ nguyên, không loại bỏ.

## 5. Ablation λ (d=128, seed 0)

* **Validation** (λ_max = 0 / 0.1 / 0.5 / 1 / 2): Val acc 0.496 / 0.491 / 0.494 / 0.485 / 0.483; Val MSE
  tăng đơn điệu 0.074 → 0.147 (`lambda_ablation.png`, `lambda_ablation_val.csv`).
  **Val không phân biệt rõ các λ; do Val đồng thời được dùng để chọn checkpoint, kết quả này chỉ mang
  tính thăm dò.**
* **Test, phân tích hậu kiểm** (không dùng để chọn λ; λ = 1 cố định từ đặc tả;
  `lambda_ablation_test_posthoc.csv`): λ=0: 0.396, 0.1: 0.417, 0.5: 0.442, 1: 0.427, 2: 0.435;
  AE-MSE: 0.404. **Trên Test, λ=0 thấp hơn rõ so với λ ≥ 0.5, phù hợp với giả thuyết rằng
  classification loss đóng góp vào việc bảo toàn nhiệm vụ.** λ=0 dùng cùng quy tắc chọn checkpoint với
  Task-AE và cho kết quả ngang AE-MSE, nên khác biệt khó giải thích chỉ bằng quy tắc chọn checkpoint.
  Đây là seed 0, 6 fold, chưa kiểm định thống kê.

## 6. Packet / byte stream

* Packet = 4 B header (version u8, method_id u8, config_id u16 LE) + d B INT8. Kiểm tra hai tiến
  trình (node ghi file .bin → gateway đọc file, không có feature gốc): 500/500 packet đúng 132 B,
  prediction trùng 100 % với pipeline tham chiếu (`results/logs/packet_roundtrip_*.json`).
* Mọi kết quả Task-AE/AE-MSE trong bảng đều đi qua đường thật encoder → int8 → bytes → parse → decoder.

## 7. ESP32-S3 (trạng thái)

Đã xong, không cần board:
* Encoder xuất sang C float32 (92.9k / 93.2k / 93.7k tham số = 371.7 / 372.8 / 374.8 KB cho d = 64/128/256).
  Bản C chạy trên host khớp PyTorch trên 500 recordings Test fold 1: Mức 1 (log-mel → INT8) trùng
  **100 %**; Mức 2 (audio → STFT/mel/log/CMVN trên C → INT8) sai số log-mel tối đa 1.4e-5, ≤ 0.006 %
  giá trị INT8 lệch ±1, prediction trùng 100 % (`results/hardware/host_c_consistency.csv`).
* Firmware ESP-IDF v5.5.5 build được cho d = 64/128/256 (c3_node.bin ≈ 1.03 MB, gồm ~400 KB test
  vector nhúng). Đã boot trong **QEMU ESP32-S3** và chạy toàn bộ giao thức gateway: self-check 10/10,
  60 packet, latent + prediction trùng 100 % (`results/hardware/*_qemu.*` — **latency QEMU không phải
  số đo phần cứng**).

**Đo trên board thật** (ESP32-S3 N16R8, rev v0.2, 240 MHz, 16 MB flash, 8 MB octal PSRAM, USB-Serial-JTAG;
`results/hardware/hardware_benchmark.csv`, `hardware_predictions.csv`, `hardware_latency_stats.csv`,
`hardware_consistency.csv`, `serial_logs/`; notebook 08):
* Latency median (1000 lần, 50 warm-up), d = 64 / 128 / 256: encoder 489.0 / 490.8 / 494.7 ms; INT8
  0.03 / 0.06 / 0.11 ms; packet ≤ 0.002 ms; tiền xử lý trên node ≈ 95.5 ms; end-to-end node
  584.4 / 586.5 / 590.5 ms. p95 − median ≤ 0.004 ms.
* Bộ nhớ: trọng số 371.7 / 372.8 / 374.8 KB (Flash), firmware ≈ 1.01 MB, bộ đệm activation 141 KB,
  RAM nội trống 221.6 KB sau khi khởi tạo.
* Round-trip 500 recording Test fold 1 × 2 chế độ × 3 d: prediction trùng Python 100 %; latent INT8
  trùng 99.99–100 % (lệch tối đa ±1); accuracy từ packet ESP32 41.0 / 47.2 / 52.8 % = PyTorch.
* Không kết luận về năng lượng (không đo công suất).

## 8. Giới hạn

* A0 chỉ ≈ 42–44 %; phương sai giữa speaker lớn (std ≈ 13–15 pp) — so sánh dựa vào cặp cùng fold/seed.
* Chỉ 6 speaker → kiểm định ở mức speaker có lực thấp; Task-AE > AE-MSE là xu hướng (5/6 speaker, p ≈ 0.07–0.09), chưa có ý nghĩa thống kê.
* Seed 1–2 chỉ có ở d=128; ablation λ chỉ seed 0.
* Đặc trưng khác đặc tả gốc (log1p) — đã ghi nhận và có bằng chứng.
* Weights/activations trên ESP32 là float32; chỉ latent là INT8.

## 9. Follow-up (post-hoc): classifier V2

Thực hiện sau khi thí nghiệm chính đã đánh giá trên Test → không thay thế kết quả chính.
Chi tiết đầy đủ: `report/BAO_CAO_C3.md` mục 8; dữ liệu: `results/summary/results_summary_v2.csv`,
`classifier_v2_search_val.csv`, `task_vs_mse_d128_speaker_level_v2.csv`; script `run_followup_v2.sh`.

* Cấu hình V2 chọn **chỉ bằng Validation** trong 14 cấu hình (`src/classifier_search.py`, quy tắc: Val
  trung bình 6 fold cao nhất, `results/logs/followup_v2_locked_config.json`): stride lớp 1 = (2,1),
  dropout 0.3, augmentation chỉ trên Train (speed perturbation 0.9–1.1 + nhiễu SNR 20–40 dB, SpecAugment,
  dịch thời gian), 200 epoch, early stopping 40. Val 0.489 → 0.675.
* A0 (Test): V1 44.2 % → V2 **55.3 %** (trung bình 3 seed; seed 0/1/2 = 60.9 / 53.3 / 51.6 %).
* Seed 0, accuracy Task-AE / AE-MSE / DCT: 68 B 53.6 / 47.9 / 40.9 %; 132 B 53.2 / 54.3 / 51.9 %;
  260 B 60.3 / 57.9 / 58.2 %. Low-Mel ≈ 10–14 %.
* d = 128, 3 seed, mức speaker: Task-AE − AE-MSE = +2.37 ± 2.01 pp, 5/6 speaker; Wilcoxon chính xác
  p = 0.0625, paired t p = 0.035. Tương tự thí nghiệm chính (+2.53 pp, 5/6, p = 0.094); riêng seed 0 thì
  Task-AE thấp hơn 1.1 pp.
* Mở rộng (seed 0): d = 512 → AE-MSE 0.609 > DCT 0.597 > Task-AE 0.586 (lợi ích Task-AE mất khi
  ngân sách byte lớn); λ chọn theo Val từng fold → Test 0.541 vs λ = 1 cố định 0.532 (không ổn định);
  lịch λ chậm 30/60/90 → Test 0.572 vs 0.532 (6/6 fold ≥), nhưng Val chọn lịch đề cương và lịch chậm
  train 90 epoch — chỉ là quan sát hậu kiểm. Dữ liệu: `followup_v2_lambda_val_selection.csv`,
  `followup_v2_schedule_ablation.csv`.
