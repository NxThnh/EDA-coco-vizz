# Nhật ký thực thi HGAI

## 6. Trạng thái và tái lập

Đã có năm manifest cố định, **7.500 dòng processor**, token caption của cả hai mô hình, hình học toàn val/COCO bbox, đối chiếu nguồn và rà EXIF. Trọng số **SmolVLM2 chính thức đã được tải và xác minh SHA-256**, tổng 8.987.224.056 byte; [provenance trọng số](eda_results/hgai/smol_weights_provenance.json). Không còn dùng “thiếu trọng số” làm lý do cho SmolVLM2.

Máy hiện tại có RTX 3050 Laptop 4 GB; runner `smol_baseline.py` dùng NF4, greedy, template của processor chính thức và bật chia ô. Caption dùng prompt chi tiết, giới hạn 512 token. CHAIR dùng mã evaluator gốc đã pin, chỉ chuyển cú pháp print Python 2 và IO JSONL/val2014; vẫn giữ tokenizer NLTK, singularize Pattern, synonym và các quy tắc cụm từ. POPE giữ parser yes/no trong evaluator chính thức. Nguồn và revision nằm trong [provenance evaluator](eda_results/hgai/metrics_provenance.json).

**Pilot không thay đánh giá đầy đủ:** caption pilot là prefix 20 ảnh của dev mới, không phải mẫu eval500. POPE pilot giữ prefix câu hỏi theo thứ tự gốc, không phải đủ 3.000 câu mỗi biến thể. Không dùng các điểm pilot để kết luận hiệu quả HGAI hoặc so trực tiếp với hàng LLaVA 53,0; NF4 cũng là một điều kiện precision riêng. Chưa có α được chọn từ dev500 hoàn chỉnh hoặc benchmark/đối chứng đầy đủ ba VLM.

**Trạng thái triển khai:** `vlm_experiment.py` hỗ trợ cả ba mô hình, ba điều kiện prompt, CHAIR/POPE, baseline/HGAI và resume với cấu hình/hash cố định; `score_vlm_runs.py` dùng chung schema, guard từ chối/output trống/lặp, Distinct-1 và CI ghép cặp theo ảnh. CI POPE gom toàn bộ câu hỏi của cùng ảnh trong một mẫu bootstrap. TinyLLaVA giữ 26 layer SigLIP thực tế theo mã legacy; cả 877 tên tensor khớp index checkpoint. Trọng số Tiny/LLaVA đang tải; chưa khẳng định forward thực tế hoặc equivalence native/HF của chúng đã đạt. [Kiểm chứng adapter](eda_results/hgai/hgai_adapter_validation.json), [kiểm chứng scorer](eda_results/hgai/hgai_scoring_validation.json), [rà soát tổng quan](EDA_HGAI_Review.md).

**Kiểm chứng Smol thật trên hai ảnh dev:** α=0 cho caption và số token khớp baseline native; α=0,5 thực thi can thiệp nhưng độ dài trung bình giảm từ **298 xuống 10,5 token**, object F1 giảm về 0 và CHAIR-I trở thành không xác định vì không có object được nhận diện. Đây là kiểm tra đường chạy, không phải bằng chứng cải thiện HGAI. Guard chọn cấu hình yêu cầu recall/F1 không giảm quá 0,02, độ dài còn ít nhất 50% baseline; refusal/output trống/truncation/lặp không tăng quá 0,05. Các ngưỡng được khóa trước lượt tuning dev500 đầy đủ. [Kiểm chứng trọng số thật](eda_results/hgai/hgai_smol_real_weight_validation.json).

Campaign cũ đã dừng trước lưới α. `run_hgai_suite.py` phiên bản mới đòi khóa thiết kế α theo mô hình và bằng chứng layer; Smol/Tiny dùng dev500 rồi eval và POPE, LLaVA chỉ chạy hàng baseline/HGAI 5–18, α=0,5. Đối chứng Smol có chia ô mặc định, cạnh dài 768 và tắt chia ô. Bảng tiến độ dưới đây là ảnh chụp, không phải trạng thái trực tiếp; output chưa đủ dòng không có điểm. [Thời điểm và trạng thái chiến dịch](eda_results/hgai/experiment_run_status.json).



### Pilot SmolVLM2 NF4 — kiểm tra pipeline, không phải điểm benchmark đầy đủ

<!-- CSV {"path": "eda_results/hgai/baseline_pilot_summary.csv", "title": "Pilot SmolVLM2 NF4 — kiểm tra pipeline, không phải điểm benchmark đầy đủ", "columns": null} -->
| task | Bộ ảnh | model_precision | role | Số ảnh | questions | CHAIR_S_pct | CHAIR_I_pct | accuracy_pct | precision_pct | recall_pct | F1_pct | yes_rate_pct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| chair | dev_common_unmatched500 | nf4 | feasibility_pilot | 20 | — | 35 | 9,294 | — | — | — | — | — |
| pope | official-pope-adversarial | nf4 | feasibility_pilot | 10 | 60 | — | — | 70 | 63,636 | 93,333 | 75,676 | 73,333 |
| pope | official-pope-popular | nf4 | feasibility_pilot | 10 | 60 | — | — | 76,667 | 70 | 93,333 | 80 | 66,667 |
| pope | official-pope-random | nf4 | feasibility_pilot | 10 | 60 | — | — | 90 | 87,500 | 93,333 | 90,323 | 53,333 |
<!-- /CSV -->


### Tiến độ thực tế — tách pilot khỏi benchmark đầy đủ

<!-- CSV {"path": "eda_results/hgai/experiment_run_status.csv", "title": "Tiến độ thực tế — tách pilot khỏi benchmark đầy đủ", "columns": ["model_key", "task", "partition", "alpha", "precision", "generated_rows", "requested_rows", "status", "role", "CHAIR_S_pct", "CHAIR_I_pct", "accuracy"]} -->
| model_key | task | Bộ ảnh | alpha | precision | generated_rows | requested_rows | Giới hạn kết luận | role | CHAIR_S_pct | CHAIR_I_pct | accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| smol | chair | dev_common_unmatched500 | 0,500 | nf4 | 2 | 2 | scored | feasibility_pilot | 0 | — | — |
| smol | chair | dev_common_unmatched500 | 0 | nf4 | 2 | 2 | scored | feasibility_pilot | 0 | 0 | — |
| smol | chair | eval_common_unmatched500 | 0 | nf4 | 65 | 500 | partial | baseline | — | — | — |
| smol | chair | dev_common_unmatched500 | 0 | nf4 | 20 | 20 | scored | feasibility_pilot | 35 | 9,294 | — |
| smol | pope | official-pope-adversarial | 0 | nf4 | 60 | 60 | scored | feasibility_pilot | — | — | 0,700 |
| smol | pope | official-pope-popular | 0 | nf4 | 60 | 60 | scored | feasibility_pilot | — | — | 0,767 |
| smol | pope | official-pope-random | 0 | nf4 | 60 | 60 | scored | feasibility_pilot | — | — | 0,900 |
| smol | chair | dev_common_unmatched500 | 0 | nf4 | 1 | 1 | scored | feasibility_pilot | 0 | 0 | — |
<!-- /CSV -->


Chạy phần CPU mới bằng môi trường riêng; không thay môi trường cũ dùng CLIP:

```powershell
# Dùng Python 3.10–3.12; CHAIR legacy cần lib2to3.
python -m venv --system-site-packages .venv-eda
.venv-eda/Scripts/python.exe -m pip install -r requirements-hgai-cpu.txt
python fetch_hgai_assets.py
python fetch_training_metadata.py
python fetch_onevision_ids.py
.venv-eda/Scripts/python.exe hgai_cpu_audit.py --prepare-only
.venv-eda/Scripts/python.exe small_vlm_training_overlap.py
.venv-eda/Scripts/python.exe lock_common_unmatched.py
.venv-eda/Scripts/python.exe small_vlm_training_overlap.py
.venv-eda/Scripts/python.exe hgai_cpu_audit.py --missing-only
.venv-eda/Scripts/python.exe verify_tiny_processor.py
.venv-eda/Scripts/python.exe sampling_fix_diagnostics.py
python caption_sampling_sensitivity.py
python vizwiz_rotation_review.py
.venv-eda/Scripts/python.exe rebuild_hgai.py
```

Các fetch EDA chỉ tải asset CPU/annotation/ID; `fetch_smol_baseline.py` là bước riêng tải trọng số chính thức. Trọng số, tokenizer, annotation lớn và môi trường Python bị `.gitignore` bỏ qua. Năm manifest cùng khóa, CSV/hình và output pilot được lưu trong workspace.

Chạy lại pilot hoặc mở rộng baseline bằng runner có kiểm tra khóa và từ chối ghi đè:

```powershell
.venv-eda/Scripts/python.exe -m pip install -r requirements-hgai-inference.txt
.venv-eda/Scripts/python.exe fetch_smol_baseline.py
.venv-eda/Scripts/python.exe fetch_hgai_metrics.py
.venv-eda/Scripts/python.exe smol_baseline.py --partition dev_common_unmatched500 --limit 20
.venv-eda/Scripts/python.exe score_hgai_baseline.py experiment_outputs/smol_baseline/chair_dev_common_unmatched500_nf4_split_on_n20.jsonl
```

Để chạy đủ 500 ảnh, dùng `--partition eval_common_unmatched500 --limit 500`; đây là baseline NF4 trên nhánh lọc, không phải tái lập FP16 của bài báo. POPE dùng `--task pope --pope-variant random --limit 3000`, lặp lại cho popular/adversarial. Chấm từng JSONL bằng `score_hgai_baseline.py`; `--resume` chỉ tiếp tục khi config/hash trùng, không mở rộng limit trong một file pilot đã khóa. [Mã runner](smol_baseline.py), [mã chấm điểm](score_hgai_baseline.py).
