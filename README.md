# EDA cho HGAI trên VLM nhỏ — bản 9

Đánh giá chính trên **COCO val2014**, với **SmolVLM2-2.2B** và **TinyLLaVA-3.1B (SigLIP 384px)**. LLaVA-1.5-7B là mốc kiểm chứng; CHAIR là thước đo chính, POPE kiểm tra chéo. VizWiz chỉ giữ làm bối cảnh/phụ lục theo hướng đồ án người dùng cung cấp.

Đọc [báo cáo chính](EDA_COCO_vs_VizWiz_Report.md), [notebook có output thật](EDA_COCO_vs_VizWiz_Report.ipynb) hoặc [bản HTML có ảnh nhúng, mở độc lập](eda_results/hgai/EDA_HGAI_Small_VLM_Report.html). Tên file báo cáo được giữ để không làm hỏng liên kết cũ. Phân tích COCO–VizWiz chi tiết trước đây được giữ ở [tài liệu bối cảnh](EDA_COCO_vs_VizWiz_Context.md).

## Kết quả CPU đã tính

- Processor chạy trên **2.500 ảnh cố định**, gồm cả 1.000 ảnh common-unmatched mới; tổng **7.500 dòng** cho SmolVLM2 chia ô bật/tắt và TinyLLaVA pad384.
- SmolVLM2: 640×480 cho **1.053 visual token**, ảnh 640×640 cho **1.377**, tắt chia ô cho **81**. Số token được đếm từ input processor, không suy từ độ phân giải gốc.
- Tokenizer đúng checkpoint trên 40.504 caption COCO và 7.542 caption VizWiz ở cả hai mô hình; không cộng token ảnh/prompt.
- Pad/crop/scale TinyLLaVA và LLaVA được tính trên toàn val; hình học SmolVLM2 được kiểm tra trên cả năm manifest, tách độ méo tile/global. Processor TinyLLaVA khớp tensor lớp legacy trên 20 ảnh, sai lệch tối đa 0.
- Lệch tâm bbox đã đo trên **40.137 ảnh COCO có instance**; 367 ảnh không có bbox được để trống.
- Rò rỉ LLaVA gồm COCO trực tiếp và alias VG/GQA: **16.618/40.504 (41,028%)**. TinyLLaVA được đối chiếu cả hai recipe; SmolVLM2 được kiểm tra ID nguồn upstream OneVision và ghi rõ giới hạn.
- Rà trực quan 40 cặp ảnh VizWiz raw/sau EXIF; cập nhật nhóm từ chỉ người về cùng đồng nghĩa CHAIR (37,52% COCO và 8,66% VizWiz).

**Đã sửa lỗi chọn ảnh 433/500:** hai bộ mới mỗi bộ 500 ảnh có **0/500 khớp toàn bộ nguồn ID đã kiểm tra** cho LLaVA, ShareGPT4V SFT/pretraining/cap100K và OneVision ShareGPT4V-COCO upstream. Các bộ cũ giữ nguyên. Chênh lệch hình học/token và phân bố 80 lớp object sau lọc đã được tính riêng.

Trọng số **SmolVLM2 chính thức đã tải và kiểm tra SHA-256**; caption thực tế và pilot CHAIR/POPE nằm trong [`experiment_outputs/smol_baseline/`](experiment_outputs/smol_baseline/). Adapter HGAI đã chạy với trọng số Smol thật: α=0 khớp baseline, nhưng α=0,5 làm mất nội dung object trong smoke hai ảnh. Guard recall/F1/độ dài ngăn chọn cấu hình kiểu này. Pilot NF4 trên GPU 4 GB không thay kết quả đầy đủ hoặc chứng minh hiệu quả HGAI.

Đã kiểm tra thêm 110 mục trong YAML mixture và mã huấn luyện của tác giả SmolVLM2, lưu ở [`eda_resources/smol_training/`](eda_resources/smol_training/). Recipe công bố không cung cấp đủ JSON nguồn nội bộ và danh sách mẫu thực tế gắn với checkpoint, nên chưa thể xác minh toàn bộ lịch sử huấn luyện. “Common-unmatched” chỉ nói về nguồn đã kiểm tra. Người dùng xác nhận chưa có hai tài liệu định hướng gốc; phạm vi dựa trên nội dung đã cung cấp.

## Manifest và giao thức

[`data_prepared/hgai/`](data_prepared/hgai/README.md) có năm bộ **500 ảnh**, tách ID/MD5 và khóa SHA-256. **eval_common_unmatched500** là nhánh chính các VLM nhỏ; **dev_common_unmatched500** dùng tuning. **eval_random500** giữ seed42 để báo riêng khi đối chiếu giao thức bài báo; eval_unmatched500/dev500 chỉ là manifest lịch sử bản 8. Không gộp điểm mẫu lọc và mẫu ngẫu nhiên.

Giao thức chính: prompt mô tả chi tiết của HGAI, **512 token**, greedy, 500 ảnh. Có control **cùng prompt 64 token** và **prompt ngắn 64 token**, tránh trộn ảnh hưởng prompt với độ dài. Cấu hình được lưu trong [`experiment_config.json`](eda_results/hgai/experiment_config.json). Mã HGAI chính thức được giữ theo commit ở `eda_resources/hgai_original/`; chưa có manifest gốc tác giả nên mẫu hiện tại khớp giao thức, chưa khớp chính xác mọi ID của bài báo.

`llava_pilot.py` mặc định dùng dev_common_unmatched500/detailed512/crop theo mã đánh giá gốc; pad là control riêng. Đây là runner baseline LLaVA, yêu cầu checkpoint local. `smol_baseline.py` chạy baseline SmolVLM2/POPE trên GPU; cả hai đều không phải adapter HGAI và không ghi đè output cũ.

## Tái lập

Các script EDA cũ vẫn dùng môi trường cũ. Dùng Python 3.10–3.12 (CHAIR legacy cần `lib2to3`) và tạo môi trường CPU riêng cho processor mới. Các phụ thuộc CPU, Jupyter và chấm điểm đã khai báo trong requirements; notebook chạy bằng đúng interpreter của pipeline:

```powershell
python -m venv --system-site-packages .venv-eda
.venv-eda/Scripts/python.exe -m pip install -r requirements-hgai-cpu.txt
python fetch_hgai_assets.py
python fetch_training_metadata.py
python fetch_onevision_ids.py
.venv-eda/Scripts/python.exe hgai_cpu_audit.py --prepare-only
python small_vlm_training_overlap.py
python lock_common_unmatched.py
python small_vlm_training_overlap.py
.venv-eda/Scripts/python.exe hgai_cpu_audit.py --missing-only
.venv-eda/Scripts/python.exe verify_tiny_processor.py
python sampling_fix_diagnostics.py
python caption_sampling_sensitivity.py
python vizwiz_rotation_review.py
.venv-eda/Scripts/python.exe rebuild_hgai.py
```

Các fetch EDA lấy processor/tokenizer, annotation và ID; `fetch_smol_baseline.py` là bước riêng tải trọng số chính thức. Cần mạng khi fetch và dữ liệu COCO/VizWiz đã có. Hash/revision được lưu; manifest không tự đổi khi nguồn thay đổi.

Để chạy pilot GPU và chấm output thật:

```powershell
.venv-eda/Scripts/python.exe -m pip install -r requirements-hgai-inference.txt
.venv-eda/Scripts/python.exe fetch_smol_baseline.py
.venv-eda/Scripts/python.exe fetch_hgai_metrics.py
.venv-eda/Scripts/python.exe smol_baseline.py --partition dev_common_unmatched500 --limit 20
.venv-eda/Scripts/python.exe score_hgai_baseline.py experiment_outputs/smol_baseline/chair_dev_common_unmatched500_nf4_split_on_n20.jsonl
```

`run_smol_pilots.py` chạy/chấm lần lượt pilot caption 20 ảnh dev và 60 câu hỏi cho mỗi biến thể POPE; dùng cùng GPU tuần tự. Mỗi pilot được gắn nhãn riêng, không trình bày như điểm benchmark đầy đủ.

Đủ 500 ảnh: `smol_baseline.py --partition eval_common_unmatched500 --limit 500`. Đủ POPE: `--task pope --pope-variant random --limit 3000`, lặp lại cho popular/adversarial. Chấm từng JSONL bằng `score_hgai_baseline.py`. Chạy lại một output đã có cần `--resume` và đúng config/hash; đổi số ảnh/precision phải tạo output khác. Runner dùng NF4 mặc định; không coi điểm đó là FP16 hoặc hàng tái lập LLaVA của bài báo.

Notebook được dựng từ một nguồn Markdown duy nhất; bảng đọc CSV ngay cạnh phần giải thích và hình nhúng vào attachment. Hình 8/9 xuất hiện một lần. `validate_hgai.py` kiểm tra khóa manifest, split ID/byte, token processor, cờ rò rỉ, độ dài caption, output notebook và ảnh nhúng HTML.

Sau khi đã có đủ output CPU, chạy `python rebuild_hgai.py` để dựng, thực thi, xuất và kiểm tra báo cáo trong một lệnh; pipeline dừng ngay nếu một bước lỗi. Dùng `--refresh-training-overlap` khi metadata nguồn vừa được bổ sung; `--recompute-cpu` chỉ khi cần chạy lại toàn bộ processor.

Xem [hướng dẫn thí nghiệm](HGAI_Experiments.md) và [rà soát tổng quan](EDA_HGAI_Review.md). Runner chung `vlm_experiment.py` hỗ trợ Smol/Tiny/LLaVA, cả ba prompt, baseline/HGAI và POPE; scorer chung có refusal/output trống/repetition/Distinct-1 và CI ghép cặp theo ảnh. Port Tiny đã khớp 877 tên tensor, cần kiểm chứng forward khi tải xong trọng số. Chiến dịch đầy đủ đã khởi chạy, gồm tuning dev500, khóa selection, eval500, đối chứng và POPE 3.000 câu mỗi biến thể/điều kiện. [Bảng tiến độ](eda_results/hgai/experiment_run_status.csv) là snapshot có thời điểm, chưa phải điểm đầy đủ. Số tham chiếu CHAIR-S 64 token đã đối chiếu [bảng 3 HGAI v3](https://arxiv.org/html/2411.16724v3#S4.T3): **23,8**.

```powershell
.venv-eda/Scripts/python.exe fetch_vlm_weights.py --model all
.venv-eda/Scripts/python.exe test_hgai_attention.py
.venv-eda/Scripts/python.exe test_hgai_scoring.py
.venv-eda/Scripts/python.exe run_hgai_suite.py --models smol tiny llava --external-smol-baseline experiment_outputs/full_baseline/smol_chair_eval500_nf4_detailed512.jsonl
.venv-eda/Scripts/python.exe show_hgai_progress.py
```

Máy đang chạy campaign nền tuần tự; không mở thêm runner GPU đồng thời. Campaign và caption dở được giữ local, flush từng ảnh, `--resume` giữ nguyên config/hash. Các kết quả hoàn tất được lưu cùng hash và có thể đưa lên Git riêng. Toàn grid có thể mất nhiều ngày trên GPU 4 GB; NF4 và HF conversion không phải hàng tái lập native FP16 của bài báo.

## Dữ liệu nền được giữ

`prepare_data.py` và `data_prepared/dev.jsonl`/`eval.jsonl` ở thư mục cha phục vụ EDA COCO–VizWiz cũ, không quy định dev/eval HGAI. Các biểu đồ, Laplacian, CLIP, nguồn CHAIR và audit LLaVA/VG/EXIF ở `eda_results/tables/`, `eda_results/figures/`, `eda_results/vlm_extensions/` vẫn được giữ để đối chiếu.

Nhãn object VizWiz chưa hoàn thành; phiếu A/B ở `data_prepared/vizwiz_object_annotation_worklist.csv` vẫn là phiếu trống, không được dùng làm ground truth CHAIR/POPE.
