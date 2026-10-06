# Thực thi HGAI cho ba VLM

Các mô-đun mới triển khai adapter, runner, scorer, guard, CI và chiến dịch tuning/eval. Kết quả đang chạy được ghi riêng khỏi các kết quả đã hoàn tất; không xem kế hoạch hoặc output còn thiếu dòng là điểm benchmark.

## Adapter và kiểm chứng

`hgai_attention.py` giữ nguyên projection, RoPE, GQA và KV-cache của Transformers 4.53.3. Nó chỉ đổi logits của query cuối tại vị trí ảnh trước softmax, bằng α nhân trung bình liên head của trị tuyệt đối logits. Các query khác dùng SDPA; α=0 gọi SDPA trực tiếp. Vị trí ảnh được cấp lại từng input, không lấy một đoạn cố định 576 token cho mọi VLM.

`test_hgai_attention.py` kiểm tra công thức gốc, GQA, mask, cache, α=0 bit-exact và khôi phục backend. Kết quả ở [hgai_adapter_validation.json](eda_results/hgai/hgai_adapter_validation.json). Đây là kiểm tra kiến trúc nhỏ ngẫu nhiên; kiểm chứng bằng trọng số thật được ghi ở thư mục `experiment_outputs/adapter_smoke/` và phải hoàn tất riêng.

TinyLLaVA dùng đúng checkpoint `bczhou/TinyLLaVA-3.1B` đã pin. Mã legacy bỏ layer thứ 27 và pooling head của SigLIP, rồi lấy hidden state cuối trước post-layernorm. Port giữ đúng 26 layer thực tế và toàn bộ tên tensor checkpoint, không thêm layer có trọng số ngẫu nhiên hoặc dùng encoder generic thay encoder đã fine-tune.

## Chạy và chấm điểm

```powershell
.venv-eda/Scripts/python.exe fetch_vlm_weights.py --model all
.venv-eda/Scripts/python.exe test_hgai_attention.py
.venv-eda/Scripts/python.exe vlm_experiment.py --model tiny --partition dev_common_unmatched500 --limit 2
.venv-eda/Scripts/python.exe vlm_experiment.py --model smol --alpha 0.5 --layers 5 18 --partition dev_common_unmatched500 --limit 2
.venv-eda/Scripts/python.exe score_vlm_runs.py PATH_TO_COMPLETE_OUTPUT.jsonl
.venv-eda/Scripts/python.exe score_vlm_runs.py --compare BASELINE.jsonl HGAI.jsonl --ci-output PAIRED_CI.json
```

Runner chung hỗ trợ `--model smol|tiny|llava`, ba điều kiện `detailed512|detailed64|concise64`, CHAIR/POPE, NF4/FP16, layer inclusive và α. Split_on/off dành cho Smol; pad/crop là đối chứng LLaVA. Tiny luôn pad384 theo checkpoint legacy, không tự đổi crop. Output có schema chung để LLaVA và TinyLLaVA dùng cùng scorer. `llava_pilot.py` cũ được giữ như runner lịch sử; sử dụng runner chung cho chiến dịch mới.

Mỗi run khóa manifest, trọng số, processor/tokenizer, mã chạy, annotation và decoding. `--resume` chỉ tiếp tục khi cấu hình/hash trùng; không mở rộng pilot thành full bằng cách đổi limit. Scorer từ chối run chưa đủ số dòng. Bootstrap CHAIR dùng ghép cặp theo ảnh, CHAIR-I là tỷ số tổng numerator/denominator mỗi mẫu bootstrap, không trung bình các tỷ lệ theo ảnh. Bootstrap POPE resample ảnh rồi giữ toàn bộ câu hỏi positive/negative của ảnh đó cùng nhau. Replica có mẫu số 0 được báo không xác định. [Kiểm tra scorer](eda_results/hgai/hgai_scoring_validation.json).

Guard refusal là regex tiếng Anh được công bố trong scorer, không phải nhãn từ chối chuẩn. Distinct-1 và tỷ lệ lặp 4-gram được tính từng câu trả lời rồi lấy trung bình. CHAIR đi kèm recall/F1, độ dài, truncation, thời gian và bộ nhớ; POPE giữ parser chính thức.

`hgai_modified_calls` trong mỗi dòng là bộ đếm tích lũy của run, không cộng các dòng để ra tổng. Peak memory hiện đo GPU, chưa bao gồm RAM dùng cho CPU offload. Hai smoke Smol được giữ cùng bản runner trước sửa metadata Tiny tại `eda_resources/hgai_implementation_snapshots/`; không đổi caption cũ khi cập nhật mã.

## Chiến dịch đầy đủ

```powershell
.venv-eda/Scripts/python.exe run_hgai_suite.py --models smol tiny llava --external-smol-baseline experiment_outputs/full_baseline/smol_chair_eval500_nf4_detailed512.jsonl
.venv-eda/Scripts/python.exe build_experiment_status.py
```

Campaign chạy tuần tự trên một GPU: dev500 baseline và 12 cấu hình can thiệp/mô hình; chọn cấu hình bằng CHAIR-S, CHAIR-I và guard đã khai báo; khóa selection; sau đó eval500 baseline–HGAI, CI và 3×3.000 câu POPE cho mỗi điều kiện baseline/HGAI. Smol/Tiny có thêm hai control prompt, Smol có split_off; LLaVA có pad/crop và random500 đối chiếu giao thức gốc. Baseline α=0 là ứng viên hợp lệ nếu mọi can thiệp vi phạm guard hoặc không tốt hơn.

Guard chọn trên dev: recall và F1 không giảm quá 0,02; refusal, output trống, truncation và lặp 4-gram không tăng quá 0,05; độ dài trung bình giữ ít nhất 50% baseline. Đây là quy tắc khám phá được chốt trước lượt tuning dev500 đầy đủ, không phải ngưỡng do tác giả HGAI xác lập. Không chọn α bằng eval/POPE.

Trạng thái động nằm ở `experiment_outputs/hgai_campaign/campaign_status.json`; mỗi caption được flush và có log tiến độ. Campaign dừng khi lỗi, không ghi trạng thái complete nếu bất kỳ bước nào chưa đạt. Lặp lại cùng lệnh để tiếp tục run với cấu hình đã khóa. Máy GPU 4 GB phải dùng NF4, và LLaVA có thể phải offload CPU; toàn grid và benchmark có thể kéo dài nhiều ngày. NF4/HF conversion không được trình bày như tái lập native FP16 của bài báo.

## Nguồn chưa thể bổ sung

`show_hgai_progress.py` đọc tiến độ local mà không khởi chạy thêm GPU. `monitor_hgai_report.py` chờ campaign kết thúc hoặc báo lỗi rồi tự dựng lại báo cáo; nó không commit/push output đang ghi dở. Workflow `.github/workflows/hgai-cpu.yml` chạy kiểm tra công thức/CI/guard trên CPU khi push hoặc mở PR, không tuyên bố đã kiểm tra benchmark GPU.

Người dùng đã xác nhận chưa có hai tài liệu định hướng gốc. Phạm vi giữ theo nội dung đã thống nhất: COCO chính, VizWiz bối cảnh, TinyLLaVA-3.1B tạm chọn; không dựng hai tài liệu thay thế rồi gọi là bản gốc.

Recipe SmolVLM2 đã kiểm tra gồm 110 nguồn và model card công bố 3,3 triệu mẫu. Nguồn công khai hiện kiểm tra chưa có manifest mẫu đầy đủ gắn checkpoint; YAML trỏ `/fsx` và còn lịch sử mô hình nền. Do đó 0/500 vẫn chỉ là chưa khớp nguồn đã kiểm tra. Không thể sửa giới hạn dữ liệu này bằng đổi nhãn hoặc số trong báo cáo. [Model card chính thức](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct#training-data), [bằng chứng recipe](eda_results/hgai/smol_training_evidence.json).
