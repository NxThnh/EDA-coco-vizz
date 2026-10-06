# Các manifest HGAI cố định — bản 9

Hai bộ common-unmatched là nhánh đánh giá/tuning đang dùng. Ba bộ bản 8 được giữ nguyên để đối chiếu; phần val còn lại là kho dự phòng.

| Manifest | Ảnh | Vai trò | Seed |
| :--- | ---: | :--- | ---: |
| dev_common_unmatched500 | 500 | Tuning layer/α, chưa khớp toàn bộ nguồn ID đã kiểm tra | 46 |
| eval_common_unmatched500 | 500 | So sánh chính các mô hình nhỏ, cùng bộ lọc nguồn | 45 |
| eval_random500 | 500 | Đối chiếu riêng giao thức bài báo theo mẫu ngẫu nhiên | 42 |
| eval_unmatched500 | 500 | Lịch sử bản 8: chỉ lọc LLaVA; vẫn khớp ShareGPT4V 433/500 | 43 |
| dev500 | 500 | Dev lịch sử bản 8, không dùng để tuning hiện tại | 44 |

Năm bộ tách ID và MD5. `manifest_lock.json` bản 8 giữ nguyên byte; `manifest_lock_common_v9.json` khóa hai bộ mới, hash nguồn và các cột ID đã kiểm tra. Chọn 1.000 ảnh mới từ 2.830 ảnh đủ điều kiện chưa dùng, trước khi có output mô hình. `lock_common_unmatched.py` kiểm tra khóa khi chạy lại, không tự rút mẫu khác.

Hai bộ mới có 0/500 khớp từng nguồn: LLaVA SFT COCO/VG/GQA; ShareGPT4V SFT, pretraining và cap100K; OneVision ShareGPT4V-COCO upstream. Đây là **chưa khớp nguồn đã kiểm tra**, không bảo đảm chưa xuất hiện trong mọi lịch sử huấn luyện của SmolVLM2, TinyLLaVA hoặc encoder/LLM nền.

JSONL giữ mọi caption chuẩn; CSV cung cấp đường dẫn, ID, MD5 và cờ nguồn. Xem [đối chiếu từng bộ](../../eda_results/hgai/small_vlm_training_overlap_summary.csv) và [thay đổi phân bố object sau lọc](../../eda_results/hgai/sampling_object_distribution.csv). Không chọn mẫu hoặc thông số theo điểm eval/POPE.
