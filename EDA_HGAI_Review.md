# Rà soát tổng quan EDA HGAI — 06/10/2026

EDA đã có dữ liệu, manifest và kiểm tra processor cần thiết cho hướng VLM nhỏ. Adapter, runner, scorer, guard và CI đã được triển khai. Campaign cũ đã dừng trước lưới α=0,25–1,0; phép dò độ dài 10 ảnh đã khóa lưới Smol {0,02; 0,035; 0,05}. Chưa có kết luận hiệu quả HGAI. [Hướng dẫn thực thi](HGAI_Experiments.md), [kết quả dò liều](experiment_outputs/alpha_dose_probe/paired_length_analysis.json).

## Lỗi đã sửa trong lượt rà soát

| Phát hiện | Sửa và kiểm chứng |
| :--- | :--- |
| Hai CSV nguồn của khóa manifest có CRLF tại máy nhưng LF trong commit Git | Lưu nguyên byte hai CSV nguồn trong Git; giữ nguyên cả hai khóa và mọi ID ảnh. Validator so SHA-256 với blob Git, không chỉ với tệp local. |
| 22 liên kết CSV/JSON trong HTML trỏ từ sai thư mục | Bước xuất HTML đổi đường dẫn theo thư mục chứa HTML; validator kiểm tra từng đích. Ảnh vẫn nhúng trực tiếp. |
| Notebook có thể chạy bằng kernel Python toàn cục | Kernel dùng interpreter gọi script; pipeline dùng `.venv-eda` cho mọi bước. |
| Requirements CPU thiếu các gói dùng trực tiếp để kiểm tra/chạy notebook | Bổ sung Torch, NLTK, BeautifulSoup, PyYAML, ipykernel và jupyter-client; ghi Python 3.10–3.12 cho evaluator legacy. |
| Fetch asset có thể lỗi khi chưa có thư mục model_assets/scratch | Tạo thư mục cha trước khi tải asset. |
| Số tham chiếu CHAIR-S 64 token bị ghi thành 20,8 | [Bảng 3 HGAI v3](https://arxiv.org/html/2411.16724v3#S4.T3) ghi **23,8**; [bảng 2](https://arxiv.org/html/2411.16724v3#S4.T2) vẫn ghi **53,0** ở 512 token. Bản v1 cũng ghi 23,8. Đã sửa nguồn dựng báo cáo. |
| Kiểm tra cũ chỉ kiểm tra một số công thức điểm và sự tồn tại của ảnh | Bổ sung MD5 thực tế của 2.500 ảnh, kích thước sau EXIF, ID từng nhóm processor và tính lại CHAIR/POPE từ output gốc. |

Số 433/500 vẫn thuộc `eval_unmatched500` cũ; số 0/500 thuộc hai manifest common-unmatched mới. Không thay số của bộ cũ, không rút lại ảnh và không dùng output mô hình để chọn ảnh.

## Kết quả đã xác minh

- Năm manifest, mỗi manifest 500 ảnh, tách biệt theo ID và MD5; SHA-256 của khóa không đổi.
- Hai manifest mới không khớp năm cờ nguồn đã kiểm tra. Đây không phải bảo đảm không từng xuất hiện trong mọi pretraining.
- 7.500 dòng processor đúng ID ảnh và kích thước; Smol split_off 81 token, split_on theo tile; TinyLLaVA 729 patch token.
- 96.092 dòng token caption cho hai tokenizer; TinyLLaVA processor đối chiếu 20 tensor có sai lệch 0.
- CHAIR/POPE pilot được tính lại từ caption/câu trả lời đã lưu; không sinh lại caption để chọn kết quả tốt hơn.
- Notebook thực thi lại; HTML giữ năm hình nhúng và các liên kết tài liệu có đích trong workspace. Số kiểm tra cuối nằm trong [validation.json](eda_results/hgai/validation.json).

## Phần còn thiếu trước khi kết luận hiệu quả HGAI

| Hạng mục | Trạng thái và việc cần hoàn thành |
| :--- | :--- |
| Adapter HGAI trên SmolVLM2/Phi-2 | Đã kiểm tra công thức, GQA, mask/cache, α=0 bit-exact trên kiến trúc CPU; Smol thật α=0 khớp baseline, α>0 thực thi được. Tiny/LLaVA cần forward với trọng số đang tải. |
| Tuning trên dev | Lưới cũ đã khóa chạy. Campaign v2 đòi kết quả dò liều và bằng chứng VAR/độ nhạy che ảnh theo layer trước khi tạo khóa thiết kế; sau đó mới chọn trên dev500 và lưu trước eval. |
| Baseline và HGAI đủ 500 ảnh | Smol full500 đang chạy. Tiny port khớp 877 tên tensor, runner/scorer chung đã nối cả ba mô hình. Chưa có điểm đầy đủ ba mô hình. |
| Control độ dài/prompt | Runner hỗ trợ cả ba điều kiện; campaign có detailed64/concise64 cho Smol/Tiny, split_off cho Smol và pad/crop cho LLaVA. Chờ thực thi hoàn tất. |
| POPE đầy đủ | Mã chạy 3.000 câu mỗi biến thể cho baseline/HGAI đã có; hiện chỉ pilot 60 câu/biến thể hoàn tất. CI giữ các câu hỏi cùng ảnh trong một cluster. |
| Guard và độ bất định | Recall/F1, độ dài, truncation, latency/memory, refusal/output trống/repetition/Distinct-1 và bootstrap ghép cặp đã triển khai, kiểm tra CPU và smoke thật. α=0,5 làm caption nghèo object trên hai ảnh; chưa phải cải thiện HGAI. |
| Tái lập hàng LLaVA gốc | Phạm vi còn baseline và HGAI 5–18/α=0,5 trên random500. Native/HF equivalence, native FP16 và manifest gốc vẫn chưa xác minh. Cần đối chiếu FP16–NF4 trên cùng ảnh ở GPU đủ bộ nhớ. |
| Manifest huấn luyện SmolVLM2 đầy đủ | Chưa xác minh được subset thực tế gắn checkpoint, các nguồn còn lại và lịch sử mô hình nền. Recipe/upstream không đủ để bảo đảm sạch. |
| Tài liệu định hướng gốc | Người dùng xác nhận chưa có; COCO chính, VizWiz bối cảnh và TinyLLaVA-3.1B tạm chọn giữ theo nội dung đã thống nhất. Không dựng bản thay thế và gọi là gốc. |

Mẫu lọc có thay đổi phân bố: person giảm 3,8 điểm phần trăm và car giảm 2,6 điểm phần trăm so với random500; carrot tăng 2,8 điểm phần trăm. Đây là chênh lệch quan sát trong mẫu, chưa phải kết luận ý nghĩa thống kê. Giữ hai nhánh và báo điểm riêng.

Trọng số, ảnh và annotation lớn không nằm trong Git. Bản clone cần được cung cấp dữ liệu gốc/asset đã pin trước khi chạy toàn bộ validator; báo cáo HTML vẫn hiển thị hình khi mở độc lập, còn liên kết tới CSV/JSON cần cấu trúc thư mục project.
