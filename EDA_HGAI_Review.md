# Rà soát tổng quan EDA HGAI — 06/10/2026

EDA đã có dữ liệu, manifest và kiểm tra processor cần thiết cho hướng VLM nhỏ. Thí nghiệm chứng minh hiệu quả HGAI chưa hoàn tất. Rà soát này phân biệt kết quả đã đo, phần triển khai còn thiếu và nguồn chưa thể xác minh.

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
| Adapter HGAI trên SmolVLM2/Phi-2 | Chưa triển khai. Cần giữ công thức logit trước softmax, vị trí token ảnh thật, GQA/KV-cache/mask; α=0 phải khớp baseline. |
| Tuning trên dev | Mới có grid đề xuất. Chưa đo attention/ablation hoặc chọn α/dải layer; phải khóa lựa chọn trước eval. |
| Baseline và HGAI đủ 500 ảnh | Smol chỉ có pilot dev20 NF4. TinyLLaVA chưa có runner sinh caption; LLaVA chưa có output và chưa nối schema runner vào scorer. |
| Control độ dài/prompt | LLaVA có lựa chọn ba điều kiện; runner Smol hiện chỉ có detailed512 và POPE. detailed64/concise64 của Smol còn là thiết kế. |
| POPE đầy đủ | Mỗi biến thể hiện chỉ có 60/3.000 câu; là prefix kiểm tra pipeline, không phải ước lượng benchmark đại diện. |
| Guard và độ bất định | Recall/F1, độ dài, truncation, latency, memory đã có trong điểm CHAIR pilot. Refusal/repetition/Distinct-1, CI bootstrap ghép cặp và tiêu chí chọn cấu hình cụ thể chưa hoàn tất. |
| Tái lập hàng LLaVA gốc | Cần trọng số, kiểm tra native/HF equivalence và baseline→HGAI. Chưa có manifest gốc của tác giả; random500 hiện chỉ khớp kiểu giao thức. |
| Manifest huấn luyện SmolVLM2 đầy đủ | Chưa xác minh được subset thực tế gắn checkpoint, các nguồn còn lại và lịch sử mô hình nền. Recipe/upstream không đủ để bảo đảm sạch. |
| Tài liệu định hướng gốc | Chưa có hai tài liệu gốc; phạm vi VizWiz bối cảnh và checkpoint TinyLLaVA tạm chọn vẫn dựa vào nội dung người dùng cung cấp. |

Mẫu lọc có thay đổi phân bố: person giảm 3,8 điểm phần trăm và car giảm 2,6 điểm phần trăm so với random500; carrot tăng 2,8 điểm phần trăm. Đây là chênh lệch quan sát trong mẫu, chưa phải kết luận ý nghĩa thống kê. Giữ hai nhánh và báo điểm riêng.

Trọng số, ảnh và annotation lớn không nằm trong Git. Bản clone cần được cung cấp dữ liệu gốc/asset đã pin trước khi chạy toàn bộ validator; báo cáo HTML vẫn hiển thị hình khi mở độc lập, còn liên kết tới CSV/JSON cần cấu trúc thư mục project.
