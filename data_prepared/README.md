# Dữ liệu sau EDA

Tạo lại bằng lệnh `python prepare_data.py --verify-images` từ thư mục gốc. Script chỉ đọc ảnh và annotation; không sửa ảnh gốc.

**Trạng thái nhãn object VizWiz (kiểm tra ngày 04/10/2026): 0/500 ảnh đã gán nhãn, 1.000/1.000 dòng A/B có `review_status=unlabeled`.** File CSV bên dưới là phiếu trống để người gán nhãn điền, chưa phải ground truth và chưa thể dùng để tính CHAIR/POPE trên VizWiz. Không có nhãn object VizWiz trong dữ liệu nguồn hiện có.

## File đầu ra

- `train.jsonl`: metadata và caption từ annotation train. **Chưa dùng để huấn luyện được** vì máy hiện chưa có file ảnh train.
- `dev.jsonl`: các ảnh val nằm ngoài bộ đánh giá cố định **và không trùng byte với ảnh eval**. Dùng để kiểm tra pipeline, chọn prompt và tinh chỉnh cấu hình.
- `eval.jsonl`: đúng 500 ảnh COCO và 500 ảnh VizWiz theo `eda_results/vlm_extensions/evaluation_manifest_seed42.csv`. Giữ tập này cho báo cáo cuối.
- `summary.json`: số lượng, tình trạng file ảnh và kết quả giải mã pixel + áp dụng EXIF transpose.
- `vizwiz_object_annotation_worklist.csv`: **phiếu gán nhãn trống** cho hai người trên 500 ảnh VizWiz eval. `A` và `B` là chỗ dành cho hai người gán nhãn, không có nghĩa hai lượt gán nhãn đã được thực hiện.
- `exact_duplicate_dev_exclusions.csv`: 9 ảnh VizWiz val bị loại khỏi dev vì trùng byte với 8 ảnh eval (49 nhóm trùng byte toàn val). Eval giữ nguyên 500 ảnh và không có hai ảnh trùng byte trong nội bộ manifest.

Mỗi dòng JSONL là một ảnh. `captions` giữ caption gốc đã chuẩn hóa khoảng trắng, ID và các cờ. `valid_captions` chỉ gồm caption có nội dung, không có cờ `is_precanned` hoặc `is_rejected`. `majority_refusal_proxy` là tín hiệu từ annotation, **không phải** nhãn chắc chắn ảnh không thể nhận diện. `image_path` là đường dẫn tương đối từ thư mục gốc; áp dụng EXIF transpose khi mở ảnh bằng Pillow trước khi resize/crop.

## Quy tắc sử dụng

1. Không dùng `dev` hoặc `eval` để train/fine-tune. Chỉ dùng `dev` để chỉnh pipeline/prompt; chạy `eval` sau khi đã khóa cấu hình.
2. Khi bổ sung ảnh train vào `coco/train2014` và `vizwiz/train`, chạy lại script và kiểm tra `summary.json` cho `missing_images=0` ở hai tập train.
3. Ảnh VizWiz không có caption hợp lệ vẫn được giữ trong manifest cho bài toán từ chối/ảnh khó; bỏ chúng ra khỏi loss captioning nếu mô hình yêu cầu caption đích.
4. Không suy ra object ground truth từ caption. CHAIR/POPE trên VizWiz cần nhãn object được con người kiểm tra.
5. Dùng cùng manifest, prompt, processor và cấu hình decoding khi so sánh mô hình hay ablation. Lưu output theo `dataset`, `image_id`, model/checkpoint, prompt, preprocessing và seed.

## Gán nhãn object VizWiz

Phiếu gán nhãn có hai dòng cho mỗi ảnh (`annotator_id=A/B`). Hai người xem ảnh **độc lập**, chưa xem caption tham chiếu hay output mô hình khi gán nhãn. Điền:

- `objects_visible`: danh sách tên object nhìn thấy, ngăn cách bằng dấu `|`, dùng một tên nhất quán cho cùng loại object.
- `objects_uncertain`: object nghi ngờ nhưng không đủ bằng chứng hình ảnh.
- `readable_scene_text`: chữ thực sự đọc được trong ảnh, nếu có.
- `image_unanswerable`: `yes`/`no`/`uncertain` dựa trên việc có thể mô tả nội dung đáng tin cậy.
- `review_status`: `done` khi người gán nhãn hoàn tất. Sau đó người phân xử đối chiếu A/B và lưu nhãn đồng thuận riêng; không tự động xem một trong hai dòng là ground truth.

Chỉ tính chỉ số object trên các ảnh đã phân xử và nêu rõ số ảnh đủ nhãn. Cần lập bảng ánh xạ synonym sang COCO-80 nếu dùng CHAIR/POPE; những object ngoài COCO-80 cần được ghi riêng, không xem là object không tồn tại.
