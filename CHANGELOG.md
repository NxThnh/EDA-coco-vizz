# Changelog

## 2026-10-07 — Hiệu đính báo cáo HGAI theo yêu cầu

- Theo ủy quyền tiếp theo của người dùng, chốt quy tắc layer SHA-256 `7faf7ebde361aed139c4d592537f8f2814e6345bee9ae6c62bd67702ec382fdc` và đo VAR/độ nhạy che ảnh trên 20 ảnh dev; quy tắc trả “không tìm thấy ranh giới”. Không dùng điểm CHAIR để chọn ranh giới.
- Chọn checkpoint `bczhou/TinyLLaVA-3.1B` ở revision cố định, tải và đối chiếu SHA-256 đủ ba shard. Chạy baseline và α=0,02 trên một ảnh dev để kiểm tra đường nạp và can thiệp attention; lượt can thiệp có 896 lần gọi được sửa.
- Phép dò α trên 20 ảnh × ba chế độ processor × năm liều được chạy trên GPU với caption tối đa 512 token; đủ 360 caption, bảng ghép cặp và bảng tổng hợp 15 dòng. Cả năm liều được giữ trong khóa lưới trước guard 500 ảnh dev.
- Dựng Markdown và notebook từ cùng bảng Markdown; lọc ba manifest đang dùng và chuẩn hóa cột, số theo tiếng Việt.
- Chuyển nội dung siêu danh mục, nhóm từ chỉ người và hình lệch tâm vào phụ lục; lược các ghi chú phiên bản và giọng nhật ký khỏi báo cáo.
- Đo phân bố COCO-80 trên 2.977 ảnh chưa khớp và toàn bộ val2014, giao với annotation val2017, độ phủ crop theo cặp ảnh–lớp, cùng token ảnh SmolVLM2 ở ba chế độ processor.
- Khóa quy tắc chọn dải layer trước phép đo đường cong và thêm ngân sách tính toán từ nhật ký pilot. Ba manifest và các tệp khóa manifest được giữ nguyên.
- Đối chiếu hai checkpoint TinyLLaVA; checkpoint bczhou được chọn và kiểm chứng trên một ảnh dev. Chưa chạy benchmark TinyLLaVA đầy đủ.
- Dọn sau build: dựng bảng hiển thị trực tiếp từ CSV/JSON nguồn, xóa 10 CSV trung gian, một JSON tổng hợp trùng lặp và CSV phân bố token có thể tính từ dữ liệu từng ảnh; bỏ hai script tải/ghi khóa dùng một lần.
