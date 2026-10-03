# 🔬 EDA: MS-COCO vs VizWiz-Captions
### Nghiên cứu Chuyên sâu về Domain Gap trong Trợ lý Thị giác Hỗ trợ Người Khiếm thị
*Đề tài Khóa Luận Tốt Nghiệp — Image Captioning & Multimodal Vision-Language Models*

---

## 📌 Tổng Quan Dự Án

Kho lưu trữ này chứa toàn bộ mã nguồn phân tích khám phá dữ liệu (EDA), biểu đồ trực quan hóa độ phân giải cao, bảng số liệu thống kê chi tiết và file báo cáo Jupyter Notebook đối sánh giữa hai tập dữ liệu thị giác lớn:
- **MS-COCO val2014:** 40.504 ảnh có caption và instance annotation; dùng làm tập đối chiếu.
- **VizWiz-Captions (Thực tế Hỗ trợ Đời thực):** Tập dữ liệu chụp từ góc nhìn thứ nhất (egocentric) của người khiếm thị qua điện thoại thông minh, phục vụ việc giải quyết các câu hỏi thường nhật.

---

## 🎯 6 Phát Hiện Định Lượng Cốt Lõi

1. **Đảo ngược Tỷ lệ Khung hình (Aspect Ratio Inversion):**
   - COCO chiếm **71.71% ảnh ngang (Landscape)**.
   - VizWiz có **99.23% ảnh dọc** sau EXIF transpose.
2. **Độ phân giải & Cảm biến:**
   - Kích thước file COCO trung bình $576.5 \times 485.1\text{px}$ (0.28 MP).
   - Kích thước file VizWiz sau EXIF trung bình $1032.1 \times 1374.7\text{px}$ (1.57 MP), có ngoại lai rộng 23 px.
3. **Chỉ báo chi tiết tần số cao:**
   - Trên 1.000 ảnh mỗi tập, sau EXIF transpose và resize cạnh ngắn 480 px, median Laplacian là **1197.90** (COCO) và **149.19** (VizWiz). Đây là chỉ báo biên/texture; có phân tích độ nhạy theo ảnh bị phóng lớn.
4. **Tín hiệu ảnh không thể nhận diện:**
   - **32.62%** ảnh có ít nhất 1/5 annotator dùng caption từ chối; tỷ lệ đồng thuận đa số **≥3/5 là 9.69%**, và 5/5 là **2.48%**.
5. **Nhu cầu Nhận diện Chữ viết (Scene Text / OCR Relevance):**
   - **73.72%** ảnh có ít nhất 1/5 annotator đánh dấu có chữ; tỷ lệ đồng thuận đa số **≥3/5 là 64.75%**.
6. **Lệch pha Ngữ nghĩa (Semantic Shift):**
    - Trong annotation instance COCO, `person` chiếm **30.20%** số object; đây là đơn vị khác với tần suất từ trong caption.
    - Trên 1.000 caption hợp lệ, VizWiz nhắc `computer` 95,58 lần và `screen` 92,96 lần; COCO nhắc `man` 122,31 lần và `people` 69,46 lần. Caption VizWiz mang cờ `is_precanned` hoặc `is_rejected` được loại khỏi thống kê ngôn ngữ. Bảng `person_reference_summary.csv` gộp các dạng chỉ người và báo tỷ lệ theo caption/ảnh.

---

## 📂 Cấu Trúc Thư Mục

```bash
├── EDA_COCO_vs_VizWiz_Report.ipynb    # Báo cáo Jupyter Notebook hoàn chỉnh (nhúng sẵn hình ảnh & bảng biểu)
├── EDA_COCO_vs_VizWiz_Report.md       # Báo cáo Markdown chi tiết phục vụ khóa luận tốt nghiệp
├── eda_coco_vs_vizwiz.py             # Script Python tính toán và kết xuất toàn bộ số liệu & biểu đồ
├── build_eda_notebook.py              # Script tự động đóng gói toàn bộ dữ liệu vào file Notebook
├── run_eda_notebook.py                # Chạy lại mọi code cell trong kernel mới và lưu output thật
├── eda_results/
│   ├── figures/                       # 6 biểu đồ phân tích độ phân giải cao (300 DPI)
│   │   ├── 01_dataset_splits_comparison.png
│   │   ├── 02_resolution_aspect_ratio.png
│   │   ├── 03_caption_length_and_tokens.png
│   │   ├── 04_coco_category_distribution.png
│   │   ├── 05_vizwiz_quality_and_text_issues.png
│   │   └── 07_qualitative_samples_grid.png
│   ├── tables/                        # Các bảng số liệu CSV
│   │   ├── dataset_overview.csv
│   │   ├── resolution_aspect_ratio.csv
│   │   ├── caption_stats.csv
│   │   ├── coco_supercategories.csv
│   │   ├── coco_categories.csv
│   │   ├── vizwiz_quality_breakdown.csv
│   │   ├── caption_noun_entities_top15.csv
│   │   ├── laplacian_by_precanned.csv
│   │   ├── laplacian_resize_sensitivity.csv
│   │   ├── person_reference_summary.csv
│   │   └── qualitative_samples_manifest.csv
│   ├── coco500/                      # Bảng seed 42 và độ nhạy qua 200 seed
│   └── eda_summary_metrics.json       # Tổng hợp toàn bộ chỉ số định lượng dưới dạng JSON
└── README.md
```

---

## 🚀 Hướng Dẫn Tái Lập (Reproducibility)

### 1. Cài đặt Môi trường
```bash
pip install -r requirements-eda.txt
```

### 2. Chạy Phân Tích & Sinh Lại Kết Quả
```bash
python eda_coco_vs_vizwiz.py
python rerun_coco500_tables.py
```

### 3. Đóng Gói Lại File Báo Cáo Notebook
```bash
python build_eda_notebook.py
python run_eda_notebook.py
```

---

## 📊 Biểu Đồ Trực Quan Tiêu Biểu

### Phân bố tỷ lệ khung hình và độ phân giải
![Phân bố độ phân giải](eda_results/figures/02_resolution_aspect_ratio.png)

### Lưới Ảnh Đối Sánh Định Tính 8 Trường Hợp Mẫu
![Qualitative Grid](eda_results/figures/07_qualitative_samples_grid.png)

---

## 🎓 Giả thuyết cho thực nghiệm tiếp theo

1. So sánh cơ chế từ chối với mô hình gốc trên từng mức đồng thuận caption từ chối.
2. So sánh bật/tắt OCR trên ảnh có và không có chữ, đo độ chính xác thông tin chữ.
3. So sánh tiền xử lý ảnh và tăng cường dữ liệu trên cùng mô hình, đo nội dung bị cắt và sai số đầu ra.
