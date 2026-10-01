# 🔬 EDA: MS-COCO vs VizWiz-Captions
### Nghiên cứu Chuyên sâu về Domain Gap trong Trợ lý Thị giác Hỗ trợ Người Khiếm thị
*Đề tài Khóa Luận Tốt Nghiệp — Image Captioning & Multimodal Vision-Language Models*

---

## 📌 Tổng Quan Dự Án

Kho lưu trữ này chứa toàn bộ mã nguồn phân tích khám phá dữ liệu (EDA), biểu đồ trực quan hóa độ phân giải cao, bảng số liệu thống kê chi tiết và file báo cáo Jupyter Notebook đối sánh giữa hai tập dữ liệu thị giác lớn:
- **MS-COCO 2014 (Benchmark Chuẩn mực):** Đại diện cho ảnh chụp đời sống hàng ngày với bố cục chuyên nghiệp, cân đối và chất lượng cao từ Flickr.
- **VizWiz-Captions (Thực tế Hỗ trợ Đời thực):** Tập dữ liệu chụp từ góc nhìn thứ nhất (egocentric) của người khiếm thị qua điện thoại thông minh, phục vụ việc giải quyết các câu hỏi thường nhật.

---

## 🎯 6 Phát Hiện Định Lượng Cốt Lõi

1. **Đảo ngược Tỷ lệ Khung hình (Aspect Ratio Inversion):**
   - COCO chiếm **71.71% ảnh ngang (Landscape)**.
   - VizWiz có tới **92.86% ảnh dọc (Portrait)** do người khiếm thị cầm smartphone thẳng đứng khi chụp.
2. **Độ phân giải & Cảm biến:**
   - COCO chuẩn hóa cố định trung bình $576.5 \times 485.1\text{px}$ (0.28 MP).
   - VizWiz giữ nguyên cảm biến camera di động với trung bình $1067.7 \times 1339.2\text{px}$ (1.57 MP), dao động từ $23\text{px}$ đến $3264\text{px}$.
3. **Chỉ báo chi tiết tần số cao:**
   - Sau EXIF transpose và resize cạnh ngắn 480 px, median Laplacian là **1197.90** (COCO) và **149.19** (VizWiz). Đây không phải bộ phân loại riêng cho motion blur.
4. **Tín hiệu ảnh không thể nhận diện:**
   - **32.62%** ảnh có ít nhất 1/5 annotator dùng caption từ chối; tỷ lệ đồng thuận đa số **≥3/5 là 9.69%**, và 5/5 là **2.48%**.
5. **Nhu cầu Nhận diện Chữ viết (Scene Text / OCR Relevance):**
   - **73.72%** ảnh có ít nhất 1/5 annotator đánh dấu có chữ; tỷ lệ đồng thuận đa số **≥3/5 là 64.75%**.
6. **Lệch pha Ngữ nghĩa (Semantic Shift):**
   - COCO tập trung vào con người (`person` chiếm **30.20%**) và thể thao ngoài trời.
   - Top-15 danh từ được POS-tag trực tiếp từ caption và chuẩn hóa theo 1.000 caption; `person` vẫn nằm trong Top-15 VizWiz và `can` không nằm trong Top-15 sau POS disambiguation.

---

## 📂 Cấu Trúc Thư Mục

```bash
├── EDA_COCO_vs_VizWiz_Report.ipynb    # Báo cáo Jupyter Notebook hoàn chỉnh (nhúng sẵn hình ảnh & bảng biểu)
├── EDA_COCO_vs_VizWiz_Report.md       # Báo cáo Markdown chi tiết phục vụ khóa luận tốt nghiệp
├── eda_coco_vs_vizwiz.py             # Script Python tính toán và kết xuất toàn bộ số liệu & biểu đồ
├── build_eda_notebook.py              # Script tự động đóng gói toàn bộ dữ liệu vào file Notebook
├── eda_results/
│   ├── figures/                       # 7 biểu đồ phân tích độ phân giải cao (300 DPI)
│   │   ├── 01_dataset_splits_comparison.png
│   │   ├── 02_resolution_aspect_ratio.png
│   │   ├── 03_caption_length_and_tokens.png
│   │   ├── 04_coco_category_distribution.png
│   │   ├── 05_vizwiz_quality_and_text_issues.png
│   │   ├── 06_domain_gap_characteristics.png
│   │   └── 07_qualitative_samples_grid.png
│   ├── tables/                        # 6 bảng số liệu CSV
│   │   ├── dataset_overview.csv
│   │   ├── resolution_aspect_ratio.csv
│   │   ├── caption_stats.csv
│   │   ├── coco_supercategories.csv
│   │   ├── coco_categories.csv
│   │   └── vizwiz_quality_breakdown.csv
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
```

### 3. Đóng Gói Lại File Báo Cáo Notebook
```bash
python build_eda_notebook.py
```

---

## 📊 Biểu Đồ Trực Quan Tiêu Biểu

### Biểu đồ Radar Đối sánh 7 Chiều Domain Gap
![Domain Gap Radar](eda_results/figures/06_domain_gap_characteristics.png)

### Lưới Ảnh Đối Sánh Định Tính 8 Trường Hợp Mẫu
![Qualitative Grid](eda_results/figures/07_qualitative_samples_grid.png)

---

## 🎓 Đóng Góp Kỹ Thuật Đề Xuất Cho Mô Hình

1. **Cơ chế Từ chối Thông minh (Smart Refusal Mechanism):** Phân loại ảnh hỏng 2 giai đoạn hoặc token đặc biệt `<QUALITY_REFUSAL>` tránh hiện tượng ảo tưởng (Hallucination) gây nguy hiểm tính mạng khi nhận diện sai thuốc men.
2. **Kiến trúc Tích hợp OCR (OCR-Grounded Vision):** Đưa đặc trưng văn bản đọc được từ bao bì vào làm prefix prompt cho Language Decoder.
3. **Chiến lược Tăng Cường Dữ Liệu (Assistive Data Augmentation):** Mô phỏng rung mờ (motion blur), cắt lệch mép (egocentric crop) và bàn tay che khuất (hand occlusion).
