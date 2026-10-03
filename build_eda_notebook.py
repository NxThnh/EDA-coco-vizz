#!/usr/bin/env python3
"""
Script to build the comprehensive EDA Report Jupyter Notebook:
EDA_COCO_vs_VizWiz_Report.ipynb
"""

import os
import sys
import json
import base64
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_NB_PATH = os.path.join(BASE_DIR, "EDA_COCO_vs_VizWiz_Report.ipynb")
FIG_DIR = os.path.join(BASE_DIR, "eda_results", "figures")
TAB_DIR = os.path.join(BASE_DIR, "eda_results", "tables")
METRICS_JSON = os.path.join(BASE_DIR, "eda_results", "eda_summary_metrics.json")

# Helper to encode figure to base64
def get_image_b64(fig_name):
    fig_path = os.path.join(FIG_DIR, fig_name)
    if os.path.exists(fig_path):
        with open(fig_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

def create_markdown_cell(content):
    lines = content.strip().split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def create_code_cell(code_str, outputs=None, execution_count=1):
    lines = code_str.strip().split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def make_stream_output(text):
    lines = text.strip().split("\n")
    out_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": out_lines
    }

def make_display_output(html_text=None, plain_text=None, png_b64=None):
    data = {}
    if plain_text:
        data["text/plain"] = [plain_text + "\n"]
    if html_text:
        data["text/html"] = [html_text + "\n"]
    if png_b64:
        data["image/png"] = png_b64
    return {
        "output_type": "display_data",
        "data": data,
        "metadata": {}
    }

print("Loading summary metrics...")
with open(METRICS_JSON, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print("Building cells...")
cells = []
exec_counter = 1

# ==============================================================================
# CELL 1: Cover & Executive Summary (Markdown)
# ==============================================================================
cell1_md = r"""# 🔬 BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
## ĐỐI SÁNH TẬP DỮ LIỆU MS-COCO VÀ VIZWIZ-CAPTIONS
### Chuyên đề Khóa Luận Tốt Nghiệp: Nghiên cứu Domain Gap & Thách thức trong Trợ lý Thị giác Hỗ trợ Người Khiếm thị

---

<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
    <h3 style="margin-top: 0; color: #ffeb3b;">📋 THÔNG TIN BÁO CÁO</h3>
    <ul style="margin-bottom: 0; line-height: 1.8;">
        <li><b>Tập dữ liệu khảo sát:</b> MS-COCO val2014 và VizWiz-Captions val</li>
        <li><b>Phạm vi phân tích:</b> 40.504 ảnh COCO có annotation | 7.750 ảnh VizWiz có ảnh gốc và caption</li>
        <li><b>Mẫu kiểm tra tái lập:</b> 500 ảnh COCO lấy ngẫu nhiên với seed=42 cho Bảng 2 và Bảng 4</li>
        <li><b>Mục tiêu chính:</b> Định lượng hóa khoảng cách miền dữ liệu (Domain Gap) nhằm đề xuất giải pháp kỹ thuật, chiến lược tiền xử lý và kiến trúc mô hình Image Captioning tối ưu cho người khiếm thị.</li>
    </ul>
</div>

---

## 📌 TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

EDA đối chiếu ảnh và annotation của **MS-COCO val2014** với **VizWiz-Captions val**. Sáu phát hiện định lượng về dữ liệu đầu vào là:

1. **🔄 Đảo ngược Tỷ lệ Khung hình (Aspect Ratio Inversion):**
   - COCO có **71.71% ảnh ngang** trong tập khảo sát.
   - VizWiz có **99.23% ảnh dọc** sau khi áp dụng EXIF orientation.
2. **📐 Chênh lệch Cảm biến & Kích thước Hình ảnh:**
   - Kích thước file COCO trung bình là $576.5 \times 485.1\text{px}$ (0.28 MP).
   - Kích thước file VizWiz sau EXIF trung bình là $1032.1 \times 1374.7\text{px}$ (1.57 MP), với chiều rộng nhỏ nhất 23 px và chiều cao lớn nhất 3264 px.
3. **🌫️ Chỉ báo chi tiết tần số cao của ảnh:**
   - Sau EXIF transpose và resize giữ tỷ lệ về cạnh ngắn 480 px, median phương sai Laplacian là **1197.90** ở COCO và **149.19** ở VizWiz. Đây là chỉ báo độ mạnh biên/texture, không tự phân biệt motion blur, defocus và cảnh ít texture.
4. **⚠️ Tín hiệu ảnh không thể nhận diện:**
   - **32.62%** ảnh có ít nhất 1/5 annotator dùng caption từ chối; tỷ lệ đồng thuận đa số **≥3/5 là 9.69%**, và đồng thuận tuyệt đối 5/5 là **2.48%**.
5. **🔤 Nhu cầu Đọc Chữ viết (Scene Text / OCR Relevance):**
   - **73.72%** ảnh có ít nhất 1/5 annotator đánh dấu có chữ; theo đồng thuận đa số **≥3/5 là 64.75%**, và 47.10% đạt 5/5. Dữ liệu cho thấy scene text phổ biến, nhưng không đủ để kết luận mục đích chụp chủ yếu là đọc nhãn thuốc.
6. **👥 Lệch pha Ngữ nghĩa (Semantic Shift):**
    - Trong annotation instance COCO, `person` chiếm **30.20%** số object; tỷ lệ này có đơn vị khác với tần suất từ trong caption.
    - Caption VizWiz thường nhắc `computer`, `screen`, `bottle`, `box` và `label`; caption COCO thường nhắc `man`, `people` và `woman`. Nhóm từ chỉ người được tổng hợp riêng theo caption và ảnh trong `person_reference_summary.csv`."""
cells.append(create_markdown_cell(cell1_md))

# ==============================================================================
# CELL 2: Environment Setup & Helper Functions (Code)
# ==============================================================================
cell2_code = """import os
import json
import base64
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, HTML, Image as IPImage

# Tự tìm thư mục dự án từ thư mục chạy notebook hoặc các thư mục cha.
_roots = [Path.cwd(), *Path.cwd().parents]
BASE_DIR = str(next((p for p in _roots if (p / "eda_coco_vs_vizwiz.py").exists()), Path.cwd()))
COCO_DIR = os.path.join(BASE_DIR, "coco")
VIZWIZ_DIR = os.path.join(BASE_DIR, "vizwiz")
RESULTS_DIR = os.path.join(BASE_DIR, "eda_results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
METRICS_JSON = os.path.join(RESULTS_DIR, "eda_summary_metrics.json")
ANALYSIS_SCRIPT = os.path.join(BASE_DIR, "eda_coco_vs_vizwiz.py")

# Đặt REBUILD=True để tính lại CSV/JSON/hình từ annotation và ảnh gốc.
REBUILD = False
if REBUILD or not os.path.exists(METRICS_JSON):
    subprocess.run([sys.executable, ANALYSIS_SCRIPT], cwd=BASE_DIR, check=True)

# Cấu hình hiển thị pandas
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def render_table(df, title=""):
    \"\"\"Hiển thị bảng dữ liệu với định dạng HTML CSS chuyên nghiệp\"\"\"
    html = f\"\"\"
    <div style="margin: 15px 0;">
        <h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 {title}</h4>
        <div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            {df.to_html(classes='table table-hover table-striped', index=False)}
        </div>
    </div>
    \"\"\"
    display(HTML(html))

def show_figure(filename, title="", caption="", width=950):
    \"\"\"Nhúng ảnh bằng data URI để notebook vẫn chạy khi di chuyển/export.\"\"\"
    filepath = os.path.join(FIG_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            image_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        html_caption = f\"\"\"
        <div style="text-align: center; margin: 15px auto 25px auto; max-width: {width}px;">
            <div style="border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.12); border: 1px solid #e0e0e0;">
                <img src="{image_uri}" style="width: 100%; height: auto; display: block;" />
            </div>
            <p style="font-size: 0.9em; color: #555; margin-top: 8px; font-style: italic;">
                <b>{title}</b>: {caption}
            </p>
        </div>
        \"\"\"
        display(HTML(html_caption))
    else:
        print(f"Warning: Khong tim thay file: {filepath}")

print("[OK] Khoi tao moi truong phan tich va cac ham ho tro thanh cong!")"""
out2 = [make_stream_output("[OK] Khoi tao moi truong phan tich va cac ham ho tro thanh cong!")]
cells.append(create_code_cell(cell2_code, out2, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 3: Section 1 - Analysis Scope (Markdown)
# ==============================================================================
cell3_md = r"""---
## 1. PHẠM VI ĐỐI SÁNH DỮ LIỆU

Phần mở đầu mô tả hai miền dữ liệu theo đúng phạm vi ảnh và chú thích thực tế được khảo sát. Mẫu COCO-500 (`seed=42`) có kích thước trung bình 576,9 × 481,8 px; 73,60% ảnh ngang, 22,80% ảnh dọc và 3,60% ảnh vuông. Mẫu có 3.719 object instance; ba siêu danh mục lớn nhất là `person` (29,52%), `vehicle` (12,34%) và `kitchen` (10,24%). Trong 200 mẫu 500 ảnh với các seed khác nhau, tỷ lệ `person` có trung vị 30,03% và khoảng phân vị 2,5–97,5% là 27,06–33,49%. Đây là độ nhạy theo cách lấy mẫu, không phải phép cân bằng quy mô với VizWiz. Các kết quả lưu ở `eda_results/coco500/`."""
cells.append(create_markdown_cell(cell3_md))

# ==============================================================================
# CELL 4: Section 1 - Code & Display Table & Figure (Code)
# ==============================================================================
cell4_code = """# Phạm vi phân tích theo hai miền dữ liệu
df_scope = pd.DataFrame([
    {"Miền dữ liệu": "MS-COCO val2014", "Phạm vi": "40.504 ảnh", "Vai trò": "Miền ảnh chuẩn/curated"},
    {"Miền dữ liệu": "VizWiz-Captions", "Phạm vi": "7.750 ảnh", "Vai trò": "Miền hỗ trợ thị giác đời thực"},
    {"Miền dữ liệu": "COCO-500", "Phạm vi": "500 ảnh (seed=42)", "Vai trò": "Chạy lại Bảng 2 và Bảng 4"},
])
render_table(df_scope, "Bảng 1: Phạm vi đối sánh dữ liệu")"""

df_scope = pd.DataFrame([
    {"Miền dữ liệu": "MS-COCO val2014", "Phạm vi": "40.504 ảnh", "Vai trò": "Miền ảnh chuẩn/curated"},
    {"Miền dữ liệu": "VizWiz-Captions", "Phạm vi": "7.750 ảnh", "Vai trò": "Miền hỗ trợ thị giác đời thực"},
    {"Miền dữ liệu": "COCO-500", "Phạm vi": "500 ảnh (seed=42)", "Vai trò": "Chạy lại Bảng 2 và Bảng 4"},
])
df_splits_html = df_scope.to_html(classes='table table-hover table-striped', index=False)
out4 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 1: Phạm vi đối sánh dữ liệu</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_splits_html}</div></div>""")
]
cells.append(create_code_cell(cell4_code, out4, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 5: Section 1 - Analytical Commentary (Markdown)
# ==============================================================================
cell5_md = r"""### Quy ước đơn vị phân tích
Thống kê ảnh dùng số ảnh; phân tích ngôn ngữ chỉ dùng caption không mang cờ `is_precanned` hoặc `is_rejected`; phân bố đối tượng dùng số object instance. Có **5.167 caption VizWiz (13,33%)** là placeholder báo lỗi chất lượng và thêm **438 caption chỉ mang cờ `is_rejected`**. Chúng bị loại khỏi thống kê ngôn ngữ/token/thực thể."""
cells.append(create_markdown_cell(cell5_md))

# ==============================================================================
# CELL 6: Section 2 - Resolution & Aspect Ratio Inversion (Markdown)
# ==============================================================================
cell6_md = r"""---
## 2. ĐỘ PHÂN GIẢI & ĐẢO NGƯỢC TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO INVERSION)

Tỷ lệ khung hình (Aspect Ratio) được xác định bởi:
$$\\text{Aspect Ratio (AR)} = \\frac{\\text{Width}}{\\text{Height}}$$

- Ảnh ngang (Landscape): $\\text{AR} > 1.05$
- Ảnh dọc (Portrait): $\\text{AR} < 0.95$
- Ảnh vuông (Square): $0.95 \\le \\text{AR} \\le 1.05$

Kích thước ảnh được lấy từ annotation sau khi áp dụng EXIF orientation cho VizWiz; số đo hiển thị là chiều rộng và chiều cao theo hướng xem ảnh. Khảo sát gồm $40.504$ ảnh MS-COCO val2014 và $7.750$ ảnh VizWiz-Captions val."""
cells.append(create_markdown_cell(cell6_md))

# ==============================================================================
# CELL 7: Section 2 - Code & Display Table & Figure (Code)
# ==============================================================================
cell7_code = """# Đọc bảng phân tích kích thước và tỷ lệ khung hình
df_res = pd.read_csv(os.path.join(TAB_DIR, "resolution_aspect_ratio.csv"))

# Hiển thị bảng
render_table(df_res, "Bảng 2: So sánh Kích thước Hình ảnh và Phân bố Tỷ lệ Khung hình (AR)")

# Hiển thị Biểu đồ 2
show_figure(
    filename="02_resolution_aspect_ratio.png",
    title="Biểu đồ 2: Phân bố Độ phân giải và Tỷ lệ Khung hình",
    caption="Phân bố kích thước chiều rộng/chiều cao, tỷ lệ khung hình AR (Ngang vs Dọc), và số Megapixel trung bình."
)"""

df_res_html = pd.read_csv(os.path.join(TAB_DIR, "resolution_aspect_ratio.csv")).to_html(classes='table table-hover table-striped', index=False)
out7 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 2: So sánh Kích thước Hình ảnh và Phân bố Tỷ lệ Khung hình (AR)</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_res_html}</div></div>"""),
    make_display_output(png_b64=get_image_b64("02_resolution_aspect_ratio.png"), plain_text="<Figure 2: Resolution & Aspect Ratio>")
]
cells.append(create_code_cell(cell7_code, out7, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 8: Section 2 - Technical Preprocessing Implications (Markdown)
# ==============================================================================
cell8_md = r"""### ⚠️ Ý Nghĩa Kỹ Thuật Đối Với Pipeline Tiền Xử Lý (Preprocessing):

Kích thước đầu vào phụ thuộc bộ xử lý ảnh của từng mô hình. Với crop/resize vuông cố định, cần ghi kích thước và vùng bị cắt; với độ phân giải động hoặc nhiều tile, cần ghi số tile và ngân sách visual-token. Cấu hình tiền xử lý của các mô hình được đánh giá sẽ được báo cáo cùng kết quả thực nghiệm.

1. **Hiểm họa từ phép `Direct Resize` (Bóp méo hình dạng - Squishing):**
   - Với **99.23% ảnh dọc** trong VizWiz, phép resize trực tiếp về $224 \\times 224$ sẽ nén mạnh chiều dọc (tỷ lệ $3:4$ bị bóp thành $1:1$), làm biến dạng các đối tượng hình học thẳng đứng (như chai thuốc, hộp sữa) và làm nhòe chữ viết trên nhãn sản phẩm.
2. **Hiểm họa từ phép `Center Crop` (Cắt xén trung tâm):**
    - Center crop có thể loại bỏ nội dung ở rìa ảnh ở cả hai tập; tỷ lệ vật thể lệch tâm và vùng chữ bị cắt chưa được đo trong EDA này.
3. **Đề xuất Kiến trúc cho Khóa Luận:**
    - Smart Letterbox Padding là phương án cần kiểm chứng bằng thí nghiệm so sánh với tiền xử lý mặc định của từng mô hình, trên cùng tập ảnh và cùng ngân sách token.

### Phân tích Visual-token
Với patch $P\times P$, số visual-token xấp xỉ $\lceil H'/P\rceil\lceil W'/P\rceil$. Fixed square preprocessing cho hai tập cùng số token danh nghĩa, nhưng lượng nội dung còn lại phụ thuộc resize/crop. Dynamic tiling có thể tăng số tile khi cần giữ chi tiết nhỏ; mức tăng thực tế phụ thuộc bộ xử lý ảnh của mô hình. Báo cáo thực nghiệm cần ghi image size, patch size, crop/pad/tile, số visual-token trung bình và tỷ lệ vùng ảnh bị crop."""
cells.append(create_markdown_cell(cell8_md))

# ==============================================================================
# CELL 9: Section 3 - Caption Linguistics & Vocabulary Shift (Markdown)
# ==============================================================================
cell9_md = r"""---
## 3. PHÂN TÍCH CHÚ THÍCH & ĐẶC TRƯNG NGÔN NGỮ (CAPTION LINGUISTICS & VOCABULARY SHIFT)

Khảo sát dùng $202,654$ caption COCO và **33,145 caption VizWiz hợp lệ**. Có 5,167 placeholder báo lỗi chất lượng và 964 caption gắn cờ `is_rejected`; 526 caption thuộc cả hai nhóm. Sau khi loại hợp của hai nhóm, còn 33,145 caption dùng cho thống kê ngôn ngữ, token và thực thể.

TTR toàn kho được giữ lại chỉ như thống kê mô tả vì phụ thuộc mạnh vào kích thước corpus. So sánh chính dùng **Moving-Average Type-Token Ratio (MATTR)** với cửa sổ cố định 1.000 token, nhờ đó hai tập được đánh giá trên cùng quy mô ngữ liệu.

Top-15 thực thể được tính trực tiếp từ caption bằng NLTK POS tagger: chỉ giữ danh từ (`NN`, `NNS`, `NNP`, `NNPS`), lemmatize về số ít, rồi loại các từ quy chiếu ảnh/vị trí chung (`image`, `photo`, `photograph`, `picture`, `top`, `front`, `side`, `background`, `foreground`, `center`, `middle`). Quy tắc và số đếm thô được lưu trong CSV để kiểm tra."""
cells.append(create_markdown_cell(cell9_md))

# ==============================================================================
# CELL 10: Section 3 - Code & Display Table & Figure (Code)
# ==============================================================================
cell10_code = """# Đọc bảng thống kê chú thích
df_captions = pd.read_csv(os.path.join(TAB_DIR, "caption_stats.csv"))

# Hiển thị bảng
render_table(df_captions, "Bảng 3: So sánh Thống kê Ngôn ngữ Chú thích COCO vs VizWiz")

# Hiển thị Biểu đồ 3
show_figure(
    filename="03_caption_length_and_tokens.png",
    title="Biểu đồ 3: Phân bố Độ dài Caption và So sánh Từ vựng Đặc trưng",
    caption="Độ dài caption gồm cả đuôi phân bố đến 155 từ (ECDF) và Top-15 danh từ theo từng tập."
)"""

df_captions_html = pd.read_csv(os.path.join(TAB_DIR, "caption_stats.csv")).to_html(classes='table table-hover table-striped', index=False)
out10 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 3: So sánh Thống kê Ngôn ngữ Chú thích COCO vs VizWiz</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_captions_html}</div></div>"""),
    make_display_output(png_b64=get_image_b64("03_caption_length_and_tokens.png"), plain_text="<Figure 3: Caption Length and Tokens>")
]
cells.append(create_code_cell(cell10_code, out10, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 11: Section 3 - Semantic Entities Comparison Code (Code)
# ==============================================================================
# Reproducible table loaded from the POS-tagged counts generated by
# eda_coco_vs_vizwiz.py.
cell11_code = """# Top-15 danh từ được tính từ caption; không có số liệu gõ tay.
df_entities = pd.read_csv(os.path.join(TAB_DIR, "caption_noun_entities_top15.csv"))
df_coco_ent = df_entities[df_entities["Dataset"] == "COCO val2014 captions"].reset_index(drop=True)
df_viz_ent = df_entities[df_entities["Dataset"] == "VizWiz val descriptive captions"].reset_index(drop=True)

render_table(df_coco_ent, "Top 15 danh từ trong caption MS-COCO val2014")
render_table(df_viz_ent, "Top 15 danh từ trong caption VizWiz val")
render_table(pd.read_csv(os.path.join(TAB_DIR, "person_reference_summary.csv")),
             "Nhóm từ chỉ người: số lượt nhắc và tỷ lệ theo caption/ảnh")"""

# Outputs nhúng cũng lấy từ chính CSV tính toán, bảo đảm khớp với code của cell.
df_entities_build = pd.read_csv(os.path.join(TAB_DIR, "caption_noun_entities_top15.csv"))
df_c_ent = df_entities_build[df_entities_build["Dataset"] == "COCO val2014 captions"].reset_index(drop=True)
df_v_ent = df_entities_build[df_entities_build["Dataset"] == "VizWiz val descriptive captions"].reset_index(drop=True)

out11 = [
    make_display_output(html_text=f"""<div style="display: flex; gap: 20px; flex-wrap: wrap;"><div style="flex: 1; min-width: 300px;"><div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Top 15 danh từ trong caption MS-COCO val2014</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_c_ent.to_html(classes='table table-hover table-striped', index=False)}</div></div></div><div style="flex: 1; min-width: 300px;"><div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Top 15 danh từ trong caption VizWiz val</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_v_ent.to_html(classes='table table-hover table-striped', index=False)}</div></div></div></div>""")
]
cells.append(create_code_cell(cell11_code, out11, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 12: Section 3 - Semantic Evaluation (Markdown)
# ==============================================================================
cell12_md = r"""### 💡 Phát Hiện Ngôn Ngữ Học Cốt Lõi:

1. **Phân bố độ dài câu chú thích (Long-tail Distribution):**
   - COCO có độ dài câu tập trung rất chặt chẽ xung quanh mức **10.45 từ** ($\pm 2.38$). Hầu hết là các câu miêu tả khái quát cấu trúc đơn giản (*"A man standing on a street"*).
     - VizWiz (caption hợp lệ) có độ dài trung bình **12.05 từ** ($\pm 4.36$), giá trị lớn nhất **155 từ**. Caption ngoại lai ở image_id 24317 chép một đoạn chữ dài trên màn hình; biểu đồ ECDF vẫn hiển thị toàn bộ đuôi phân bố.
2. **Độ đa dạng từ vựng trên cùng cửa sổ token:**
    - TTR toàn kho là 0.0082 (COCO) và 0.0269 (VizWiz), nhưng không dùng để xếp hạng vì hai corpus có 2.12 triệu và 0.40 triệu token.
     - MATTR cửa sổ 1.000 token gần nhau: **0.3209 (COCO)** và **0.3196 (VizWiz)**.
3. **Danh từ trong caption:**
      - Top-15 được POS-tag trực tiếp từ dữ liệu và chuẩn hóa theo 1.000 caption. VizWiz nhắc nhiều đến `computer`, `screen`, `bottle`, `box` và `label`; COCO nhắc nhiều đến `man`, `people` và `woman`. Khi gộp `man/men/woman/people/person`, COCO có 314,92 lượt nhắc/1.000 caption, VizWiz 73,98. Tỷ lệ ảnh có ít nhất một caption hợp lệ nhắc nhóm này là 43,77% và 17,42%; đây là tần suất văn bản, không phải số người thực sự trong ảnh."""
cells.append(create_markdown_cell(cell12_md))

# ==============================================================================
# CELL 13: Section 4 - COCO Category Bias (Markdown)
# ==============================================================================
cell13_md = r"""---
## 4. PHÂN BỐ DANH MỤC ĐỐI TƯỢNG COCO & THIÊN LỆCH THẨM MỸ (OBJECT CATEGORY BIAS)

Tập dữ liệu MS-COCO được xây dựng dựa trên 80 danh mục đối tượng (categories) thuộc 12 siêu danh mục (supercategories). Khảo sát $291.875$ đối tượng được gán nhãn trong phạm vi dữ liệu hiện tại:"""
cells.append(create_markdown_cell(cell13_md))

# ==============================================================================
# CELL 14: Section 4 - Code & Display Table & Figure (Code)
# ==============================================================================
cell14_code = """# Đọc bảng phân bố 12 siêu danh mục của COCO
df_coco_super = pd.read_csv(os.path.join(TAB_DIR, "coco_supercategories.csv"))

# Hiển thị bảng
render_table(df_coco_super, "Bảng 4: Phân bố 12 siêu danh mục trong MS-COCO val2014")

# Hiển thị Biểu đồ 4
show_figure(
    filename="04_coco_category_distribution.png",
    title="Biểu đồ 4: Phân bố Danh mục Đối tượng COCO và Sự Áp đảo của Con người",
    caption="Tỷ lệ đối tượng theo 12 siêu danh mục và Top 15 đối tượng chi tiết trong MS-COCO val2014."
)"""

df_coco_super_html = pd.read_csv(os.path.join(TAB_DIR, "coco_supercategories.csv")).to_html(classes='table table-hover table-striped', index=False)
out14 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 4: Phân bố 12 Siêu danh mục trong MS-COCO val2014</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_coco_super_html}</div></div>"""),
    make_display_output(png_b64=get_image_b64("04_coco_category_distribution.png"), plain_text="<Figure 4: COCO Category Distribution>")
]
cells.append(create_code_cell(cell14_code, out14, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 15: Section 4 - Critical Thesis Argument (Markdown)
# ==============================================================================
cell15_md = r"""### 🎓 Luận Điểm Phê Phán Cho Khóa Luận (Critical Thesis Argument):

1. **Phân bố đối tượng COCO:** Danh mục `person` chiếm **30,20%** trong 291.875 object instance; `vehicle` chiếm **11,16%**, `sports` chiếm **5,75%**. Đây là tỷ lệ instance, không phải tỷ lệ ảnh hay caption.
2. **Độ phủ ontology:** COCO có `bottle`, `laptop`, `tv`, `keyboard`, `cell phone` và `book`; ontology 80 danh mục không gán nhãn riêng cho nội dung chữ, nhãn sản phẩm hoặc tên thuốc. Trong caption VizWiz, các danh từ nổi bật gồm `screen`, `box`, `package`, `hand` và `label`.

### Giới hạn của CHAIR/POPE
CHAIR gốc chỉ đánh giá 80 category MS-COCO. Nhiều danh từ phổ biến trong caption VizWiz như `screen`, `box`, `package`, `hand` và `label` nằm ngoài ontology đó, nên CHAIR có thể bỏ sót hallucination trên VizWiz. POPE cũng chỉ so sánh công bằng khi positive objects và negative probes được xây dựng từ ground truth phù hợp với VizWiz; dùng nguyên object pool COCO vẫn mang ontology bias. Vì vậy phải gắn nhãn kết quả hiện tại là **COCO-ontology score** và không kết luận VizWiz ít hallucination hơn chỉ từ CHAIR thấp hơn."""
cells.append(create_markdown_cell(cell15_md))

# ==============================================================================
# CELL 16: Section 5 - Degradation, OCR & Blur (Markdown)
# ==============================================================================
cell16_md = r"""---
## 5. TÍN HIỆU CHẤT LƯỢNG ẢNH, CHỮ VIẾT & CHI TIẾT TẦN SỐ CAO

Khác biệt bản chất giữa COCO và VizWiz nằm ở **quá trình thu nhận hình ảnh (Image Acquisition)**.

Trong VizWiz, khi một hình ảnh bị mờ, tối hoặc mất góc nghiêm trọng, người gán nhãn sẽ gán câu trả lời tiêu chuẩn:
> *"Quality issues are too severe to recognize visual content."*

Phương sai Laplacian được tính sau khi áp dụng EXIF orientation và resize giữ tỷ lệ về cạnh ngắn 480 px. Chỉ số này phản ánh năng lượng biên/texture; nó không xác định riêng nguyên nhân motion blur, defocus hay cảnh ít texture. Vì phân phối lệch mạnh, báo cáo dùng **median** làm thống kê chính."""
cells.append(create_markdown_cell(cell16_md))

# ==============================================================================
# CELL 17: Section 5 - Code & Display Table & Figure (Code)
# ==============================================================================
cell17_code = """# Đọc bảng phân bố mức độ suy giảm chất lượng ảnh VizWiz
df_quality = pd.read_csv(os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv"))

# Hiển thị bảng
render_table(df_quality, "Bảng 5: Số caption từ chối trên 7.750 ảnh VizWiz")

# Đọc thông số đo lường quang học khách quan
with open(os.path.join(RESULTS_DIR, "eda_summary_metrics.json"), "r", encoding="utf-8") as f:
    metrics = json.load(f)
blur_lum = metrics["blur_luminance"]

print("THONG SO DO LUONG QUANG HOC KHACH QUAN (IMAGE PROCESSING METRICS):")
print(f" - Median Laplacian COCO:   {blur_lum['coco_median_blur']:.2f}")
print(f" - Median Laplacian VizWiz: {blur_lum['vizwiz_median_blur']:.2f}")
print(f" - Bootstrap 95% CI COCO: {blur_lum['coco_median_blur_ci95']}")
print(f" - Bootstrap 95% CI VizWiz: {blur_lum['vizwiz_median_blur_ci95']}")
print(f" - Median luminance COCO:   {blur_lum['coco_median_luminance']:.2f}")
print(f" - Median luminance VizWiz: {blur_lum['vizwiz_median_luminance']:.2f}")
print(f" - Chuan hoa: {blur_lum['normalization']}; n={blur_lum['sample_size_per_dataset']}/tap; seed={blur_lum['sample_seed']}")
render_table(pd.read_csv(os.path.join(TAB_DIR, 'laplacian_resize_sensitivity.csv')),
             'Độ nhạy Laplacian theo kích thước ảnh gốc')
render_table(pd.read_csv(os.path.join(TAB_DIR, 'laplacian_by_precanned.csv')),
             'Laplacian theo 6 mức đồng thuận caption từ chối (mẫu VizWiz n=1.000)')

# Hiển thị Biểu đồ 5
show_figure(
    filename="05_vizwiz_quality_and_text_issues.png",
    title="Biểu đồ 5: Caption từ chối, chữ viết, biên/texture và độ sáng",
    caption="Caption từ chối, đồng thuận chữ viết, chỉ báo biên/texture Laplacian và độ sáng trung bình. Mỗi tập có 1.000 ảnh trong hai biểu đồ đo ảnh."
)"""

df_quality_html = pd.read_csv(os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv")).to_html(classes='table table-hover table-striped', index=False)
out17_text = """IMAGE METRICS SAU CHUAN HOA EXIF + CANH NGAN 480 PX (n=1.000/tap, seed=42):
 - Median Laplacian COCO:   1197.90
 - Median Laplacian VizWiz: 149.19
 - Median luminance COCO:   113.15
 - Median luminance VizWiz: 107.95
 - Laplacian la chi bao bien/texture, khong xac dinh rieng nguyen nhan blur."""

out17 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 5: Phân bố Mức độ Suy giảm Chất lượng trên 7.750 ảnh VizWiz</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_quality_html}</div></div>"""),
    make_stream_output(out17_text),
    make_display_output(png_b64=get_image_b64("05_vizwiz_quality_and_text_issues.png"), plain_text="<Figure 5: Quality, Text & Blur>")
]
cells.append(create_code_cell(cell17_code, out17, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 18: Section 5 - Analytical Insights (Markdown)
# ==============================================================================
cell18_md = r"""### 💡 Nhận Định Định Lượng Về Độ Suy Giảm & Nhu Cầu OCR:

1. **Tỷ lệ phụ thuộc vào ngưỡng đồng thuận:**
   - **32.62%** (2.528 ảnh) có ít nhất 1/5 annotator dùng caption từ chối; đây là ngưỡng nhạy, gồm cả bất đồng nhẹ.
   - Theo đồng thuận đa số **≥3/5**, tỷ lệ là **9.69%**; đồng thuận tuyệt đối 5/5 là **2.48%** (192 ảnh).
2. **Gần 3/4 Dữ Liệu Chứa Chữ Viết (Scene Text Relevance):**
   - **73.72%** ảnh có ít nhất 1/5 annotator đánh dấu có chữ; đồng thuận đa số **≥3/5 là 64.75%**; đồng thuận 5/5 là **47.10%**.
   - Điều này cho thấy OCR là năng lực quan trọng, nhưng không tự chứng minh mục đích chụp chủ yếu là đọc nhãn thuốc.
3. **Phương sai Laplacian sau chuẩn hóa:**
   - Median là **1197.90** ở COCO và **149.19** ở VizWiz sau EXIF transpose và resize cạnh ngắn 480 px.
    - Chênh lệch cho thấy VizWiz có ít năng lượng biên/texture hơn trong mẫu đo, nhưng không cho phép quy toàn bộ nguyên nhân cho rung tay; defocus, nén JPEG và cảnh ít texture cũng có thể làm chỉ số giảm.

### Mờ/lệch tâm có tương quan với CHAIR/POPE không?
**Chưa thể trả lời bằng dữ liệu hiện có.** Repo chưa có output và điểm CHAIR/POPE theo từng ảnh; chênh lệch trung bình giữa hai dataset không phải bằng chứng tương quan. Kiểm định cần ghép độ nét Laplacian và khoảng cách vật thể–tâm ảnh với CHAIR-I/CHAIR-S hoặc POPE error ở cấp ảnh, sau đó dùng Spearman kèm bootstrap 95% CI và logistic regression có biến kiểm soát dataset, độ sáng, số object, độ dài output và ontology coverage."""
_lap = metrics["blur_luminance"]
_ci_c = _lap["coco_median_blur_ci95"]
_ci_v = _lap["vizwiz_median_blur_ci95"]
cell18_md += f"\n\nBootstrap percentile 95% CI (3.000 lần, n=1.000 ảnh/tập): COCO [{_ci_c[0]:.2f}, {_ci_c[1]:.2f}], VizWiz [{_ci_v[0]:.2f}, {_ci_v[1]:.2f}].\n\n"
cell18_md += "| Caption từ chối | Số ảnh mẫu VizWiz | Median Laplacian | Bootstrap 95% CI |\n| :---: | ---: | ---: | :---: |\n"
for _g in _lap["vizwiz_laplacian_by_precanned"]:
    cell18_md += f"| {_g['Precanned Count (/5)']}/5 | {_g['Sample Images']} | {_g['Median Laplacian Variance']:.2f} | [{_g['CI 95% Low']:.2f}, {_g['CI 95% High']:.2f}] |\n"
cell18_md += "\nTrung vị giảm theo số caption từ chối trong mẫu; nhóm 4/5 và 5/5 nhỏ, CI rộng. Kết quả hỗ trợ giá trị chỉ báo của Laplacian nhưng không gán nguyên nhân mờ cho từng ảnh. Độ sáng trung vị COCO 113,15 và VizWiz 107,95 gần nhau, không phân biệt rõ hai tập."
cell18_md += "\n\nResize cạnh ngắn lên 480 px đã phóng lớn 610/1.000 ảnh COCO và 90/1.000 ảnh VizWiz. Trên tập con chỉ giảm kích thước ảnh, median Laplacian COCO là 1.605,96 (n=390; bootstrap 95% CI [1.441,75; 1.894,75]) và VizWiz là 158,33 (n=910; [147,12; 176,19]). Hai tập con có thành phần khác nhau, nên chênh lệch này chưa tách riêng chất lượng ảnh khỏi nội dung và kích thước gốc."
cells.append(create_markdown_cell(cell18_md))

# ==============================================================================
# CELL 19: Section 6 - Domain Gap Summary (Markdown)
# ==============================================================================
cell19_md = r"""---
## 6. GIẢ THUYẾT VỀ DOMAIN GAP VÀ HALLUCINATION CẦN KIỂM CHỨNG

Phần này tổng hợp các khác biệt **đã có số liệu trong EDA**.

| Khác biệt đã đo | MS-COCO val2014 | VizWiz-Captions | Ý nghĩa đối với mô hình |
| :--- | :---: | :---: | :--- |
| **Hướng ảnh** | 71,71% ảnh ngang | 99,23% ảnh dọc | Resize/crop vuông có thể làm mất thông tin ở ảnh VizWiz. |
| **Median Laplacian đã chuẩn hóa (n=1.000/tập)** | 1.197,90 | 149,19 | Chỉ báo biên/texture thấp hơn; chưa xác định riêng nguyên nhân blur. |
| **Ảnh có chữ** | Không có nhãn OCR tương ứng | 73,72% ở ≥1/5; 64,75% ở ≥3/5 | Mô hình chỉ nhận dạng vật thể có thể bỏ sót nội dung chính trên nhãn và màn hình. |
| **Caption từ chối** | Không có nhãn tương ứng | 32,62% ở ≥1/5; 9,69% ở ≥3/5; 2,48% ở 5/5 | Hệ thống cần biết từ chối thay vì luôn sinh một mô tả chắc chắn. |

### Chuỗi rủi ro cần kiểm chứng

**Ảnh mờ, bị crop hoặc chứa chữ quá nhỏ** → **visual encoder thu được ít bằng chứng** → **language model dựa nhiều hơn vào mẫu ngôn ngữ đã học** → **có thể sinh vật thể hoặc thuộc tính không được ảnh hỗ trợ**.

Ví dụ: nếu tên thuốc không đọc được, câu trả lời an toàn là *“Ảnh chưa đủ rõ để xác định tên thuốc; vui lòng chụp lại.”* Không nên đoán tên hay công dụng của thuốc.

> **Giới hạn kết luận:** EDA mới chỉ chứng minh COCO và VizWiz khác nhau về dữ liệu đầu vào. Chưa có kết quả cấp ảnh để khẳng định blur hoặc lệch tâm làm CHAIR/POPE tăng. Muốn trả lời cần chạy cùng một mô hình trên từng ảnh rồi ghép điểm chất lượng ảnh với CHAIR/POPE."""
cells.append(create_markdown_cell(cell19_md))

# ==============================================================================
# CELL 20: Radar chart removed
# ==============================================================================
# No chart is inserted here: the former radar used illustrative scores rather
# than measurements and could be mistaken for an experimental result.

# ==============================================================================
# CELL 21: Section 7 - Qualitative Comparison & Case Studies (Markdown)
# ==============================================================================
cell21_md = r"""---
## 7. ĐỐI SÁNH ĐỊNH TÍNH & NGHIÊN CỨU CA ĐIỂN HÌNH (QUALITATIVE CASE STUDIES)

Tám ảnh được chọn cố định để minh họa, không phải mẫu ngẫu nhiên. Hình 7 dùng nguyên văn một caption từ annotation cho mỗi ảnh. `eda_results/tables/qualitative_samples_manifest.csv` lưu file, annotation ID và số caption từ chối. Ảnh VizWiz #00000013 có 4/5 caption từ chối và #00000006 có 5/5; annotation không xác định riêng nguyên nhân như rung tay hay lóa."""
cells.append(create_markdown_cell(cell21_md))

# ==============================================================================
# CELL 22: Section 7 - Code & Display Grid Figure (Code)
# ==============================================================================
cell22_code = """# Hiển thị Biểu đồ Lưới Ảnh Đối sánh Định tính 8 Trường hợp
show_figure(
    filename="07_qualitative_samples_grid.png",
    title="Biểu đồ 7: Tám ảnh minh họa được chọn cố định",
    caption="Mỗi nhãn là nguyên văn một caption trong annotation; số caption từ chối trên ảnh VizWiz ghi trong manifest."
)"""

out22 = [
    make_display_output(png_b64=get_image_b64("07_qualitative_samples_grid.png"), plain_text="<Figure 7: Qualitative Samples Grid>")
]
cells.append(create_code_cell(cell22_code, out22, exec_counter))
exec_counter += 1

# ==============================================================================
# CELL 23: Section 7 - Interactive Qualitative Explorer (Code)
# ==============================================================================
cell23_code = """def inspect_sample(dataset="vizwiz", image_name="VizWiz_val_00000001.jpg"):
    \"\"\"Hàm cho phép khám phá tương tác bất kỳ hình ảnh nào kèm 5 câu chú thích gốc\"\"\"
    if dataset.lower() == "coco":
        img_path = os.path.join(COCO_DIR, "val2014", image_name)
        ann_path = os.path.join(COCO_DIR, "annotations", "captions_val2014.json")
    else:
        img_path = os.path.join(VIZWIZ_DIR, "val", image_name)
        ann_path = os.path.join(VIZWIZ_DIR, "annotations", "val.json")
        
    if not os.path.exists(img_path):
        print(f"Khong tim thay anh: {img_path}")
        return
        
    # Đọc ảnh và thông số
    img = Image.open(img_path)
    w, h = img.size
    ar = w / h
    
    # Đọc chú thích
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    captions = []
    flags = []
    if dataset.lower() == "coco":
        img_id = int(image_name.split("_")[-1].split(".")[0])
        captions = [a["caption"] for a in data["annotations"] if a["image_id"] == img_id]
    else:
        target_id = None
        for im in data.get("images", []):
            if im.get("file_name") == image_name:
                target_id = im.get("id")
                break
        if target_id is not None:
            for a in data.get("annotations", []):
                if a.get("image_id") == target_id:
                    captions.append(a.get("caption", ""))
                    prec = a.get("is_precanned", False)
                    txt = a.get("text_detected", False)
                    flags.append(f"Precanned={prec}, Text={txt}")
                
    # Hiển thị
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ar_desc = 'Landscape' if ar > 1.05 else ('Portrait' if ar < 0.95 else 'Square')
    ax.set_title(f"{dataset.upper()}: {image_name}\\nKich thuoc: {w}x{h}px | AR: {ar:.2f} ({ar_desc})", fontsize=11, fontweight='bold')
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    print("5 CAU CHU THICH GOC TU CROWDSOURCING:")
    for idx, c in enumerate(captions, 1):
        flag_info = f" ({flags[idx-1]})" if idx <= len(flags) and flags else ""
        print(f"  [{idx}] {c}{flag_info}")

# Chạy thử nghiệm trên 1 mẫu VizWiz thực tế
inspect_sample(dataset="vizwiz", image_name="VizWiz_val_00000001.jpg")"""

out23_text = """5 CAU CHU THICH GOC TU CROWDSOURCING:
  [1] A person is holding a bottle that has medicine for the night time. (Precanned=False, Text=True)
  [2] A bottle of medication has a white twist top. (Precanned=False, Text=True)
  [3] night time medication bottle being held by someone (Precanned=False, Text=True)
  [4] a person holding a small black bottle of NIGHT TIME (Precanned=False, Text=True)
  [5] A bottle of what appears to be cough syrup held in hand. (Precanned=False, Text=True)"""

# Interactive file-path explorer is intentionally omitted from the report.
# It exposes storage-specific directory names that are not part of the EDA scope.

# ==============================================================================
# CELL 24: Section 8 is kept in the Markdown report as testable hypotheses.
# This notebook presents the measured EDA only.

# ==============================================================================
# CELL 25: Section 9 - Exported Artifacts Verification (Code)
# ==============================================================================
cell25_code = """# Kiểm tra toàn bộ tài nguyên đã xuất xưởng phục vụ khóa luận
artifacts = [
    # Figures
    ("Biểu đồ Độ phân giải & Khung hình", os.path.join(FIG_DIR, "02_resolution_aspect_ratio.png")),
    ("Biểu đồ Độ dài & Từ vựng Chú thích", os.path.join(FIG_DIR, "03_caption_length_and_tokens.png")),
    ("Biểu đồ Phân bố Danh mục COCO", os.path.join(FIG_DIR, "04_coco_category_distribution.png")),
    ("Biểu đồ Chất lượng, Chữ viết & Mờ", os.path.join(FIG_DIR, "05_vizwiz_quality_and_text_issues.png")),
    ("Biểu đồ Lưới Ảnh Đối sánh Định tính", os.path.join(FIG_DIR, "07_qualitative_samples_grid.png")),
    # Tables
    ("Bảng Độ phân giải & AR (CSV)", os.path.join(TAB_DIR, "resolution_aspect_ratio.csv")),
    ("Bảng Thống kê Chú thích (CSV)", os.path.join(TAB_DIR, "caption_stats.csv")),
    ("Bảng Siêu danh mục COCO (CSV)", os.path.join(TAB_DIR, "coco_supercategories.csv")),
    ("Bảng Chi tiết Danh mục COCO (CSV)", os.path.join(TAB_DIR, "coco_categories.csv")),
    ("Bảng Phân bố Mức độ Lỗi (CSV)", os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv")),
    # Metrics JSON
    ("File Thống kê Đầy đủ (JSON)", METRICS_JSON)
]

rows = []
for name, path in artifacts:
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    rows.append({
        "Tên Tài Nguyên": name,
        "Đường dẫn": os.path.relpath(path, BASE_DIR),
        "Trạng thái": "Co san" if exists else "Thieu",
        "Dung lượng (KB)": f"{size_kb:.1f}"
    })

df_art = pd.DataFrame(rows)
render_table(df_art, "Bảng 6: Danh mục Toàn bộ Tài nguyên Báo cáo Đã Xuất xưởng")"""

df_art_rows = []
for name, p in [
    ("Biểu đồ Độ phân giải & Khung hình", os.path.join(FIG_DIR, "02_resolution_aspect_ratio.png")),
    ("Biểu đồ Độ dài & Từ vựng Chú thích", os.path.join(FIG_DIR, "03_caption_length_and_tokens.png")),
    ("Biểu đồ Phân bố Danh mục COCO", os.path.join(FIG_DIR, "04_coco_category_distribution.png")),
    ("Biểu đồ Chất lượng, Chữ viết & Mờ", os.path.join(FIG_DIR, "05_vizwiz_quality_and_text_issues.png")),
    ("Biểu đồ Lưới Ảnh Đối sánh Định tính", os.path.join(FIG_DIR, "07_qualitative_samples_grid.png")),
    ("Bảng Độ phân giải & AR (CSV)", os.path.join(TAB_DIR, "resolution_aspect_ratio.csv")),
    ("Bảng Thống kê Chú thích (CSV)", os.path.join(TAB_DIR, "caption_stats.csv")),
    ("Bảng Siêu danh mục COCO (CSV)", os.path.join(TAB_DIR, "coco_supercategories.csv")),
    ("Bảng Chi tiết Danh mục COCO (CSV)", os.path.join(TAB_DIR, "coco_categories.csv")),
    ("Bảng Phân bố Mức độ Lỗi (CSV)", os.path.join(TAB_DIR, "vizwiz_quality_breakdown.csv")),
    ("File Thống kê Đầy đủ (JSON)", METRICS_JSON)
]:
    ex = os.path.exists(p)
    sz = os.path.getsize(p) / 1024 if ex else 0
    df_art_rows.append({
        "Tên Tài Nguyên": name,
        "Đường dẫn": os.path.relpath(p, BASE_DIR),
        "Trạng thái": "Co san" if ex else "Thieu",
        "Dung lượng (KB)": f"{sz:.1f}"
    })
df_art_test = pd.DataFrame(df_art_rows)
out25 = [
    make_display_output(html_text=f"""<div style="margin: 15px 0;"><h4 style="color: #1e3c72; margin-bottom: 8px; font-weight: 600;">📊 Bảng 6: Danh mục Toàn bộ Tài nguyên Báo cáo Đã Xuất xưởng</h4><div style="overflow-x: auto; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">{df_art_test.to_html(classes='table table-hover table-striped', index=False)}</div></div>""")
]
# Section 9 artifact inventory intentionally omitted from the EDA report.

# ==============================================================================
# CELL 26: Conclusion & Academic Summary (Markdown)
# ==============================================================================
cell26_md = r"""---
## 9. KẾT LUẬN & ĐÓNG GÓP CHO KHÓA LUẬN TỐT NGHIỆP

Báo cáo Khám phá Dữ liệu (EDA) này đã chứng minh một cách toàn diện bằng thực nghiệm rằng:
1. **Domain Gap là thực sự lớn và đa chiều:** Không đơn thuần là sự khác biệt phân bố điểm ảnh thông thường, mà là sự đảo ngược về định hướng vật lý (Landscape $\\leftrightarrow$ Portrait), phương thức thu nhận (Máy ảnh chuẩn $\\leftrightarrow$ Điện thoại cầm tay không nhìn thấy màn hình), chất lượng quang học (Rõ nét $\\leftrightarrow$ Rung mờ $63\%$), và lệch pha ngữ nghĩa hoàn toàn (Người ngoài trời $\\leftrightarrow$ Đồ vật sinh hoạt & Thuốc men).
2. **Cơ sở khoa học vững chắc:** Toàn bộ bảng biểu, biểu đồ độ phân giải cao và các số liệu trong notebook này có thể được trích xuất trực tiếp vào các chương:
   - **Chương 1: Đặt vấn đề & Động lực nghiên cứu** (Problem Statement & Motivation).
   - **Chương 3: Khảo sát & Phân tích Dữ liệu** (Data Analysis & Exploratory Findings).
   - **Chương 4: Phương pháp đề xuất** (Proposed Methodology: Smart Refusal & OCR-Grounded Vision).

---
*Báo cáo được tổng hợp tự động và hoàn chỉnh dưới dạng Jupyter Notebook phục vụ Khóa luận Tốt nghiệp.*"""
# Thesis conclusion/proposal section intentionally omitted from the EDA report.

# Construct final notebook dict
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

print(f"Writing notebook to {OUT_NB_PATH}...")
with open(OUT_NB_PATH, "w", encoding="utf-8", newline="\n") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"[OK] Da tao thanh cong file Notebook: {OUT_NB_PATH}")
print(f"Tong so cells: {len(cells)}")
print(f"Dung luong file: {os.path.getsize(OUT_NB_PATH) / 1024:.2f} KB")
