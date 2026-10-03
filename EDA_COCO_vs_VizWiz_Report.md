# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA): ĐỐI SÁNH TẬP DỮ LIỆU MS-COCO VÀ VIZWIZ-CAPTIONS
## Phục vụ Khóa Luận Tốt Nghiệp: Nghiên cứu Domain Gap và Thách thức Thị giác Đời thực

---

## TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo so sánh MS-COCO val2014 (40.504 ảnh) và VizWiz-Captions val (7.750 ảnh). Các kết quả dưới đây mô tả khác biệt dữ liệu đầu vào; báo cáo chưa đo sai số của mô hình hoặc quan hệ nhân quả với hallucination.

### Các phát hiện định lượng cốt lõi

1. COCO có 71,71% ảnh ngang; VizWiz có 99,23% ảnh dọc.
2. Kích thước trung bình lần lượt là 576,5 × 485,1 px và 1.032,1 × 1.374,7 px. Kích thước VizWiz được đọc sau EXIF transpose.
3. Phương sai Laplacian chuẩn hóa cạnh ngắn 480 px có trung vị 1.197,90 ở COCO và 149,19 ở VizWiz (mỗi tập 1.000 ảnh). Đây là chỉ báo biên/texture, không phải nhãn chất lượng ảnh.
4. Trong VizWiz, 32,62% ảnh có ít nhất một caption từ chối; 9,69% có ít nhất ba và 2,48% có năm caption từ chối.
5. Có 73,72% ảnh VizWiz được ít nhất một annotator đánh dấu có chữ; 64,75% đạt đồng thuận đa số.
6. Caption VizWiz nhắc nhiều đến computer (94,42), screen (91,80), bottle (81,53), box (74,80) và label (39,57) trên 1.000 caption. COCO nổi bật với man (122,31), people (69,46) và woman (64,32). Riêng person là 50,20 ở VizWiz và 41,75 ở COCO; các dạng chỉ người chưa được gộp.

---

## PHẦN 1: PHẠM VI ĐỐI SÁNH DỮ LIỆU

Phần mở đầu mô tả **hai miền dữ liệu** theo đúng phạm vi ảnh và chú thích thực tế được khảo sát:

| Miền dữ liệu | Phạm vi dùng trong EDA | Vai trò trong đối sánh | Đặc trưng thu nhận |
| :--- | :---: | :--- | :--- |
| **MS-COCO val2014** | 40.504 ảnh có caption và instance annotation | Miền ảnh đối chiếu | Ảnh trong tập validation COCO |
| **VizWiz-Captions** | 7.750 ảnh, 38.750 caption gốc | Miền hỗ trợ thị giác đời thực | Ảnh smartphone do người khiếm thị chụp |
| **COCO-500 (báo cáo lại Bảng 2 & 4)** | 500 ảnh lấy ngẫu nhiên, `seed=42` | Kiểm tra độ ổn định khi cân bằng quy mô đánh giá | Danh sách ID được lưu để tái lập |

Đơn vị phân tích được ghi rõ ở từng bảng: thống kê ảnh dùng số ảnh; phân tích ngôn ngữ dùng caption mô tả hợp lệ; phân bố đối tượng dùng số object instance. Cách trình bày này tránh cộng gộp những tập không có cùng loại annotation.

---

## PHẦN 2: ĐỘ PHÂN GIẢI & TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO)

Khảo sát gồm 40.504 ảnh MS-COCO val2014 và 7.750 ảnh VizWiz-Captions val. Kích thước VizWiz được đo từ file sau EXIF transpose (493 ảnh có thẻ xoay 90°); kích thước COCO lấy từ annotation. Trước EXIF, tỷ lệ ảnh dọc VizWiz là 92,86%; sau EXIF là 99,23%.

Để đối chiếu theo yêu cầu với đúng 500 ảnh COCO, một mẫu ngẫu nhiên cố định (`seed=42`) cũng được chạy lại riêng. Mẫu cho kết quả: kích thước trung bình $576,9\times481,8$ px, $73,60\%$ ảnh ngang, $22,80\%$ ảnh dọc và $3,60\%$ ảnh vuông. Ảnh chụp bảng: [`table2_coco500_resolution.png`](eda_results/coco500/table2_coco500_resolution.png).

| Chỉ số hình ảnh | MS-COCO val2014 ($N=40.504$) | VizWiz-Captions ($N=7.750$) | Ý nghĩa trong Khóa luận |
| :--- | :---: | :---: | :--- |
| **Chiều rộng trung bình (Mean Width)** | $576.5 \pm 91.9\text{ px}$ | **$1032.1 \pm 334.1\text{ px}$** | VizWiz có phương sai kích thước rất lớn |
| **Trung vị Chiều rộng (Median Width)** | $640.0\text{ px}$ | $968.0\text{ px}$ | COCO bị giới hạn cận trên cố định |
| **Chiều rộng [Min, Max]** | $[120, 640]\text{ px}$ | **$[23, 2448]\text{ px}$** | VizWiz có ảnh siêu nhỏ ($23\text{px}$) và siêu lớn |
| **Chiều cao trung bình (Mean Height)** | $485.1 \pm 97.8\text{ px}$ | **$1374.7 \pm 445.2\text{ px}$** | Chiều dọc VizWiz vượt trội chiều ngang |
| **Trung vị Chiều cao (Median Height)** | $480.0\text{ px}$ | $1296.0\text{ px}$ | Độ phân giải camera điện thoại Apple/Android |
| **Chiều cao [Min, Max]** | $[111, 640]\text{ px}$ | $[25, 3264]\text{ px}$ | Chênh lệch độ phân giải cực đoan |
| **Tỷ lệ khung hình trung bình (AR Mean)** | **$1.25$** (Ngang) | **$0.75$** (Dọc) | **Sự đảo ngược định hướng vật lý** |
| **Trung vị AR (Median AR)** | $1.33$ ($4:3$) | $0.75$ ($3:4$) | Tỷ lệ cảm biến chuẩn của 2 thiết bị khác nhau |
| **Tỷ lệ ảnh Ngang (Landscape: $AR > 1.05$)** | **$71.71\%$** ($29,044$ ảnh) | **$0.62\%$** ($48$ ảnh) | Người sáng mắt cầm máy ngang chụp phong cảnh |
| **Tỷ lệ ảnh Dọc (Portrait: $AR < 0.95$)** | **$23.70\%$** ($9,601$ ảnh) | **$99.23\%$** ($7,690$ ảnh) | **Người dùng cầm điện thoại thẳng đứng** |
| **Tỷ lệ ảnh Vuông ($0.95 \le AR \le 1.05$)** | $4.59\%$ ($1,859$ ảnh) | $0.15\%$ ($12$ ảnh) | Rất ít ảnh vuông tự nhiên |
| **Số Megapixel trung bình (Mean MP)** | $0.276\text{ MP}$ | **$1.566\text{ MP}$** | VizWiz nặng hơn 5.6 lần về mật độ điểm ảnh |

Ngoại lai 23 px là chiều rộng của ảnh VizWiz_val_00001384.jpg (23 × 30 px sau EXIF). Đây là ảnh cực nhỏ trong file nguồn, không đại diện cho độ phân giải thông thường của VizWiz. Các giá trị min/max được giữ để mô tả toàn bộ phân bố.

Trong Hình 2C2, nhiều ảnh có cùng kích thước cảm biến và chồng lên một tọa độ. Mỗi bong bóng biểu diễn một kích thước khác nhau; diện tích bong bóng tăng theo số ảnh có kích thước đó. Toàn bộ 7.750 ảnh VizWiz được tính.

### Hàm ý cho tiền xử lý ảnh

Hai tập khác nhau rõ về tỷ lệ khung hình. Phép resize vuông có thể làm biến dạng tỷ lệ; center crop có thể bỏ vùng rìa. EDA này chưa đo vị trí chủ thể hoặc tỷ lệ chữ bị cắt, nên hiệu quả của Smart Letterbox Padding cần được kiểm chứng trên từng mô hình được đánh giá. Một số bộ xử lý dùng crop/resize cố định, số khác dùng độ phân giải động hoặc nhiều tile; báo cáo thực nghiệm cần ghi đúng cấu hình tương ứng.

### Phân tích visual-token:

Visual-token không đồng nghĩa với pixel. Với một vision encoder chia ảnh thành patch kích thước $P\times P$, số token trước khi pooling xấp xỉ
$$N_{visual}=\left\lceil H'/P\right\rceil\left\lceil W'/P\right\rceil,$$
trong đó $H',W'$ là kích thước **sau tiền xử lý**. Vì vậy cần tách hai trường hợp:

- **Fixed square resize/crop**: COCO và VizWiz đều có cùng số token danh nghĩa (ví dụ ảnh $224\times224$, patch $16\times16$ cho 196 patch token), nhưng lượng thông tin được giữ lại phụ thuộc phép resize/crop. Số token bằng nhau không có nghĩa lượng thông tin hữu ích bằng nhau.
- **Aspect-ratio preserving / dynamic tiling**: ảnh dọc hoặc chứa chữ nhỏ có thể cần nhiều tile/token hơn nếu giữ độ chi tiết. Nếu ép cùng ngân sách token với COCO, mỗi token phải đại diện vùng nội dung lớn hơn hoặc hệ thống phải downsample nhiều hơn, làm giảm khả năng đọc nhãn/OCR.
- **Hệ quả cần báo cáo**: cùng với CHAIR/POPE, cần ghi rõ image size, patch size, chiến lược crop/pad/tile, số visual-token trung bình và tỷ lệ vùng ảnh bị crop. Nếu thiếu các thông tin này, chưa thể quy kết chênh lệch hallucination cho bản thân mô hình thay vì cho bước tokenization.

---

## PHẦN 3: PHÂN TÍCH CHÚ THÍCH & ĐẶC TRƯNG NGÔN NGỮ

VizWiz có 5.167/38.750 caption ($13,33\%$) là cùng một câu placeholder báo ảnh không nhận diện được. Các caption này được **loại khỏi toàn bộ thống kê ngôn ngữ, token và thực thể**; chúng chỉ được giữ như nhãn chất lượng ở Phần 5. Sau khi lọc, bảng ngôn ngữ dùng $202.654$ caption COCO và $33.583$ caption VizWiz mô tả hợp lệ:

| Chỉ số Ngôn ngữ | COCO | VizWiz (đã bỏ placeholder) |
| :--- | :---: | :---: |
| **Số lượng Caption ($N$)** | $202.654$ | $33.583$ |
| **Độ dài từ trung bình (Word Mean)** | $10,45 \pm 2,38$ | **$12,04 \pm 4,34$** |
| **Trung vị số từ (Word Median)** | $10,0$ | $11,0$ |
| **Khoảng số từ [Min, Max]** | $[6, 50]$ | **$[6, 155]$** |
| **Độ dài ký tự trung bình (Char Mean)**| $52,29$ | $62,48$ |
| **Tổng số token từ** | $2.122.711$ | $407.796$ |
| **Quy mô từ vựng** | $17.348$ | $10.876$ |
| **Type-Token Ratio (TTR)** | $0,0082$ | **$0,0267$** |
| **MATTR (cửa sổ 1.000 token)** | $0,3209$ | $0,3223$ |

Giá trị lớn nhất 155 từ là một caption chép lại đoạn chữ dài trên màn hình trong ảnh có image_id 24317; nó không đại diện cho độ dài caption điển hình. Giá trị nhỏ nhất 6 từ ở cả hai tập là câu mô tả ngắn hợp lệ theo cách đếm bằng khoảng trắng, không phải ảnh bị thiếu annotation.

### Các Phát hiện Ngôn ngữ học Nổi bật:
1. **Độ dài câu và đuôi phân phối dài (Long-tail)**:
   - Câu chú thích COCO có độ dài tập trung hẹp xung quanh 10 từ (độ lệch chuẩn chỉ $2.38$).
   - Câu chú thích VizWiz miêu tả có độ dài trung bình lớn hơn ($12.04$ từ) và kéo dài tới **$155$ từ**. Giá trị 155 từ là ngoại lai độ dài; không thể suy ra nguyên nhân chỉ từ thống kê tổng hợp.
2. **Độ đa dạng từ vựng trên cùng quy mô**:
   - TTR toàn kho không được dùng để xếp hạng vì COCO có 2,12 triệu token còn VizWiz có 0,41 triệu. MATTR cửa sổ 1.000 token gần như bằng nhau: **0.3209 (COCO)** và **0.3223 (VizWiz)**.
3. **Top-15 danh từ tính từ dữ liệu**:
   - NLTK POS tagger chỉ giữ danh từ và WordNet lemmatizer chuẩn hóa số ít. Bảng báo cả count lẫn số lượt nhắc trên 1.000 caption.
   - VizWiz nhắc nhiều đến `computer`, `screen`, `bottle`, `box` và `label`; COCO nổi bật với `man`, `people` và `woman`. `person` riêng lẻ đạt 50,20/1.000 caption ở VizWiz và 41,75 ở COCO. CSV nguồn: [`caption_noun_entities_top15.csv`](eda_results/tables/caption_noun_entities_top15.csv).

---

## PHẦN 4: PHÂN BỐ DANH MỤC ĐỐI TƯỢNG COCO (OBJECT CATEGORY BIAS)

Tập dữ liệu COCO được xây dựng dựa trên hệ thống phân cấp 80 danh mục đối tượng thuộc 12 siêu danh mục (Supercategories). Phân tích $291.875$ đối tượng được gán nhãn trong phạm vi khảo sát cho thấy:

Trên đúng mẫu COCO-500 ở Phần 1 có 3.719 object instance. Ba siêu danh mục lớn nhất vẫn là `person` (29,52%), `vehicle` (12,34%) và `kitchen` (10,24%), gần sát phân bố toàn tập. Ảnh chụp bảng: [`table4_coco500_supercategories.png`](eda_results/coco500/table4_coco500_supercategories.png).

### 12 siêu danh mục trong MS-COCO val2014:
1. **`person` (Con người)**: **$88,153$ đối tượng ($30.20\%$)** — chiếm gần $1/3$ toàn bộ đối tượng trong tập dữ liệu!
2. **`vehicle` (Phương tiện giao thông)**: $32,574$ ($11.16\%$)
3. **`kitchen` (Đồ dùng nhà bếp)**: $29,445$ ($10.09\%$)
4. **`furniture` (Nội thất)**: $26,296$ ($9.01\%$)
5. **`animal` (Động vật)**: $21,806$ ($7.47\%$)
6. **`food` (Thực phẩm)**: $21,423$ ($7.34\%$)
7. **`sports` (Dụng cụ thể thao)**: $16,786$ ($5.75\%$)
8. **`indoor` (Vật dụng trong nhà)**: $15,795$ ($5.41\%$)
9. **`accessory` (Phụ kiện cá nhân)**: $15,483$ ($5.30\%$)
10. **`electronic` (Đồ điện tử)**: $9,822$ ($3.37\%$)
11. **`outdoor` (Vật thể ngoài trời)**: $9,710$ ($3.33\%$)
12. **`appliance` (Thiết bị gia dụng)**: $4,582$ ($1.57\%$)

### Phân bố và độ phủ danh mục

Trong 291.875 object instance của COCO val2014, person chiếm 30,20%, vehicle 11,16% và sports 5,75%. Đây là tỷ lệ instance, không phải tỷ lệ ảnh hoặc caption. Hình 4 vẽ Top 15 danh mục chi tiết; bottle đứng thứ năm. COCO cũng có laptop, tv, keyboard, cell phone và book.

Ontology 80 danh mục COCO không có nhãn riêng cho nội dung chữ, nhãn sản phẩm hay tên thuốc. Caption VizWiz thường nhắc screen, box, package, hand và label. Danh sách danh từ này phản ánh lời mô tả, chưa đo nhu cầu sử dụng hoặc tần suất nhiệm vụ đọc thuốc/hạn dùng.

### Giới hạn của CHAIR/POPE khi so sánh COCO với VizWiz:

Không nên đọc điểm CHAIR trên COCO và VizWiz như hai đại lượng hoàn toàn tương đương. CHAIR gốc chủ động giới hạn đối tượng vào **80 category của MS-COCO segmentation**. Nhiều danh từ phổ biến trong caption VizWiz như `screen`, `box`, `package`, `hand` và `label` không phải category COCO; chỉ một phần như `bottle`, `laptop/computer` hoặc `table` có thể ánh xạ gần đúng. Vì vậy CHAIR có thể **không đếm** hallucination ngoài ontology và làm điểm VizWiz trông tốt giả tạo.

POPE giảm phụ thuộc vào phong cách caption bằng câu hỏi yes/no về sự tồn tại của vật thể, nhưng vẫn chỉ hợp lệ khi tập positive object và negative probe được xây dựng từ annotation phù hợp với VizWiz. Tái sử dụng nguyên object pool COCO cho VizWiz vẫn kế thừa ontology bias. Báo cáo chính nên:

1. Giữ CHAIR/POPE-COCO để so sánh với literature, nhưng gắn nhãn rõ là **COCO-ontology score**.
2. Báo cáo thêm metric open-vocabulary/domain-specific cho VizWiz, với ontology mở rộng gồm scene text, bao bì, nhãn, thuốc, màn hình và bộ phận cơ thể.
3. Không kết luận mô hình hallucinate ít hơn trên VizWiz chỉ từ CHAIR thấp hơn.

---

## PHẦN 5: TỶ LỆ ẢNH KHÓ & CHẤT LƯỢNG KÉM Ở VIZWIZ

Điểm khác biệt căn bản nhất giữa COCO và VizWiz nằm ở **chất lượng thu nhận hình ảnh (Image Acquisition Quality)**:

### 1. Phân bố Chú thích Báo lỗi Cấp độ Ảnh:
Trong VizWiz, khi một hình ảnh bị mờ, tối hoặc sai góc độ nghiêm trọng đến mức không thể nhận biết nội dung, người gán nhãn sẽ chọn câu trả lời mẫu chuẩn (precanned caption):
> *"Quality issues are too severe to recognize visual content."*

Khảo sát mức độ đồng thuận của 5 annotators trên 7.750 ảnh VizWiz-Captions:

| Caption từ chối / ảnh | Số lượng ảnh | Tỷ lệ (%) | Đồng thuận annotator |
| :--- | :---: | :---: | :--- |
| **0/5** | **5,222** | **67.38%** | Cả năm annotator mô tả nội dung |
| **1/5** | 1,297 | 16.74% | Một annotator từ chối, bốn annotator mô tả |
| **2/5** | 480 | 6.19% | Hai annotator từ chối, ba annotator mô tả |
| **3/5** | 286 | 3.69% | Ba annotator từ chối, hai annotator mô tả |
| **4/5** | 273 | 3.52% | Bốn annotator từ chối, một annotator mô tả |
| **5/5** | **192** | **2.48%** | Cả năm annotator dùng caption từ chối |
| **Ít nhất 1/5 dùng caption từ chối** | **2,528** | **32.62%** | Ngưỡng nhạy, gồm cả bất đồng nhẹ |
| **Đa số ≥3/5 dùng caption từ chối** | **751** | **9.69%** | Tín hiệu đồng thuận mạnh hơn |

### 2. Sự Hiện diện của Chữ viết (Text / OCR Relevance):
Khảo sát sự đồng thuận của 5 annotators về việc có xuất hiện văn bản trong ảnh hay không:
- **0 người thấy chữ**: 2,037 ảnh ($26.28\%$)
- **Có từ 1 đến 4 người thấy chữ**: 2,063 ảnh ($26.62\%$)
- **Cả 5 người cùng thấy chữ rõ ràng**: **3,650 ảnh ($47.10\%$)**
- **Tổng số ảnh có chữ ($\ge 1$ annotator)**: **5,713 ảnh ($73.72\%$)**
- **Đồng thuận đa số có chữ ($\ge 3/5$ annotators)**: **5,018 ảnh ($64.75\%$)**

Scene text phổ biến trong VizWiz, vì vậy OCR là năng lực quan trọng. Tuy nhiên, cờ `text_detected` không cho biết mục đích chụp và không chứng minh rằng phần lớn ảnh được chụp để đọc nhãn thuốc.

### 3. Đo lường Khách quan bằng Thuật toán Xử lý Ảnh (Image Processing Metrics):
- **Phương sai Laplacian sau chuẩn hóa**: áp dụng EXIF orientation, giữ tỷ lệ khung hình và resize cạnh ngắn về 480 px trên mẫu ngẫu nhiên 1.000 ảnh/tập (`seed=42`).
  - **MS-COCO val2014**: median **1197.90**.
  - **VizWiz val**: median **149.19**.
  - Chênh lệch phản ánh năng lượng biên/texture thấp hơn, nhưng không xác định riêng motion blur; defocus, JPEG và cảnh ít texture cũng ảnh hưởng chỉ số.
- **Độ sáng**: median COCO **113.15**, median VizWiz **107.95**; hai trung vị gần nhau, nên độ sáng trung bình không phải yếu tố phân biệt rõ trong mẫu này.

Trung vị Laplacian trên mỗi mẫu 1.000 ảnh có bootstrap percentile 95% CI: COCO [1084.24, 1298.65] và VizWiz [141.69, 162.38] (3.000 lần lấy mẫu lại, seed=42).

Kiểm tra nội bộ trên 1.000 ảnh VizWiz cho thấy chỉ số giảm theo số annotator dùng caption từ chối:

| Caption từ chối | Số ảnh mẫu | Trung vị Laplacian | Bootstrap 95% CI |
| :---: | ---: | ---: | :---: |
| 0/5 | 681 | 205.90 | [180.16, 222.76] |
| 1/5 | 149 | 108.31 | [91.35, 123.94] |
| 2/5 | 73 | 82.63 | [51.06, 102.19] |
| 3/5 | 42 | 80.47 | [43.51, 106.31] |
| 4/5 | 29 | 25.96 | [18.01, 66.00] |
| 5/5 | 26 | 23.49 | [8.37, 61.59] |

Xu hướng này hỗ trợ giá trị chỉ báo của Laplacian đối với khả năng nhận diện nội dung, nhưng không chứng minh mọi ảnh có chỉ số thấp đều bị mờ. Các nhóm 4/5 và 5/5 nhỏ, khoảng tin cậy rộng.

### Mờ/lệch tâm có tương quan với CHAIR/POPE hay không?

**Chưa thể trả lời bằng số liệu hiện có.** EDA mới có độ nét Laplacian và mô tả định tính về framing, trong khi repo chưa có caption/answer do cùng một mô hình sinh ở cấp ảnh và chưa có điểm CHAIR/POPE theo từng ảnh. Chênh lệch trung bình giữa hai dataset không phải là bằng chứng tương quan và không được diễn giải như quan hệ nhân quả.

Kiểm định tối thiểu cần thực hiện trên cùng bộ ảnh và cùng cấu hình decoding:

- $x_1=\log(1+\mathrm{Var}(\nabla^2 I))$ cho độ nét; $x_2$ là khoảng cách chuẩn hóa từ tâm bbox/mask của vật thể chính đến tâm ảnh; thêm độ sáng, số object, độ dài caption và dataset làm biến kiểm soát.
- $y$ gồm CHAIR-I/CHAIR-S ở cấp ảnh và POPE error/F1. Dùng Spearman $\rho$ kèm bootstrap 95% CI; với POPE error nhị phân, dùng logistic regression có interaction `dataset × blur` và `dataset × off-center`.
- Phân tích riêng COCO và VizWiz, đồng thời báo cáo trên ontology chung và ontology mở rộng. Nếu không, hiệu ứng “dataset” và giới hạn 80 category sẽ gây nhiễu kết luận.
- Chỉ kết luận có tương quan khi CI không cắt 0 và hiệu ứng còn tồn tại sau khi kiểm soát độ dài output/ontology coverage. Thực nghiệm làm mờ/crop có kiểm soát trên cùng ảnh sẽ mạnh hơn tương quan quan sát trong việc lập luận nguyên nhân.

---

## PHẦN 6: TỔNG HỢP KHOẢNG CÁCH MIỀN

| Chỉ số đã đo | COCO val2014 | VizWiz val |
| :--- | :---: | :---: |
| Ảnh ngang | 71,71% | 0,62% |
| Ảnh dọc | 23,70% | 99,23% |
| Median Laplacian, n=1.000/tập | 1.197,90 | 149,19 |
| Median độ sáng, n=1.000/tập | 113,15 | 107,95 |
| Ảnh có chữ theo đa số annotator | Không có nhãn tương ứng | 64,75% |
| Ảnh có caption từ chối theo đa số annotator | Không có nhãn tương ứng | 9,69% |

Vị trí chủ thể, tỷ lệ vùng bị crop và sai số CHAIR/POPE cấp ảnh chưa được đo. Do đó bảng này mô tả đầu vào dữ liệu; tác động của tiền xử lý và tương quan với hallucination là các giả thuyết cho thực nghiệm tiếp theo.

---

## PHẦN 7: BỘ MẪU ẢNH ĐỐI SÁNH TIÊU BIỂU (QUALITATIVE COMPARISON)

Để phục vụ thuyết trình và trình bày trực quan trong khóa luận, 8 mẫu ảnh tiêu biểu đã được trích xuất và lưu tại biểu đồ `07_qualitative_samples_grid.png`:

### Nhóm 1: Chuẩn mực Benchmark COCO (Curated Photography)
1. **Ảnh COCO mẫu #391895**:
   - *Hình ảnh*: Người đàn ông đội mũ bảo hiểm đỏ lái xe máy trên đường đất, nền núi mây hùng vĩ.
   - *Đặc điểm*: Ánh sáng hoàn hảo, vật thể cân đối giữa khung hình, độ sâu trường ảnh tốt.
   - *Caption mẫu*: *"A man with a red helmet on a moped on a dirt road."*
2. **Ảnh COCO mẫu #522418**:
   - *Hình ảnh*: Người phụ nữ đeo lưới trùm tóc đang cắt chiếc bánh kem trắng lớn.
   - *Đặc điểm*: Rõ nét từng chi tiết dao cắt, bố cục trong bếp chuẩn mực.
   - *Caption mẫu*: *"A woman wearing a hair net cutting a large sheet cake."*
3. **Ảnh COCO mẫu #184613**:
   - *Hình ảnh*: Cậu bé cầm ô hoa đứng vuốt ve sừng con bò Tây Tạng (yak).
   - *Đặc điểm*: Đầy đủ bối cảnh văn hóa, màu sắc sống động, độ tương phản sắc nét.
   - *Caption mẫu*: *"A young boy with an umbrella touching the horn of a yak."*
4. **Ảnh COCO mẫu #161222**:
   - *Hình ảnh*: Bàn ăn nhà hàng với nhiều đĩa burger, khoai chiên, vòng hành và ly rượu vang.
   - *Đặc điểm*: Ảnh và caption đã được đối chiếu trực tiếp theo `image_id = 161222` trong annotation COCO.
   - *Caption mẫu*: *"A table topped with plates of food and wine glasses."*

### Nhóm 2: Thực tế Đời thực VizWiz (In-The-Wild Assistive Reality)
1. **Ảnh VizWiz mẫu #00000001 (Handheld Medicine & OCR)**:
   - *Hình ảnh*: Bàn tay người khiếm thị đang cầm một chai thuốc nhỏ màu đen, nhãn chữ *"NIGHT TIME"*.
   - *Đặc điểm*: Ngón tay che một phần thân chai, góc chụp thẳng đứng, người dùng cần biết đây là thuốc gì.
   - *Caption mẫu*: *"A person is holding a small black bottle of NIGHT TIME medicine."*
2. **Ảnh VizWiz mẫu #00000000 (Digital Screen & Assistive Prompt)**:
   - *Hình ảnh*: Màn hình laptop chụp nghiêng, hiển thị hộp thoại thông báo sửa lỗi hệ thống máy tính.
   - *Đặc điểm*: Phản chiếu bóng đèn trên màn hình LCD, chữ nhỏ, góc chụp chéo từ trên xuống.
   - *Caption mẫu*: *"A partial computer screen showing a repair prompt popup."*
3. **Ảnh VizWiz mẫu #00000013 (Severe Motion Blur & Hand Shake)**:
   - *Hình ảnh*: Một vật thể không thể xác định trên mặt bàn, vệt mờ chuyển động kéo dài khắp khung hình.
   - *Đặc điểm*: Camera di chuyển khi bấm chụp, mất hoàn toàn biên cạnh nhận dạng vật thể.
   - *Caption mẫu*: *"A blurry photo of an unrecognizable object on a table."*
4. **Ảnh VizWiz mẫu #00000006 (Severe Glare / Obscured Scene - 100% Precanned)**:
   - *Hình ảnh*: Ảnh vẫn sáng nhưng mờ sương, tương phản thấp và bị lóa/phản chiếu mạnh ở vùng trung tâm; chỉ thấy các hình khối mơ hồ.
   - *Đặc điểm*: Cả 5 người gán nhãn đều dùng caption từ chối chuẩn hóa vì chất lượng ảnh không đủ để nhận diện nội dung chắc chắn; đây không phải ảnh đen hoặc cảm biến bị che kín.
   - *Caption mẫu*: *"Quality issues are too severe to recognize visual content."* (Lặp lại 5/5 lần).

---

## PHẦN 8: KHUYẾN NGHỊ THIẾT KẾ MÔ HÌNH CHO KHÓA LUẬN

Từ các kết quả EDA trên, khóa luận tốt nghiệp nên đề xuất các giải pháp kỹ thuật cụ thể sau:

### 1. Cơ chế Từ chối Thông minh (Smart Quality-Aware Refusal Mechanism):
- Không nên ép mô hình phải sinh caption miêu tả đối tượng khi gặp ảnh suy giảm nặng.
- Cần xây dựng một **bộ phân loại chất lượng ảnh 2 giai đoạn (Two-Stage Pipeline)** hoặc huấn luyện mô hình sinh chuỗi token đặc biệt:
  - Nếu $\text{P}(\text{Quality Issue}) > \tau$: Sinh cảnh báo lỗi và hướng dẫn người dùng chụp lại.
  - Nếu ảnh đạt yêu cầu: Chuyển sang module sinh miêu tả chi tiết.

### 2. Tích hợp Module Thị giác - Chữ viết (OCR-Grounded Vision Architecture):
- Vì hơn $73\%$ ảnh có chứa văn bản, việc chỉ dùng CLIP Visual Encoder là không đủ.
- Kiến trúc mô hình nên kết hợp đặc trưng trích xuất từ **OCR Engine (như PaddleOCR, EasyOCR hoặc TrOCR)** và đưa các chuỗi text phát hiện được vào làm Prompt/Prefix cho bộ giải mã ngôn ngữ (Language Decoder).

### 3. Chiến lược Data Augmentation Mô phỏng Rung mờ & Góc chụp Egocentric:
- Khi fine-tune mô hình nền tảng (Foundation Models), cần bổ sung các phép tăng cường dữ liệu:
  - **Motion Blur & Defocus Blur** ngẫu nhiên.
  - **Random Perspective & Extreme Cropping** mô phỏng việc căn lệch khung hình.
  - **Synthetic Hand Occlusion** (ghép các mặt nạ bàn tay che khuất vật thể).
  - Giữ nguyên tỷ lệ ảnh dọc hoặc sử dụng kỹ thuật **Padding thông minh** thay vì bóp méo tỷ lệ (Squishing).

---

## TÀI NGUYÊN ĐÃ XUẤT XƯỞNG (EXPORTED ARTIFACTS)

Toàn bộ biểu đồ độ phân giải cao (300 DPI) và dữ liệu bảng biểu đã được lưu trữ hoàn chỉnh tại:
- **Thư mục Đồ thị**: `eda_results/figures/`
  1. `02_resolution_aspect_ratio.png`: Phân bố độ phân giải và tỷ lệ khung hình.
  2. `03_caption_length_and_tokens.png`: Phân bố độ dài từ và từ vựng đặc trưng.
  3. `04_coco_category_distribution.png`: Phân bố đối tượng COCO và sự áp đảo của con người.
  4. `05_vizwiz_quality_and_text_issues.png`: Tỷ lệ ảnh suy giảm chất lượng, chữ viết và độ sắc nét quang học.
  5. `07_qualitative_samples_grid.png`: Lưới ảnh đối sánh trực quan các trường hợp tiêu biểu.
- **Thư mục Bảng số liệu CSV**: `eda_results/tables/`
- **Kiểm tra Laplacian theo đồng thuận precanned**: `eda_results/tables/laplacian_by_precanned.csv`.
- **File Dữ liệu Thống kê Đầy đủ (JSON)**: `eda_results/eda_summary_metrics.json`
- **Script Thực thi Tái lập (Reproducible Python Script)**: `eda_coco_vs_vizwiz.py`
