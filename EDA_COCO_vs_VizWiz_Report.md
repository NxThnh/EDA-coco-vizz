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
6. Caption VizWiz nhắc nhiều đến computer (95,58), screen (92,96), bottle (82,34), box (75,52) và label (40,10) trên 1.000 caption hợp lệ. Khi gộp man/men/woman/people/person, COCO có 314,92 lượt nhắc/1.000 caption so với 73,98 ở VizWiz; tỷ lệ ảnh có ít nhất một caption hợp lệ nhắc nhóm này là 43,77% so với 17,42%.

---

## PHẦN 1: PHẠM VI ĐỐI SÁNH DỮ LIỆU

Phần mở đầu mô tả **hai miền dữ liệu** theo đúng phạm vi ảnh và chú thích thực tế được khảo sát:

| Miền dữ liệu | Phạm vi dùng trong EDA | Vai trò trong đối sánh | Đặc trưng thu nhận |
| :--- | :---: | :--- | :--- |
| **MS-COCO val2014** | 40.504 ảnh có caption và instance annotation | Miền ảnh đối chiếu | Ảnh trong tập validation COCO |
| **VizWiz-Captions** | 7.750 ảnh, 38.750 caption gốc | Miền hỗ trợ thị giác đời thực | Ảnh smartphone do người khiếm thị chụp |
| **COCO-500 (Bảng 2 & 4)** | 500 ảnh lấy ngẫu nhiên, `seed=42` | Mẫu mô tả bổ sung; kiểm tra độ biến thiên qua 200 seed | Danh sách ID và kết quả từng seed được lưu để tái lập |

Đơn vị phân tích được ghi rõ ở từng bảng: thống kê ảnh dùng số ảnh; phân tích ngôn ngữ dùng caption mô tả hợp lệ; phân bố đối tượng dùng số object instance. Cách trình bày này tránh cộng gộp những tập không có cùng loại annotation.

---

## PHẦN 2: ĐỘ PHÂN GIẢI & TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO)

Khảo sát gồm 40.504 ảnh MS-COCO val2014 và 7.750 ảnh VizWiz-Captions val. Kích thước VizWiz được đo từ file sau EXIF transpose (493 ảnh có thẻ xoay 90°); kích thước COCO lấy từ annotation. Trước EXIF, tỷ lệ ảnh dọc VizWiz là 92,86%; sau EXIF là 99,23%.

Để đối chiếu theo yêu cầu với đúng 500 ảnh COCO, một mẫu ngẫu nhiên cố định (`seed=42`) cũng được chạy lại riêng. Mẫu cho kết quả: kích thước trung bình $576,9\times481,8$ px, $73,60\%$ ảnh ngang, $22,80\%$ ảnh dọc và $3,60\%$ ảnh vuông. Ảnh chụp bảng: [`table2_coco500_resolution.png`](eda_results/coco500/table2_coco500_resolution.png).

Trong 200 mẫu ngẫu nhiên độc lập, mỗi mẫu 500 ảnh, tỷ lệ ảnh ngang có trung vị 71,90% và khoảng phân vị 2,5–97,5% là 67,80–75,81%. Seed 42 nằm trong khoảng này. Đây là phân tích độ nhạy theo cách chọn mẫu, không tạo ra một tập COCO có cùng quy mô với VizWiz.

| Chỉ số hình ảnh | MS-COCO val2014 ($N=40.504$) | VizWiz-Captions ($N=7.750$) | Ý nghĩa trong Khóa luận |
| :--- | :---: | :---: | :--- |
| **Chiều rộng trung bình (Mean Width)** | $576.5 \pm 91.9\text{ px}$ | **$1032.1 \pm 334.1\text{ px}$** | Độ rộng file VizWiz phân tán hơn trong tập khảo sát |
| **Trung vị Chiều rộng (Median Width)** | $640.0\text{ px}$ | $968.0\text{ px}$ | COCO val2014 có chiều rộng tối đa 640 px |
| **Chiều rộng [Min, Max]** | $[120, 640]\text{ px}$ | **$[23, 2448]\text{ px}$** | VizWiz có ảnh siêu nhỏ ($23\text{px}$) và siêu lớn |
| **Chiều cao trung bình (Mean Height)** | $485.1 \pm 97.8\text{ px}$ | **$1374.7 \pm 445.2\text{ px}$** | Chiều dọc VizWiz vượt trội chiều ngang |
| **Trung vị Chiều cao (Median Height)** | $480.0\text{ px}$ | $1296.0\text{ px}$ | File VizWiz có trung vị chiều cao lớn hơn |
| **Chiều cao [Min, Max]** | $[111, 640]\text{ px}$ | $[25, 3264]\text{ px}$ | Chênh lệch độ phân giải cực đoan |
| **Tỷ lệ khung hình trung bình (AR Mean)** | **$1.25$** (Ngang) | **$0.75$** (Dọc) | Phân bố hướng ảnh khác nhau |
| **Trung vị AR (Median AR)** | $1.33$ ($4:3$) | $0.75$ ($3:4$) | Tỷ lệ khung hình trung vị của file ảnh |
| **Tỷ lệ ảnh Ngang (Landscape: $AR > 1.05$)** | **$71.71\%$** ($29,044$ ảnh) | **$0.62\%$** ($48$ ảnh) | Ảnh ngang phổ biến hơn trong COCO |
| **Tỷ lệ ảnh Dọc (Portrait: $AR < 0.95$)** | **$23.70\%$** ($9,601$ ảnh) | **$99.23\%$** ($7,690$ ảnh) | Ảnh dọc chiếm đa số VizWiz sau EXIF transpose |
| **Tỷ lệ ảnh Vuông ($0.95 \le AR \le 1.05$)** | $4.59\%$ ($1,859$ ảnh) | $0.15\%$ ($12$ ảnh) | Rất ít ảnh vuông tự nhiên |
| **Số Megapixel trung bình (Mean MP)** | $0.276\text{ MP}$ | **$1.566\text{ MP}$** | Số pixel trong file VizWiz lớn hơn; chưa đo dung lượng file |

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

VizWiz có 5.167/38.750 caption ($13,33\%$) là cùng một câu placeholder báo ảnh không nhận diện được. Ngoài ra, 964 caption có cờ `is_rejected`, trong đó 526 caption đồng thời là placeholder; vì vậy có thêm 438 caption cần loại. Toàn bộ thống kê ngôn ngữ, token và thực thể chỉ dùng caption không mang cờ `is_precanned` hoặc `is_rejected`: $202.654$ caption COCO và $33.145$ caption VizWiz.

| Chỉ số Ngôn ngữ | COCO | VizWiz (đã bỏ placeholder và caption bị từ chối) |
| :--- | :---: | :---: |
| **Số lượng Caption ($N$)** | $202.654$ | $33.145$ |
| **Độ dài từ trung bình (Word Mean)** | $10,45 \pm 2,38$ | **$12,05 \pm 4,36$** |
| **Trung vị số từ (Word Median)** | $10,0$ | $11,0$ |
| **Khoảng số từ [Min, Max]** | $[6, 50]$ | **$[6, 155]$** |
| **Độ dài ký tự trung bình (Char Mean)**| $52,29$ | $62,52$ |
| **Tổng số token từ** | $2.122.711$ | $402.826$ |
| **Quy mô từ vựng** | $17.348$ | $10.851$ |
| **Type-Token Ratio (TTR)** | $0,0082$ | **$0,0269$** |
| **MATTR (cửa sổ 1.000 token)** | $0,3209$ | $0,3196$ |

Giá trị lớn nhất 155 từ là một caption chép lại đoạn chữ dài trên màn hình trong ảnh có image_id 24317; nó không đại diện cho độ dài caption điển hình. Giá trị nhỏ nhất 6 từ ở cả hai tập là câu mô tả ngắn hợp lệ theo cách đếm bằng khoảng trắng, không phải ảnh bị thiếu annotation.

Hình 3A phóng to vùng 3–30 từ để thấy phân bố chính; Hình 3B dùng ECDF trên toàn khoảng đến 155 từ để hiển thị cả ngoại lai.

### Các Phát hiện Ngôn ngữ học Nổi bật:
1. **Độ dài câu và đuôi phân phối dài (Long-tail)**:
   - Câu chú thích COCO có độ dài tập trung hẹp xung quanh 10 từ (độ lệch chuẩn chỉ $2.38$).
   - Câu chú thích VizWiz hợp lệ có độ dài trung bình lớn hơn ($12.05$ từ) và kéo dài tới **$155$ từ**. Giá trị 155 từ là ngoại lai độ dài; không thể suy ra nguyên nhân chỉ từ thống kê tổng hợp.
2. **Độ đa dạng từ vựng trên cùng quy mô**:
   - TTR toàn kho không được dùng để xếp hạng vì COCO có 2,12 triệu token còn VizWiz có 0,40 triệu. MATTR cửa sổ 1.000 token gần nhau: **0.3209 (COCO)** và **0.3196 (VizWiz)**.
3. **Top-15 danh từ tính từ dữ liệu**:
   - NLTK POS tagger chỉ giữ danh từ và WordNet lemmatizer chuẩn hóa số ít. Bảng báo cả count lẫn số lượt nhắc trên 1.000 caption.
   - VizWiz nhắc nhiều đến `computer`, `screen`, `bottle`, `box` và `label`; COCO nổi bật với `man`, `people` và `woman`. `person` riêng lẻ đạt 50,87/1.000 caption ở VizWiz và 41,75 ở COCO. CSV nguồn: [`caption_noun_entities_top15.csv`](eda_results/tables/caption_noun_entities_top15.csv).
   - Gộp `man/men/woman/people/person` sau POS tagging và lemmatization, COCO có **314,92 lượt nhắc/1.000 caption**, VizWiz **73,98**. Tỷ lệ caption có ít nhất một từ trong nhóm là **29,72%** so với **7,19%**; tỷ lệ ảnh có ít nhất một caption hợp lệ nhắc nhóm là **43,77%** so với **17,42%**. Đây là tần suất trong văn bản, không phải số người hiện diện thực tế trong ảnh. [Bảng nhóm từ chỉ người](eda_results/tables/person_reference_summary.csv).

---

## PHẦN 4: PHÂN BỐ DANH MỤC ĐỐI TƯỢNG COCO (OBJECT CATEGORY BIAS)

Tập dữ liệu COCO được xây dựng dựa trên hệ thống phân cấp 80 danh mục đối tượng thuộc 12 siêu danh mục (Supercategories). Phân tích $291.875$ đối tượng được gán nhãn trong phạm vi khảo sát cho thấy:

Trên mẫu COCO-500 seed 42 có 3.719 object instance. Ba siêu danh mục lớn nhất là `person` (29,52%), `vehicle` (12,34%) và `kitchen` (10,24%). Qua 200 seed, tỷ lệ `person` có trung vị 30,03% và khoảng phân vị 2,5–97,5% là 27,06–33,49%; `vehicle` là 11,18% [8,72–14,23%], `kitchen` là 9,99% [7,10–13,32%]. Các nhóm dẫn đầu tương đối ổn định, nhưng giá trị của một mẫu 500 ảnh có biến thiên đáng kể. [Bảng mẫu seed 42](eda_results/coco500/table4_coco500_supercategories.png) và [bảng độ nhạy](eda_results/coco500/coco500_robustness_summary.csv).

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
| **0/5** | **5,222** | **67.38%** | Không có caption mẫu từ chối |
| **1/5** | 1,297 | 16.74% | Một caption mẫu từ chối |
| **2/5** | 480 | 6.19% | Hai caption mẫu từ chối |
| **3/5** | 286 | 3.69% | Ba caption mẫu từ chối |
| **4/5** | 273 | 3.52% | Bốn caption mẫu từ chối |
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

Phép resize cạnh ngắn lên 480 px đã phóng lớn **610/1.000 ảnh COCO** và **90/1.000 ảnh VizWiz**, nên chênh lệch toàn mẫu chịu ảnh hưởng của quy tắc chuẩn hóa. Khi chỉ giữ ảnh có cạnh ngắn gốc ≥480 px, median là **1.605,96** ở COCO (n=390; bootstrap 95% CI [1.441,75; 1.894,75]) và **158,33** ở VizWiz (n=910; [147,12; 176,19]). Xu hướng thấp hơn ở VizWiz vẫn còn trong hai tập con này, nhưng đây là các tập con khác nhau về kích thước và nội dung; không thể tách riêng hiệu ứng chất lượng ảnh. [Bảng độ nhạy](eda_results/tables/laplacian_resize_sensitivity.csv).

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

Tám ảnh được chọn cố định để minh họa các kiểu nội dung và mức đồng thuận caption. Đây không phải mẫu ngẫu nhiên và không dùng để ước lượng tỷ lệ toàn tập. Hình 7 hiển thị nguyên văn một caption trong annotation cho từng ảnh; [manifest](eda_results/tables/qualitative_samples_manifest.csv) ghi file, image ID, annotation ID và số caption từ chối trên ảnh.

| Tập | Ảnh | Caption nguyên văn trong annotation | Caption từ chối |
| :--- | :--- | :--- | :---: |
| COCO | 391895 | “A man with a red helmet on a small moped on a dirt road.” | — |
| COCO | 522418 | “A woman wearing a hair net cutting a large sheet cake.” | — |
| COCO | 184613 | “A child holding a flowered umbrella and petting a yak.” | — |
| COCO | 161222 | “A dinner table with three plates of gourmet hamburgers.” | — |
| VizWiz | 00000001 | “a person holding a small black bottle of NIGHT TIME” | 0/5 |
| VizWiz | 00000000 | “A computer screen shows a repair prompt on the screen.” | 0/5 |
| VizWiz | 00000013 | “Quality issues are too severe to recognize visual content.” | 4/5 |
| VizWiz | 00000006 | “Quality issues are too severe to recognize visual content.” | 5/5 |

Hai ảnh VizWiz cuối minh họa sự khác nhau giữa đồng thuận 4/5 và 5/5; annotation không xác định riêng nguyên nhân suy giảm như rung tay hay lóa.

---

## PHẦN 8: GIẢ THUYẾT VÀ THỰC NGHIỆM TIẾP THEO

EDA đo khác biệt của dữ liệu đầu vào; chưa đo chất lượng caption/answer do cùng một mô hình sinh ra. Ba can thiệp sau cần được đánh giá bằng ablation trên cùng tập ảnh và cùng mô hình:

1. **Ngưỡng từ chối:** So sánh mô hình gốc với cơ chế từ chối theo xác suất ảnh khó nhận diện. Báo cáo precision/recall của quyết định từ chối trên các mức đồng thuận 0/5–5/5, cùng tỷ lệ câu trả lời có thông tin sai.
2. **OCR bổ trợ:** So sánh câu trả lời khi bật/tắt OCR trên ảnh có và không có nhãn chữ. Đo độ chính xác thông tin chữ và hallucination; tỷ lệ ảnh có chữ tự nó chưa chứng minh encoder thị giác hiện tại không đủ.
3. **Tiền xử lý và tăng cường dữ liệu:** Với đúng bộ xử lý ảnh của các mô hình được đánh giá, so sánh resize/crop mặc định, giữ tỷ lệ với padding, và các phép làm mờ/che khuất có kiểm soát. Đo nội dung bị cắt và kết quả đầu ra; hiệu quả của Smart Letterbox Padding vẫn là giả thuyết.

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
- **Độ nhạy theo resize và từ chỉ người**: `eda_results/tables/laplacian_resize_sensitivity.csv`, `eda_results/tables/person_reference_summary.csv`.
- **Chỉ số từng ảnh trong mẫu đo**: `eda_results/tables/image_metrics_sample.csv`.
- **Manifest ví dụ định tính và độ nhạy COCO-500**: `eda_results/tables/qualitative_samples_manifest.csv`, `eda_results/coco500/coco500_robustness_summary.csv`.
- **File Dữ liệu Thống kê Đầy đủ (JSON)**: `eda_results/eda_summary_metrics.json`
- **Script Thực thi Tái lập (Reproducible Python Script)**: `eda_coco_vs_vizwiz.py`
