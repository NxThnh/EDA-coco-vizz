# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA): ĐỐI SÁNH TẬP DỮ LIỆU MS-COCO VÀ VIZWIZ-CAPTIONS
## Phục vụ Khóa Luận Tốt Nghiệp: Nghiên cứu Domain Gap và Thách thức Thị giác Đời thực

---

## TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo so sánh MS-COCO val2014 (40.504 ảnh) và VizWiz-Captions val (7.750 ảnh). Các kết quả dưới đây mô tả khác biệt dữ liệu đầu vào; báo cáo chưa đo sai số của mô hình hoặc quan hệ nhân quả với hallucination.

Checkpoint đích là **LLaVA-1.5-7B**. Phần 8 đối chiếu trực tiếp file instruction tuning 665K, nhãn chất lượng VizWiz, độ đầy đủ nhãn object lấy từ caption, hình học tiền xử lý, CLIP embedding, tokenizer và manifest đánh giá. Chưa có output sinh bởi LLaVA trong repo; vì vậy báo cáo chưa kết luận về hallucination của mô hình.

### Các phát hiện định lượng cốt lõi

1. **Nguy cơ contamination phải theo đúng nguồn huấn luyện.** Có **35.504/40.504 ảnh COCO val2014** nằm trong COCO train2017, nhưng chỉ **5.741/40.504** có đường dẫn COCO trong [file instruction tuning LLaVA-1.5 665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json). Manifest COCO 500 ảnh gồm **70 ảnh trong mix và 430 ảnh ngoài mix**. Đây chưa phải kiểm toán toàn bộ pretraining hay ảnh trùng giữa các nguồn.
2. **Hình học xử lý ảnh khác ở mức thu nhỏ.** Với mode pad 336×336, trung vị diện tích viền là **26,72% COCO** và **25,00% VizWiz**; cạnh dài trung vị lần lượt **640** và **1.296 px**, tương đương thu về 336 khoảng **1,90** và **3,86 lần**. Nguy cơ mất chữ nhỏ do downsample đáng kiểm tra hơn chênh lệch diện tích padding.
3. **Ảnh khó có nhãn lỗi trực tiếp.** [VizWiz-QualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/) khớp đủ 7.750 ảnh val: **29,91%** có đa số gán blur và **36,45%** poor framing. Laplacian toàn val là **1.140,08 COCO** so với **152,16 VizWiz**, nhưng vẫn chỉ là chỉ báo biên/texture.
4. **Độ phủ metric theo từ chưa đủ để làm ground truth.** **52,666%** caption VizWiz hợp lệ nhắc từ COCO-80. Trên COCO, hợp nhất cả 5 caption chỉ nhắc **55,26%** số lượt lớp object được gán trong instance annotation (mỗi lớp đếm một lần/ảnh). Vì thế object chỉ suy từ caption sẽ bỏ sót nhãn hiện diện và có thể làm sai CHAIR.
5. **Khoảng cách miền có tín hiệu khác ở ảnh và chữ.** JS unigram **0,266183 bit** so với mốc chia đôi COCO **0,018418 bit**; Jaccard caption **0,1989 COCO** và **0,1633 VizWiz**. CLIP cho thấy cosine nội miền VizWiz cao hơn COCO, còn khoảng cách giữa cặp ảnh–caption đúng và cặp hoán vị hẹp hơn; Phần 8 kiểm tra tác động của padding và phân tầng theo cờ chữ/từ chối.

---

## PHẦN 1: PHẠM VI ĐỐI SÁNH DỮ LIỆU

Phần mở đầu mô tả **hai miền dữ liệu** theo đúng phạm vi ảnh và chú thích thực tế được khảo sát. Nguồn gốc và mục tiêu xây dựng hai tập được mô tả trong công trình [MS-COCO](https://doi.org/10.1007/978-3-319-10602-1_48) và [VizWiz-Captions](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2783_ECCV_2020_paper.php):

| Miền dữ liệu | Phạm vi dùng trong EDA | Vai trò trong đối sánh | Đặc trưng thu nhận |
| :--- | :---: | :--- | :--- |
| **MS-COCO val2014** | 40.504 ảnh có caption và instance annotation | Miền ảnh đối chiếu | Ảnh trong tập validation COCO |
| **VizWiz-Captions** | 7.750 ảnh, 38.750 caption gốc | Miền hỗ trợ thị giác đời thực | Ảnh smartphone do người khiếm thị chụp |

Đơn vị phân tích được ghi rõ ở từng bảng: thống kê ảnh dùng số ảnh; phân tích ngôn ngữ chính dùng **một caption hợp lệ có annotation ID thấp nhất mỗi ảnh**; Jaccard và đồng thuận dùng mọi caption hợp lệ; phân bố đối tượng dùng số object instance.

### Kiểm tra dữ liệu trước khi so VLM

Ảnh val của cả hai tập đã được giải mã đầy đủ, không thiếu hoặc rỗng. Có **49 nhóm ảnh VizWiz trùng byte** (100 ảnh). Trong manifest đánh giá 500 ảnh VizWiz không có cặp trùng byte nội bộ; 8 ảnh eval có bản sao nằm ở val ngoài eval, nên `prepare_data.py` giữ eval và loại **9 bản sao** khỏi dev. [Danh sách loại trừ](data_prepared/exact_duplicate_dev_exclusions.csv).

Tỷ lệ giao nhau COCO val2014/train2017 (**87,656%**) không phải tỷ lệ mô hình LLaVA đã thấy. Đối chiếu file instruction tuning 665K chính thức cho **5.741/40.504 ảnh (14,174%)** có đường dẫn COCO trùng ID; **70/500** ảnh trong manifest thuộc nhóm đó. Phần 8.1 ghi phương pháp, hash nguồn và giới hạn của phép đối chiếu. Báo cáo sẽ giữ toàn bộ 500 ảnh và phân tích riêng 70 ảnh trong mix so với 430 ảnh ngoài mix; tập ngoài mix chỉ “sạch” theo file mix, không được gọi là sạch toàn bộ pretraining.

---

## PHẦN 2: ĐỘ PHÂN GIẢI & TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO)

Khảo sát gồm 40.504 ảnh MS-COCO val2014 và 7.750 ảnh VizWiz-Captions val. Kích thước VizWiz được đo từ file sau EXIF transpose (493 ảnh có thẻ xoay 90°); kích thước COCO lấy từ annotation. Trước EXIF, tỷ lệ ảnh dọc VizWiz là 92,86%; sau EXIF là 99,23%.

Các số pixel dưới đây mô tả **phiên bản file/annotation được dùng trong EDA**. COCO val2014 trong bộ khảo sát bị giới hạn cạnh tối đa ở 640 px, còn VizWiz được đo từ file ảnh phân phối. Vì vậy chênh lệch số pixel có thể chịu ảnh hưởng của cách chuẩn bị/phát hành dữ liệu; không suy ra độ phân giải cảm biến, độ chi tiết ảnh gốc hay chất lượng thu nhận chỉ từ bảng này.

Các thống kê mô tả trong phần này dùng toàn bộ tập val. Bộ 500 ảnh COCO duy nhất dùng cho đánh giá mô hình được cố định ở Phần 8.4.

| Chỉ số hình ảnh | MS-COCO val2014 ($N=40.504$) | VizWiz-Captions ($N=7.750$) | Ý nghĩa trong Khóa luận |
| :--- | :---: | :---: | :--- |
| **Chiều rộng trung bình (Mean Width)** | $576.5 \pm 91.9\text{ px}$ | **$1032.1 \pm 334.1\text{ px}$** | Độ rộng file VizWiz phân tán hơn trong tập khảo sát |
| **Trung vị Chiều rộng (Median Width)** | $640.0\text{ px}$ | $968.0\text{ px}$ | COCO val2014 có chiều rộng tối đa 640 px |
| **Chiều rộng [Min, Max]** | $[120, 640]\text{ px}$ | **$[23, 2448]\text{ px}$** | VizWiz có ảnh siêu nhỏ ($23\text{px}$) và siêu lớn |
| **Chiều cao trung bình (Mean Height)** | $485.1 \pm 97.8\text{ px}$ | **$1374.7 \pm 445.2\text{ px}$** | Chiều dọc VizWiz vượt trội chiều ngang |
| **Trung vị Chiều cao (Median Height)** | $480.0\text{ px}$ | $1296.0\text{ px}$ | File VizWiz có trung vị chiều cao lớn hơn |
| **Chiều cao [Min, Max]** | $[111, 640]\text{ px}$ | $[25, 3264]\text{ px}$ | Chênh lệch kích thước file ở hai bộ khảo sát |
| **Tỷ lệ khung hình trung bình (AR Mean)** | **$1.25$** (Ngang) | **$0.75$** (Dọc) | Phân bố hướng ảnh khác nhau |
| **Trung vị AR (Median AR)** | $1.33$ ($4:3$) | $0.75$ ($3:4$) | Tỷ lệ khung hình trung vị của file ảnh |
| **Tỷ lệ ảnh Ngang (Landscape: $AR > 1.05$)** | **$71.71\%$** ($29,044$ ảnh) | **$0.62\%$** ($48$ ảnh) | Ảnh ngang phổ biến hơn trong COCO |
| **Tỷ lệ ảnh Dọc (Portrait: $AR < 0.95$)** | **$23.70\%$** ($9,601$ ảnh) | **$99.23\%$** ($7,690$ ảnh) | Ảnh dọc chiếm đa số VizWiz sau EXIF transpose |
| **Tỷ lệ ảnh Vuông ($0.95 \le AR \le 1.05$)** | $4.59\%$ ($1,859$ ảnh) | $0.15\%$ ($12$ ảnh) | Rất ít ảnh vuông tự nhiên |
| **Số Megapixel trung bình (Mean MP)** | $0.276\text{ MP}$ | **$1.566\text{ MP}$** | Số pixel trong file VizWiz lớn hơn; chưa đo dung lượng file |

Ngoại lai 23 px là chiều rộng của ảnh VizWiz_val_00001384.jpg (23 × 30 px sau EXIF). Đây là ảnh cực nhỏ trong file nguồn, không đại diện cho độ phân giải thông thường của VizWiz. Các giá trị min/max được giữ để mô tả toàn bộ phân bố.

Trong Hình 2C2, nhiều ảnh có cùng kích thước file và chồng lên một tọa độ. Mỗi bong bóng biểu diễn một kích thước khác nhau; diện tích bong bóng tăng theo số ảnh có kích thước đó. Toàn bộ 7.750 ảnh VizWiz được tính.

### Hàm ý cho tiền xử lý ảnh

Hai tập khác nhau rõ về tỷ lệ khung hình, nhưng config LLaVA-1.5-7B được chọn dùng `image_aspect_ratio: pad`, nên mỗi ảnh có cùng 576 patch và diện tích padding trung vị gần nhau (26,72%/25,00%). Khác biệt hình học quan trọng hơn là cạnh dài file VizWiz bị thu về 336 px mạnh hơn COCO (trung vị 3,86/1,90 lần), có thể gây khó đọc chữ nhỏ. EDA chưa đo tỷ lệ chữ thực sự mất sau downsample; hiệu quả của thay đổi tiền xử lý cần kiểm chứng bằng output từ đúng checkpoint.

### Phân tích visual-token:

Visual-token không đồng nghĩa với pixel. Với một vision encoder chia ảnh thành patch kích thước $P\times P$, số token trước khi pooling xấp xỉ
$$N_{visual}=\left\lceil H'/P\right\rceil\left\lceil W'/P\right\rceil,$$
trong đó $H',W'$ là kích thước **sau tiền xử lý**. Vì vậy cần tách hai trường hợp:

- **Fixed square resize/crop**: COCO và VizWiz đều có cùng số token danh nghĩa (ví dụ ảnh $224\times224$, patch $16\times16$ cho 196 patch token), nhưng lượng thông tin được giữ lại phụ thuộc phép resize/crop. Số token bằng nhau không có nghĩa lượng thông tin hữu ích bằng nhau.
- **Aspect-ratio preserving / dynamic tiling**: ảnh dọc hoặc chứa chữ nhỏ có thể cần nhiều tile/token hơn nếu giữ độ chi tiết. Nếu ép cùng ngân sách token với COCO, mỗi token phải đại diện vùng nội dung lớn hơn hoặc hệ thống phải downsample nhiều hơn, làm giảm khả năng đọc nhãn/OCR.
- **Hệ quả cần báo cáo**: cùng với CHAIR/POPE, cần ghi rõ image size, patch size, chiến lược crop/pad/tile, số visual-token trung bình và tỷ lệ vùng ảnh bị crop. Nếu thiếu các thông tin này, chưa thể quy kết chênh lệch hallucination cho bản thân mô hình thay vì cho bước tokenization.

---

## PHẦN 3: PHÂN TÍCH CHÚ THÍCH & ĐẶC TRƯNG NGÔN NGỮ

VizWiz có 5.167/38.750 caption ($13,33\%$) là cùng một câu placeholder báo ảnh không nhận diện được. Ngoài ra, 964 caption có cờ `is_rejected`, trong đó 526 caption đồng thời là placeholder; vì vậy có thêm 438 caption cần loại. Sau lọc, phép đo ngôn ngữ chính chọn caption có annotation ID thấp nhất cho mỗi ảnh: **40.504 caption COCO** và **7.542 caption VizWiz**. Chỉ phép đo Jaccard, ontology cấp caption và đồng thuận object dùng tất cả 202.654/33.145 caption hợp lệ vì cần nhiều chú thích cùng ảnh.

Sau lọc, **208/7.750 ảnh VizWiz không còn caption hợp lệ**. Quy tắc một caption mỗi ảnh tránh ảnh có nhiều caption hợp lệ được tính nặng hơn, nhưng vẫn chỉ đo nội dung lời mô tả, chưa đo vật thể có thật trong ảnh.

| Chỉ số Ngôn ngữ | COCO | VizWiz (đã bỏ placeholder và caption bị từ chối) |
| :--- | :---: | :---: |
| **Số lượng Caption ($N$)** | $40.504$ | $7.542$ |
| **Độ dài từ trung bình (Word Mean)** | $10,44 \pm 2,22$ | **$11,75 \pm 3,95$** |
| **Trung vị số từ (Word Median)** | $10,0$ | $11,0$ |
| **Khoảng số từ [Min, Max]** | $[7, 43]$ | **$[7, 63]$** |
| **Độ dài ký tự trung bình (Char Mean)**| $52,04$ | $60,64$ |
| **Tổng số token từ** | $423.828$ | $89.433$ |
| **Quy mô từ vựng** | $8.522$ | $6.069$ |
| **Type-Token Ratio (TTR)** | $0,0201$ | **$0,0679$** |
| **MATTR (cửa sổ 1.000 token)** | $0,3718$ | $0,3944$ |

Trong tập caption đã chọn, câu dài nhất VizWiz có 63 từ; ngoại lai 155 từ ở kho tất cả caption không nằm trong mẫu một caption mỗi ảnh. Giá trị nhỏ nhất 7 từ ở cả hai tập theo cách đếm bằng khoảng trắng.

Hình 3A biểu diễn phần phân phối 3–30 từ theo tỷ lệ caption trong mỗi tập; Hình 3B dùng ECDF trên toàn khoảng để hiển thị đuôi dài của mẫu đã chọn.

### Các Phát hiện Ngôn ngữ học Nổi bật:
1. **Độ dài câu và đuôi phân phối dài (Long-tail)**:
   - Câu chú thích COCO có độ dài tập trung hẹp xung quanh 10 từ (độ lệch chuẩn $2,22$).
   - Câu chú thích VizWiz đã chọn dài trung bình $11,75$ từ và tối đa **63 từ**.
2. **Độ đa dạng từ vựng trên cùng quy mô**:
   - TTR toàn kho không được dùng để xếp hạng vì COCO có 423.828 token còn VizWiz có 89.433. MATTR cửa sổ 1.000 token là **0,3718 (COCO)** và **0,3944 (VizWiz)**.
3. **Top-15 danh từ tính từ dữ liệu**:
   - NLTK POS tagger chỉ giữ danh từ và WordNet lemmatizer chuẩn hóa số ít. Bảng báo cả count lẫn số lượt nhắc trên 1.000 caption.
   - VizWiz nhắc nhiều đến `screen`, `computer`, `table`, `bottle`, `box`; COCO nổi bật với `man`, `people`, `woman`. `person` riêng lẻ đạt 54,89/1.000 caption VizWiz và 41,16 ở COCO. [Bảng danh từ](eda_results/tables/caption_noun_entities_top15.csv).
   - Gộp `man/men/woman/people/person`, COCO có **326,71 lượt nhắc/1.000 caption**, VizWiz **79,29**; tỷ lệ caption đã chọn có ít nhất một từ thuộc nhóm là **30,92%** so với **7,78%**. Đây là tần suất lời mô tả, không phải số người thực sự trong ảnh. [Bảng nhóm từ chỉ người](eda_results/tables/person_reference_summary.csv).

Quy tắc chọn một caption có ID thấp nhất được dùng nhất quán trong phép đo chính và tái lập bằng `caption_sampling_sensitivity.py`; bảng phụ không còn chọn ngẫu nhiên. [Bảng đối chiếu](eda_results/tables/person_reference_one_caption_per_image.csv).

---

## PHẦN 4: PHÂN BỐ DANH MỤC ĐỐI TƯỢNG COCO (OBJECT CATEGORY BIAS)

Tập dữ liệu COCO được xây dựng dựa trên hệ thống phân cấp 80 danh mục đối tượng thuộc 12 siêu danh mục (Supercategories). Phân tích $291.875$ đối tượng được gán nhãn trong phạm vi khảo sát cho thấy:

Hình 4 so sánh **tỷ lệ ảnh có một caption được chọn nhắc đến từng lớp COCO-80** ở hai miền, cùng một đơn vị đo. Đây là tần suất từ theo bảng đồng nghĩa CHAIR, không phải ground truth object VizWiz; [bảng đủ 80 lớp](eda_results/tables/coco80_caption_mentions.csv) giữ số đếm và denominator. Bảng siêu danh mục bên dưới vẫn là tỷ lệ object instance COCO và có đơn vị khác Hình 4.

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

Trong 291.875 object instance của COCO val2014, person chiếm 30,20%, vehicle 11,16% và sports 5,75%. Đây là tỷ lệ instance, không phải tỷ lệ ảnh hoặc caption. Hình 4 dùng chung đơn vị tần suất nhắc lớp COCO-80 trong một caption được chọn/ảnh ở hai tập; COCO cũng có laptop, tv, keyboard, cell phone và book trong nhãn instance.

Ontology 80 danh mục COCO không có nhãn riêng cho nội dung chữ, nhãn sản phẩm hay tên thuốc. Caption VizWiz thường nhắc screen, box, package, hand và label. Danh sách danh từ này phản ánh lời mô tả, chưa đo nhu cầu sử dụng hoặc tần suất nhiệm vụ đọc thuốc/hạn dùng.

### Giới hạn của CHAIR/POPE khi so sánh COCO với VizWiz:

Không nên đọc điểm CHAIR trên COCO và VizWiz như hai đại lượng hoàn toàn tương đương. [CHAIR gốc](https://aclanthology.org/D18-1437/) chủ động giới hạn đối tượng vào **80 category của MS-COCO segmentation**. Nhiều danh từ phổ biến trong caption VizWiz như `screen`, `box`, `package`, `hand` và `label` không phải category COCO; chỉ một phần như `bottle`, `laptop/computer` hoặc `table` có thể ánh xạ gần đúng. Vì vậy CHAIR có thể **không đếm** hallucination ngoài ontology và làm điểm VizWiz trông tốt giả tạo.

[POPE](https://aclanthology.org/2023.emnlp-main.20/) giảm phụ thuộc vào phong cách caption bằng câu hỏi yes/no về sự tồn tại của vật thể, nhưng vẫn chỉ hợp lệ khi tập positive object và negative probe được xây dựng từ annotation phù hợp với VizWiz. Tái sử dụng nguyên object pool COCO cho VizWiz vẫn kế thừa ontology bias. Báo cáo chính nên:

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
- **Phương sai Laplacian sau chuẩn hóa**: áp dụng EXIF orientation, giữ tỷ lệ khung hình và resize cạnh ngắn về 480 px trên **toàn bộ 40.504 ảnh COCO và 7.750 ảnh VizWiz**. Bảng độ nhạy theo resize và đồng thuận caption bên dưới cũng tính trên toàn bộ val.
  - **MS-COCO val2014**: median **1.140,08**.
  - **VizWiz val**: median **152,16**.
  - Chênh lệch phản ánh năng lượng biên/texture thấp hơn, nhưng không xác định riêng motion blur; defocus, JPEG và cảnh ít texture cũng ảnh hưởng chỉ số.
- **Độ sáng toàn val**: median COCO **113,86**, median VizWiz **108,04**; hai trung vị gần nhau, nên trung vị độ sáng không phải yếu tố phân biệt rõ.

Trung vị Laplacian toàn val là **1.140,08 COCO** và **152,16 VizWiz**. Bootstrap percentile 95% CI (1.000 lần, seed=42) lần lượt **[1.126,16; 1.154,83]** và **[147,51; 157,31]**. Có **24.799/40.504 ảnh COCO** và **759/7.750 ảnh VizWiz** được phóng lớn khi chuẩn hóa cạnh ngắn. [Bảng từng ảnh](eda_results/tables/laplacian_full_val.csv), [tóm tắt](eda_results/tables/laplacian_full_val_summary.json).

Trên toàn bộ val, khi chỉ giữ ảnh có cạnh ngắn gốc ≥480 px, median là **1.578,21** ở COCO (n=15.705; bootstrap 95% CI [1.546,88; 1.610,63]) và **164,12** ở VizWiz (n=6.991; [158,45; 169,85]). Hai tập con khác nhau về kích thước và nội dung; không thể tách riêng hiệu ứng chất lượng ảnh. [Bảng độ nhạy toàn val](eda_results/tables/laplacian_resize_sensitivity_full.csv).

Kiểm tra trên toàn bộ 7.750 ảnh VizWiz cho thấy chỉ số giảm theo số annotator dùng caption từ chối:

| Caption từ chối | Số ảnh | Trung vị Laplacian | Bootstrap 95% CI |
| :---: | ---: | ---: | :---: |
| 0/5 | 5.222 | 201,69 | [194,03; 209,30] |
| 1/5 | 1.297 | 116,76 | [107,08; 127,04] |
| 2/5 | 480 | 79,76 | [67,28; 92,70] |
| 3/5 | 286 | 45,53 | [37,29; 58,17] |
| 4/5 | 273 | 25,76 | [20,96; 32,33] |
| 5/5 | 192 | 16,18 | [12,46; 20,74] |

Xu hướng này hỗ trợ giá trị chỉ báo của Laplacian đối với khả năng nhận diện nội dung, nhưng không chứng minh mọi ảnh có chỉ số thấp đều bị mờ. [Bảng toàn val](eda_results/tables/laplacian_by_precanned_full.csv) cho phép kiểm tra từng nhóm; nhãn lỗi chất lượng chính thức được ghép ở Phần 8.

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
| Median Laplacian, toàn bộ ảnh val | 1.140,08 | 152,16 |
| Median độ sáng, toàn bộ ảnh val | 113,86 | 108,04 |
| Median diện tích pad ở mode LLaVA gốc | 26,72% | 25,00% |
| Median cạnh dài file / 336 px | 1,90 lần | 3,86 lần |
| Ảnh có chữ theo đa số annotator | Không có nhãn tương ứng | 64,75% |
| Ảnh có caption từ chối theo đa số annotator | Không có nhãn tương ứng | 9,69% |
| Blur / poor framing theo đa số VizWiz-QualityIssues | Không có nhãn tương ứng | 29,91% / 36,45% |

Vị trí chủ thể và sai số CHAIR/POPE cấp ảnh chưa được đo. Mode pad của LLaVA-1.5-7B không cắt ảnh theo hình học, nhưng vẫn downsample về 336×336; tác động lên chữ nhỏ và hallucination cần thực nghiệm.

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

<!-- VLM_EDA_START -->
## PHẦN 8: KIỂM TRA EDA BỔ SUNG CHO LLaVA-1.5-7B

Các phép đo được tái lập bằng `vlm_eda_extensions.py`, `llava_preprocessing_audit.py`, `clip_embedding_audit.py` và `audit_val_images.py`; bảng nằm trong `eda_results/vlm_extensions/`. Checkpoint đánh giá được chốt là **[liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)**. Phép đo ảnh–văn bản dùng [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) công khai, [vision tower của LLaVA-1.5](https://github.com/haotian-liu/LLaVA), như một proxy trước khi chạy sinh đáp án. Caption tham chiếu cho thống kê ngôn ngữ, token và CLIP luôn là **một caption hợp lệ có annotation ID thấp nhất mỗi ảnh**; Jaccard và đồng thuận object cần mọi caption hợp lệ vì bản chất là phép đo giữa các annotator.

### 8.1. Toàn vẹn dữ liệu và nguy cơ rò rỉ

| Kiểm tra | COCO | VizWiz | Cách đọc |
| :--- | ---: | ---: | :--- |
| Trùng ID ảnh giữa train2014/val2014 | 0 | — | Kết quả hiển nhiên của cách chia COCO 2014; không dùng làm kết luận rò rỉ VLM. |
| **ID val2014 nằm trong train2017** | **35.504/40.504 (87,656%)** | — | Đo từ `instances_train2017.json` và `captions_val2014.json`; chỉ cần annotation, không cần ảnh train. |
| **ID val2014 có đường dẫn COCO trong mix instruction tuning LLaVA-1.5** | **5.741/40.504 (14,174%)** | **0 đường dẫn tên VizWiz** | Đo từ chính file 665K, không suy từ toàn bộ train2017; không bao trùm pretraining hay ảnh có thể trùng ở nguồn khác. |
| Caption val hợp lệ trùng câu train sau chuẩn hóa | 10.488/202.654 (5,175%) | 463/33.145 (1,397%) | Trùng câu phổ biến không chứng minh trùng ảnh hoặc contamination của VLM. Caption VizWiz từ chối/bị reject đã loại. |
| Ảnh val thiếu hoặc 0 byte | 0/40.504 | 0/7.750 | Kiểm tra theo manifest. |
| Giải mã toàn bộ ảnh val | 40.504/40.504 thành công | 7.750/7.750 thành công | EXIF transpose, giải mã và băm file. |
| Annotation mồ côi; trùng ID ảnh, filename, ID annotation trong từng split | 0 | 0 | Xem số theo split trong `integrity.csv`. |
| Ảnh có số caption khác 5 | Train: 197; val: 131 | Train: 0; val: 0 | Đây là đặc điểm annotation COCO, không tự động là lỗi. |

Trong file val, MD5 cho **49 nhóm ảnh trùng byte** (100 ảnh, đều VizWiz); pHash bằng nhau cho 85 nhóm (78 VizWiz, 7 COCO). pHash bằng nhau chỉ là ứng viên cần xem ảnh. Manifest eval không có cặp trùng byte nội bộ, nhưng 8 ảnh eval có bản sao ở phần val còn lại; `prepare_data.py` giữ eval và loại 9 bản sao khỏi dev. [Danh sách loại trừ](data_prepared/exact_duplicate_dev_exclusions.csv).

[File `llava_v1_5_mix665k.json` của tác giả LLaVA](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) được xác minh SHA-256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`, rồi quét các trường `image` cấp bản ghi. Có 624.610 bản ghi mang đường dẫn ảnh, 349.034 đường dẫn duy nhất, **5.741** ID COCO val2014 trong các đường dẫn COCO và **70/500** ID của manifest COCO. Không có đường dẫn chứa tên `VizWiz`. Con số 35.504 chỉ cho biết ảnh val2014 hiện diện trong COCO train2017, không cho biết tất cả đã được LLaVA dùng. Vẫn có thể có trùng nội dung giữa các nguồn ảnh hoặc giai đoạn pretraining; “ngoài mix” không đồng nghĩa “chưa từng thấy”. [Bảng từng ID](eda_results/vlm_extensions/llava15_instruction_mix_overlap.csv), [tóm tắt và hash nguồn](eda_results/vlm_extensions/llava15_instruction_mix_overlap.json), [kiểm tra ảnh](eda_results/vlm_extensions/val_image_audit.json).

### 8.2. Phân kỳ từ vựng và đồng thuận giữa caption

Tách từ bằng regex tiếng Anh và chuyển chữ thường trên 40.504 caption COCO và 7.542 caption VizWiz. JS unigram giữa hai miền là **0,266183 bit**. Mốc nhiễu nền từ 100 lần chia ngẫu nhiên caption COCO thành hai nửa (`seed=42`) là **0,018418 bit trung bình**, phân vị 95% **0,018630 bit**; do đó 0,266 lớn hơn rõ rệt nhiễu do chia mẫu ở COCO. Mốc nền cùng miền không thay thế khoảng tin cậy hai miền và không đo embedding. OOV của VizWiz so với từ vựng COCO train2014 là **4,5314% token** và **36,4543% type** theo cùng cách tách từ. [Bảng từ vựng](eda_results/vlm_extensions/vocabulary.csv), [100 từ đặc trưng log-odds](eda_results/vlm_extensions/distinctive_words_log_odds.csv), [tóm tắt](eda_results/vlm_extensions/summary.json).

Các từ phân biệt nổi bật trong [bảng log-odds](eda_results/vlm_extensions/distinctive_words_log_odds.csv) giúp đọc nguồn phân kỳ. Điểm dương thiên về VizWiz; điểm âm thiên về COCO. Điểm được làm trơn 0,5 và chỉ xét từ có tổng tần suất ≥20, nên đây là mô tả từ vựng, không phải nhãn object thật.

| Từ | COCO | VizWiz | Log-odds VizWiz − COCO |
| :--- | ---: | ---: | ---: |
| label | 1 | 299 | +6,81 |
| packet | 0 | 55 | +6,23 |
| nutritional | 0 | 51 | +6,15 |
| frisbee | 573 | 0 | −5,53 |
| court | 507 | 0 | −5,41 |
| surfboard | 479 | 0 | −5,35 |

Tính Jaccard từ nội dung giữa mọi cặp caption hợp lệ của mỗi ảnh, bỏ cùng một tập stopword rồi lấy trung bình theo ảnh. **COCO: 0,1989** trên 40.504 ảnh; **VizWiz: 0,1633** trên 7.268 ảnh có ít nhất hai caption hợp lệ. Có 208 ảnh VizWiz không còn caption hợp lệ; chúng không bị gán điểm 0. Điểm này đo từ trùng, không xác nhận hai caption đồng nghĩa hay bám sát ảnh. [COCO từng ảnh](eda_results/vlm_extensions/coco_caption_agreement.csv), [VizWiz từng ảnh](eda_results/vlm_extensions/vizwiz_caption_agreement.csv).

### 8.3. CLIP, tokenizer, ontology và tiền xử lý

| Hạng mục | Kết quả trên dữ liệu val | Giới hạn diễn giải |
| :--- | :--- | :--- |
| Khoảng cách miền trong CLIP image embedding | Trên **500 ảnh mỗi miền** (kể cả ảnh không có caption hợp lệ), cosine trung bình giữa miền **0,49271**; trong COCO **0,51932**, trong VizWiz **0,60370**; cosine hai centroid **0,87857**. | Mẫu lấy từ manifest, vector chuẩn hóa; ảnh được pad theo config LLaVA. CLIP là proxy, chưa phải kết quả sinh caption. |
| Căn khớp ảnh–caption CLIP | COCO cặp đúng **0,25520**, hoán vị **0,08742** (`n=500`); VizWiz **0,26027** so với **0,13186** (`n=483`). | Cosine CLIP so cặp đúng với một hoán vị nội miền `seed=42`; không xác nhận object có thật. |
| Độ dài caption theo tokenizer LLaVA-1.5-7B | COCO p50/p90/p95/p99 **13/17/19/24** token (`n=40.504`); VizWiz **14/21/25/35** (`n=7.542`). | Chỉ caption tham chiếu; chưa gồm chat template, ảnh hoặc output mô hình. |
| Từ COCO-80 trong caption hai tập | VizWiz: **17.456/33.145 caption hợp lệ (52,666%)**, **5.771/7.542 ảnh (76,518%)** có ít nhất một từ CHAIR; COCO: **78,58%/202.654 caption**, **79,52% ảnh** ở một caption được chọn và **94,04% ảnh** khi hợp nhất caption. | Cùng phép khớp từ; đây là mức áp dụng từ vựng, không phải tỷ lệ object thực sự hiện diện. |
| Độ đầy đủ của nhãn từ caption trên COCO | Một caption được chọn nhắc **38,07%** số lượt lớp object có trong instance annotation; hợp nhất mọi caption nhắc **55,26%** (`116.592` lượt lớp/ảnh). | Caption bỏ sót nhiều lớp được gán; lấy object từ caption làm ground truth có nguy cơ tính nhầm object thật thành hallucination. Nhãn instance cũng không tuyệt đối đầy đủ. |
| Ground truth object VizWiz | [Gợi ý object được ≥2 caption nhắc](eda_results/vlm_extensions/vizwiz_chair_ontology_mentions.csv); [100 ảnh để rà tay](eda_results/vlm_extensions/vizwiz_object_consensus_review100_seed42.csv). | Gợi ý yếu có thể cùng sai; người rà ảnh xác nhận trước khi tính CHAIR/POPE. |

[Bảng đồng nghĩa CHAIR gốc](https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt) được lưu tại `eda_resources/chair_synonyms.txt`. Phép đếm hiện dùng khớp từ/cụm từ không phân biệt chữ hoa, nên có thể khác chính xác bộ tiền xử lý số ít và các ngoại lệ của mã CHAIR gốc. [Kết quả CLIP](eda_results/vlm_extensions/clip_embedding_audit.json), [token và ontology](eda_results/vlm_extensions/summary.json).

**Diễn giải CLIP.** VizWiz có cosine nội miền cao hơn COCO (**0,60370 so với 0,51932**) dù khác miền; ảnh VizWiz trong mẫu này đồng nhất hơn theo biểu diễn CLIP. Điều đó làm baseline ảnh–caption hoán vị của VizWiz cao hơn (**0,13186 so với 0,08742**) và biên cặp đúng trừ hoán vị hẹp hơn (**0,12841 so với 0,16778**). Vì vậy không nên so điểm cặp đúng 0,26027/0,25520 riêng lẻ rồi kết luận alignment VizWiz tốt hơn. Đây là thống kê mô tả trên mẫu cố định, không phải kiểm định mô hình. Khi bỏ bước pad vuông và dùng resize/crop mặc định của CLIP, cosine nội miền còn **0,55781 VizWiz** và **0,48087 COCO**; khoảng chênh vẫn tồn tại. Padding có thể góp phần vào độ đồng nhất nhưng không giải thích hết. Phép không pad thay đổi cả crop nên không tách riêng nhân quả của màu viền.

Phân tầng VizWiz theo số caption từ chối và cờ chữ cho thấy biên cặp đúng–hoán vị ở tầng 0 từ chối là **0,14186** khi có chữ (`n=236`) và **0,12764** khi không chữ (`n=82`). Ở tầng 3+, biên còn **0,06497** khi có chữ (`n=7`) và **0,08559** khi không chữ (`n=26`); hai nhóm nhỏ nên chưa kết luận chắc về độ dốc. Cờ chữ cũng giảm từ **74,24%** ở toàn bộ VizWiz 0/5 từ chối xuống **22,80%** ở 3+, vì vậy không đọc tác động từng biến như độc lập. [Điểm từng ảnh](eda_results/vlm_extensions/clip_pair_scores.csv), [bảng 8 tầng](eda_results/vlm_extensions/clip_alignment_by_vizwiz_stratum.csv).

![So sánh CLIP với pad và không pad](eda_results/figures/08_clip_domain_alignment.png)

[Bảng độ đầy đủ nhãn COCO](eda_results/vlm_extensions/coco_caption_label_completeness.csv) đối chiếu tập lớp có trong instance annotation với lớp xuất hiện trong một hoặc mọi caption trên cùng ảnh; [tóm tắt](eda_results/vlm_extensions/coco_caption_label_completeness.json). Chỉ số recall ở đây lấy **số lớp COCO-80 khác nhau trên từng ảnh** làm denominator, không phải số instance; từ khớp nhầm và nhãn instance bỏ sót vẫn có thể làm lệch ước lượng.

[Config LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b/blob/main/config.json) ghi `image_aspect_ratio: pad`. Với ảnh 336×336 và patch 14×14, model nhận **576 patch token** mỗi ảnh. Mode pad theo mã gốc có **0% diện tích ảnh bị cắt, hệ số méo hình 1**; trung vị vùng padding là **26,7188% COCO** và **25,0% VizWiz**, tương đương khoảng **154** và **144** patch theo diện tích (ước lượng, không phải đếm patch rỗng ở biên). Cạnh dài trung vị của file là **640 px COCO** và **1.296 px VizWiz**, nên khi đưa về 336 px, tỷ số pixel gốc/pixel mô hình là **1,9048** và **3,8571**. Do đó tỷ lệ ảnh dọc không tự tạo bất lợi về số patch hoặc diện tích pad; việc thu nhỏ mạnh hơn ở VizWiz mới là rủi ro cụ thể cho chữ nhỏ (64,75% ảnh có cờ chữ theo đa số annotator), cần thử nghiệm đầu ra để xác nhận. [Bảng từng ảnh](eda_results/vlm_extensions/llava15_preprocessing_geometry.csv), [tóm tắt](eda_results/vlm_extensions/llava15_preprocessing_geometry.json).

Con số 0% bị cắt áp dụng khi chạy pipeline **gốc** ở mode pad. [Tài liệu Transformers về LLaVA](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava.md) lưu ý bản `llava-hf` cần đặt `LlavaImageProcessor(..., do_pad=True)` để khớp triển khai gốc; phải ghi cấu hình processor thực tế khi chạy mô hình. Center crop và resize thẳng vuông chỉ là tình huống đối chứng hình học, không phải cấu hình checkpoint đã đo.

### 8.4. Nhãn lỗi chất lượng VizWiz và giới hạn của Laplacian

[VizWiz-QualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/) công bố số phiếu (trên 5 annotator) cho blur, ảnh quá sáng/tối, framing, che khuất và xoay trên đúng 7.750 ảnh val. Ghép bằng filename khớp **7.750/7.750** ảnh. Với ngưỡng đa số ≥3/5: **2.318 ảnh mờ (29,91%)**, **2.825 ảnh framing kém (36,45%)**, **219 ảnh quá tối (2,83%)**, **200 ảnh quá sáng (2,58%)** và **116 ảnh bị che (1,50%)**. Các lỗi có thể cùng xuất hiện trên một ảnh nên không cộng tỷ lệ thành 100%. [Phiếu từng ảnh](eda_results/vlm_extensions/vizwiz_quality_issue_votes_val.csv), [tóm tắt và hash nguồn](eda_results/vlm_extensions/vizwiz_quality_issue_summary.json).

Tỷ lệ blur đa số tăng từ **21,16%** ở nhóm 0/5 caption từ chối lên **65,38%** ở nhóm 3/5; nhóm 5/5 là **55,73%**. Framing kém là **31,39%** ở 0/5 và **51,46%** ở 3/5. Đây là đồng xuất hiện giữa hai loại nhãn trên cùng ảnh, chưa chứng minh blur hoặc framing gây lỗi của mô hình. Trong ảnh được gán blur đa số, median Laplacian là **75,41**, so với **260,12** ở ảnh được gán không có lỗi đa số; điều này ủng hộ dùng Laplacian làm chỉ báo, không thay nhãn nguyên nhân. [Bảng theo mức từ chối](eda_results/vlm_extensions/vizwiz_quality_by_refusal.csv).

![Nhãn lỗi chất lượng VizWiz trên toàn bộ val](eda_results/figures/09_vizwiz_quality_labels.png)

### 8.5. Thiết kế tập đánh giá

[Manifest seed 42](eda_results/vlm_extensions/evaluation_manifest_seed42.csv) cố định **500 ảnh COCO val và 500 ảnh VizWiz val**. Đây là bộ mẫu 500 ảnh COCO duy nhất của báo cáo. VizWiz chia **theo tỷ lệ** của 8 tầng số caption từ chối (0, 1, 2, 3+) × `text_detected`, phân bổ bằng phần dư lớn nhất. COCO chọn ngẫu nhiên độc lập.

Trên 500 ảnh COCO này, **70 ID** có đường dẫn trong mix instruction tuning LLaVA và **430 ID** không có. Khi có output, báo cáo kết quả cả 500 để giữ mẫu đã khóa, đồng thời báo cáo riêng 70/430 với denominator và CI. Không thay manifest sau khi thấy output mô hình. [Cờ từng ảnh](eda_results/vlm_extensions/llava15_instruction_mix_overlap.csv) cho phép tách nhóm; trạng thái ngoài mix không chứng minh ngoài mọi nguồn huấn luyện.

| Từ chối (gộp 3+) | Không có chữ: tổng/mẫu | Có chữ: tổng/mẫu | Mẫu cộng |
| :---: | ---: | ---: | ---: |
| 0 | 1.267 / 82 | 3.651 / 236 | 318 |
| 1 | 483 / 31 | 749 / 48 | 79 |
| 2 | 298 / 19 | 416 / 27 | 46 |
| 3+ | 684 / 44 | 202 / 13 | **57** |

Nhóm 3+ chỉ có 57 ảnh, trong đó tầng có chữ chỉ 13. Để ước lượng hiệu năng toàn miền, giữ manifest phân tầng theo tỷ lệ và báo cáo khoảng tin cậy. Để phân tích riêng ca khó, chạy bổ sung trên **toàn bộ 886 ảnh val có ≥3 caption từ chối** (684 không chữ, 202 có chữ), báo cáo riêng, không trộn số này vào ước lượng 500 ảnh khi chưa gán trọng số. Hai biến tầng không độc lập: cờ chữ xuất hiện ở **3.651/4.918 ảnh (74,24%)** nhóm 0/5, nhưng chỉ **202/886 (22,80%)** nhóm 3+. [Bảng phân tầng CSV](eda_results/vlm_extensions/evaluation_manifest_strata_seed42.csv).

[Danh sách POPE COCO chính thức](https://github.com/RUCAIBox/POPE/tree/main/output/coco) có 500 ảnh và 3.000 câu hỏi ở bản random; chỉ **5 ảnh** giao với manifest COCO của báo cáo. Nếu so điểm với số POPE đã công bố, phải chạy theo danh sách/câu hỏi chính thức, không gán điểm từ manifest EDA cho POPE. [Bảng đối chiếu](eda_results/vlm_extensions/pope_manifest_overlap.json).

### 8.6. Chạy thử LLaVA và trạng thái thực thi

Đã cố định giao thức trong [`llava_pilot.py`](llava_pilot.py): lấy ngẫu nhiên `seed=42` **100 ảnh mỗi tập từ `data_prepared/dev.jsonl`**, giữ nguyên eval để đánh giá cuối; EXIF transpose, pad vuông bằng màu trung bình CLIP rồi dùng processor của checkpoint Hugging Face tương đương. Prompt là `Describe the visible content in one concise sentence. If the image is too unclear to identify reliably, say "I cannot tell from this image."`, `max_new_tokens=64`, greedy (`do_sample=False`, `num_beams=1`). Script lưu nguyên câu trả lời, số output token, cờ từ chối, ID và cấu hình vào JSONL để sau này phân tích độ dài, tỷ lệ từ chối và lỗi theo ảnh.

**Chưa có output LLaVA thật.** Máy hiện chỉ có tokenizer LLaVA và trọng số CLIP, không có trọng số ngôn ngữ LLaVA-1.5-7B; GPU RTX 3050 4 GB không đủ chạy checkpoint 7B nguyên dạng trong cấu hình hiện có. Script dừng rõ khi không thấy trọng số, không tự tải mô hình lớn hoặc tạo kết quả giả. Khi có môi trường phù hợp, cài `requirements-llava-pilot.txt`, cung cấp `--model-path` trỏ đến bản chuyển đổi `llava-hf/llava-1.5-7b-hf` đã tải, chạy pilot rồi rà ít nhất 20 ca lỗi bằng ảnh và caption. Cấu hình dùng **pad vuông thủ công trước processor** để tái hiện mode pad dù phiên bản Transformers ở máy hiện tại chưa có `LlavaImageProcessor(do_pad=True)`; phải xác nhận tương đương trên checkpoint thực tế trước khi so điểm cuối.

### 8.7. Quyền riêng tư và giấy phép

VizWiz có thể chứa thông tin cá nhân như thuốc, thẻ, giấy tờ và màn hình. Manifest chỉ lưu ID và thống kê; lưới ảnh minh họa trong báo cáo cần được rà thủ công trước khi chia sẻ ra ngoài nhóm nghiên cứu, cũng như đầu ra OCR/caption có thể chép thông tin cá nhân. Kiểm tra điều khoản sử dụng của từng bộ dữ liệu trước khi phát hành ảnh hoặc bộ dẫn xuất; không suy diễn rằng giấy phép của [VizWiz-Priv](https://vizwiz.org/tasks-and-datasets/vizwiz-priv/) tự động áp dụng cho VizWiz-Captions. [Điều khoản COCO](https://cocodataset.org/#termsofuse) và [bài báo VizWiz-Captions](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2783_ECCV_2020_paper.php) là điểm đối chiếu nguồn.
<!-- VLM_EDA_END -->

---

## PHẦN 9: GIẢ THUYẾT VÀ THỰC NGHIỆM TIẾP THEO

EDA đo khác biệt của dữ liệu đầu vào; chưa đo chất lượng caption/answer do cùng một mô hình sinh ra. Ba can thiệp sau cần được đánh giá bằng ablation trên cùng tập ảnh và cùng mô hình:

1. **Ngưỡng từ chối:** So sánh mô hình gốc với cơ chế từ chối theo xác suất ảnh khó nhận diện. Báo cáo precision/recall của quyết định từ chối trên các mức đồng thuận 0/5–5/5, cùng tỷ lệ câu trả lời có thông tin sai.
2. **OCR bổ trợ:** So sánh câu trả lời khi bật/tắt OCR trên ảnh có và không có nhãn chữ. Đo độ chính xác thông tin chữ và hallucination; tỷ lệ ảnh có chữ tự nó chưa chứng minh encoder thị giác hiện tại không đủ.
3. **Tiền xử lý và tăng cường dữ liệu:** Dùng mode pad hiện có của LLaVA-1.5-7B làm baseline; so với các biến thể giữ độ chi tiết chữ nhỏ hoặc crop có kiểm soát trên cùng ảnh và cùng ngân sách token. Vì checkpoint đã pad, không trình bày "Smart Letterbox Padding" như một can thiệp mới.

---



---

## TÀI NGUYÊN ĐÃ XUẤT XƯỞNG (EXPORTED ARTIFACTS)

Biểu đồ (200–300 DPI) và dữ liệu bảng biểu được lưu tại:
- **Thư mục Đồ thị**: `eda_results/figures/`
  1. `02_resolution_aspect_ratio.png`: Phân bố độ phân giải và tỷ lệ khung hình.
  2. `03_caption_length_and_tokens.png`: Phân bố độ dài từ và từ vựng đặc trưng.
  3. `04_coco_category_distribution.png`: Tần suất nhắc COCO-80 trong caption được chọn ở hai tập.
  4. `05_vizwiz_quality_and_text_issues.png`: Tỷ lệ ảnh suy giảm chất lượng, chữ viết và độ sắc nét quang học.
  5. `07_qualitative_samples_grid.png`: Lưới ảnh đối sánh trực quan các trường hợp tiêu biểu.
  6. `08_clip_domain_alignment.png`: CLIP có/không pad và cặp caption đúng/hoán vị.
  7. `09_vizwiz_quality_labels.png`: Nhãn lỗi chất lượng VizWiz chính thức.
- **Thư mục Bảng số liệu CSV**: `eda_results/tables/`
- **Kiểm tra Laplacian theo đồng thuận precanned trên toàn val**: `eda_results/tables/laplacian_by_precanned_full.csv`.
- **Độ nhạy theo resize và từ chỉ người**: `eda_results/tables/laplacian_resize_sensitivity_full.csv`, `eda_results/tables/person_reference_summary.csv`, `eda_results/tables/person_reference_one_caption_per_image.csv`.
- **Chỉ số từng ảnh trong mẫu đo**: `eda_results/tables/image_metrics_sample.csv`.
- **Manifest ví dụ định tính và bộ đánh giá cố định**: `eda_results/tables/qualitative_samples_manifest.csv`, `eda_results/vlm_extensions/evaluation_manifest_seed42.csv`.
- **File Dữ liệu Thống kê Đầy đủ (JSON)**: `eda_results/eda_summary_metrics.json`
- **Script Thực thi Tái lập (Reproducible Python Script)**: `eda_coco_vs_vizwiz.py`
- **Script kiểm tra một caption mỗi ảnh**: `caption_sampling_sensitivity.py`.
- **EDA VLM và manifest đánh giá**: `vlm_eda_extensions.py`, `eda_results/vlm_extensions/`.
- **Đối chiếu nguồn huấn luyện/chất lượng/nhãn object**: `llava_training_overlap.py`, `vizwiz_quality_audit.py`, `coco_caption_label_completeness.py`.
- **Hình và pilot**: `plot_coco80_mentions.py`, `plot_vlm_findings.py`, `llava_pilot.py` (chưa chạy vì thiếu trọng số 7B).

---

## TÀI LIỆU THAM KHẢO

1. Lin, T.-Y. và cộng sự (2014). *Microsoft COCO: Common Objects in Context*. ECCV. [DOI: 10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48).
2. Gurari, D. và cộng sự (2020). *Captioning Images Taken by People Who Are Blind*. ECCV. [Trang bài báo ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2783_ECCV_2020_paper.php).
3. Rohrbach, A. và cộng sự (2018). *Object Hallucination in Image Captioning*. EMNLP. [ACL Anthology](https://aclanthology.org/D18-1437/).
4. Li, Y. và cộng sự (2023). *Evaluating Object Hallucination in Large Vision-Language Models*. EMNLP. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.20/).
5. Liu, H. và cộng sự. *Improved Baselines with Visual Instruction Tuning* và [file mix665k chính thức](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json).
6. Chiu, T.-Y., Zhao, Y. và Gurari, D. (2020). *Assessing Image Quality Issues for Real-World Problems*. CVPR. [VizWiz-QualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/).
7. [POPE official repository](https://github.com/RUCAIBox/POPE), danh sách câu hỏi COCO random, popular và adversarial.
