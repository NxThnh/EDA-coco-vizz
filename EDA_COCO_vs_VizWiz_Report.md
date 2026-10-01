# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA): ĐỐI SÁNH TẬP DỮ LIỆU MS-COCO VÀ VIZWIZ-CAPTIONS
## Phục vụ Khóa Luận Tốt Nghiệp: Nghiên cứu Domain Gap và Thách thức Thị giác Đời thực

---

## TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Trong các bài toán Thị giác Máy tính đa phương thức (Vision-Language / Image Captioning), **MS-COCO (Common Objects in Context)** từ lâu đã được xem là "tiêu chuẩn vàng" (gold standard benchmark). Tuy nhiên, các mô hình huấn luyện hoàn toàn trên COCO thường thất bại nghiêm trọng hoặc sinh văn bản ảo tưởng (hallucination) khi triển khai vào các ứng dụng hỗ trợ đời thực.

Báo cáo này thực hiện phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA) so sánh định lượng và định tính giữa **MS-COCO (2014)** và **VizWiz-Captions**. Kết quả phân tích chỉ ra sự tồn tại của một **Domain Gap khổng lồ** giữa ảnh chụp có chủ đích thẩm mỹ (curated/photographer bias) và ảnh chụp đời thực từ góc nhìn thứ nhất của người khiếm thị (in-the-wild assistive egocentric capture).

### Các Phát Hiện Định Lượng Cốt Lõi:
1. **Đảo ngược tỷ lệ khung hình (Aspect Ratio Inversion)**: COCO chiếm **71.71% ảnh ngang (Landscape)** do chụp từ máy ảnh kỹ thuật số Flickr chuẩn hóa $\le 640\text{px}$. Ngược lại, VizWiz có tới **92.86% ảnh dọc (Portrait)** chụp trực tiếp từ smartphone của người dùng.
2. **Độ phân giải & Cảm biến**: COCO có độ phân giải cố định trung bình $576.5 \times 485.1\text{px}$ (0.28 MP). VizWiz giữ nguyên cảm biến camera di động với trung bình $1067.7 \times 1339.2\text{px}$ (1.57 MP), dao động từ $23\text{px}$ đến $3264\text{px}$.
3. **Độ sắc nét & Rung mờ (Blur)**: Phép đo phương sai Laplacian (Laplacian Variance) chứng minh độ nét của COCO đạt trung bình **$1082.07$**, trong khi VizWiz chỉ đạt **$399.99$** (thấp hơn gần 3 lần), phản ánh hiện tượng rung tay (motion blur) và lệch tiêu cự (defocus blur).
4. **Tỷ lệ ảnh suy giảm chất lượng nặng (Quality Degradation)**: **32.62%** số ảnh trong VizWiz Val có ít nhất một annotator báo lỗi không thể nhận diện được; **2.48%** số ảnh bị hỏng hoàn toàn (100% annotators đều gán nhãn *"Quality issues are too severe to recognize visual content"*). COCO hoàn toàn không có cơ chế xử lý ảnh hỏng này.
5. **Nhu cầu nhận diện chữ viết (Text/OCR Relevance)**: **73.72%** số ảnh VizWiz có sự hiện diện của chữ viết (text detected), trong đó **47.10%** đạt đồng thuận tuyệt đối (5/5 annotators). Người khiếm thị chụp ảnh chủ yếu để đọc nhãn thuốc, bao bì, hạn sử dụng, màn hình vi tính.
6. **Lệch pha ngữ nghĩa (Semantic Gap)**: COCO tập trung áp đảo vào con người (`person` chiếm **30.20%** thực thể, `vehicle` chiếm **11.16%**) trong các bối cảnh thể thao ngoài trời. Ngược lại, từ vựng đặc trưng của VizWiz là đồ gia dụng, bao bì, màn hình, nhãn mác (`screen`, `bottle`, `box`, `can`, `package`, `label`, `medicine`, `hand`).

---

## PHẦN 1: PHẠM VI ĐỐI SÁNH DỮ LIỆU

Phần mở đầu không chia theo train/validation/test vì mục tiêu ở đây là mô tả **hai miền dữ liệu**, không phải giao thức huấn luyện. Các phân tích chi tiết sử dụng đúng hai tập ảnh có đủ ảnh gốc và chú thích cần thiết:

| Miền dữ liệu | Phạm vi dùng trong EDA | Vai trò trong đối sánh | Đặc trưng thu nhận |
| :--- | :---: | :--- | :--- |
| **MS-COCO 2014** | 40.504 ảnh có caption và instance annotation | Miền ảnh chuẩn/curated làm mốc | Ảnh Flickr được người sáng mắt căn khung |
| **VizWiz-Captions** | 7.750 ảnh, 38.750 caption gốc | Miền hỗ trợ thị giác đời thực | Ảnh smartphone do người khiếm thị chụp |
| **COCO-500 (báo cáo lại Bảng 2 & 4)** | 500 ảnh lấy ngẫu nhiên, `seed=42` | Kiểm tra độ ổn định khi cân bằng quy mô đánh giá | Danh sách ID được lưu để tái lập |

Đơn vị phân tích được ghi rõ ở từng bảng: thống kê ảnh dùng số ảnh; phân tích ngôn ngữ dùng caption mô tả hợp lệ; phân bố đối tượng dùng số object instance. Cách trình bày này tránh cộng gộp những tập không có cùng loại annotation.

---

## PHẦN 2: ĐỘ PHÂN GIẢI & TỶ LỆ KHUNG HÌNH (RESOLUTION & ASPECT RATIO)

Khảo sát được thực hiện trên 40,504 ảnh `COCO val2014` và toàn bộ 7,750 file ảnh nhị phân gốc của `VizWiz val`:

Để đối chiếu theo yêu cầu với đúng 500 ảnh COCO, một mẫu ngẫu nhiên cố định (`seed=42`) cũng được chạy lại riêng. Mẫu cho kết quả: kích thước trung bình $576,9\times481,8$ px, $73,60\%$ ảnh ngang, $22,80\%$ ảnh dọc và $3,60\%$ ảnh vuông. Ảnh chụp bảng: [`table2_coco500_resolution.png`](eda_results/coco500/table2_coco500_resolution.png).

| Chỉ số hình ảnh | MS-COCO 2014 Val ($N=40,504$) | VizWiz Val ($N=7,750$) | Ý nghĩa trong Khóa luận |
| :--- | :---: | :---: | :--- |
| **Chiều rộng trung bình (Mean Width)** | $576.5 \pm 91.9\text{ px}$ | **$1067.7 \pm 429.5\text{ px}$** | VizWiz có phương sai kích thước rất lớn |
| **Trung vị Chiều rộng (Median Width)** | $640.0\text{ px}$ | $968.0\text{ px}$ | COCO bị giới hạn cận trên cố định |
| **Chiều rộng [Min, Max]** | $[120, 640]\text{ px}$ | **$[23, 3264]\text{ px}$** | VizWiz có ảnh siêu nhỏ ($23\text{px}$) và siêu lớn |
| **Chiều cao trung bình (Mean Height)** | $485.1 \pm 97.8\text{ px}$ | **$1339.2 \pm 383.7\text{ px}$** | Chiều dọc VizWiz vượt trội chiều ngang |
| **Trung vị Chiều cao (Median Height)** | $480.0\text{ px}$ | $1296.0\text{ px}$ | Độ phân giải camera điện thoại Apple/Android |
| **Chiều cao [Min, Max]** | $[111, 640]\text{ px}$ | $[25, 2448]\text{ px}$ | Chênh lệch độ phân giải cực đoan |
| **Tỷ lệ khung hình trung bình (AR Mean)** | **$1.25$** (Ngang) | **$0.79$** (Dọc) | **Sự đảo ngược định hướng vật lý** |
| **Trung vị AR (Median AR)** | $1.33$ ($4:3$) | $0.75$ ($3:4$) | Tỷ lệ cảm biến chuẩn của 2 thiết bị khác nhau |
| **Tỷ lệ ảnh Ngang (Landscape: $AR > 1.05$)** | **$71.71\%$** ($29,044$ ảnh) | **$6.98\%$** ($541$ ảnh) | Người sáng mắt cầm máy ngang chụp phong cảnh |
| **Tỷ lệ ảnh Dọc (Portrait: $AR < 0.95$)** | **$23.70\%$** ($9,601$ ảnh) | **$92.86\%$** ($7,197$ ảnh) | **Người dùng cầm điện thoại thẳng đứng** |
| **Tỷ lệ ảnh Vuông ($0.95 \le AR \le 1.05$)** | $4.59\%$ ($1,859$ ảnh) | $0.15\%$ ($12$ ảnh) | Rất ít ảnh vuông tự nhiên |
| **Số Megapixel trung bình (Mean MP)** | $0.276\text{ MP}$ | **$1.566\text{ MP}$** | VizWiz nặng hơn 5.6 lần về mật độ điểm ảnh |

### Ý nghĩa Kỹ thuật đối với Pipeline Tiền xử lý (Preprocessing):
Hầu hết các mạng thị giác máy tính (Vision Transformer như ViT-B/16, CLIP, Swin) yêu cầu đầu vào ảnh vuông cố định $224 \times 224\text{px}$ hoặc $384 \times 384\text{px}$.
- Khi áp dụng phép **Direct Resize (Squishing)** hoặc **Center Crop** truyền thống:
  - Với ảnh COCO ($71.7\%$ ngang), phép cắt giữa vẫn giữ được trọng tâm vật thể.
  - Với ảnh VizWiz ($92.9\%$ dọc và chủ thể thường nằm lệch rìa do người khiếm thị không căn được khung ngắm), phép **Center Crop sẽ trực tiếp cắt đứt thông tin quan trọng nhất** (ví dụ cắt mất nhãn thuốc hoặc dòng chữ cần đọc).

### Phân tích Visual-token:

Visual-token không đồng nghĩa với pixel. Với một vision encoder chia ảnh thành patch kích thước $P\times P$, số token trước khi pooling xấp xỉ
$$N_{visual}=\left\lceil H'/P\right\rceil\left\lceil W'/P\right\rceil,$$
trong đó $H',W'$ là kích thước **sau tiền xử lý**. Vì vậy cần tách hai trường hợp:

- **Fixed square resize/crop**: COCO và VizWiz đều có cùng số token danh nghĩa (ví dụ ảnh $224\times224$, patch $16\times16$ cho 196 patch token), nhưng VizWiz bị nén mạnh theo trục dọc hoặc mất vùng rìa khi center-crop. Số token bằng nhau không có nghĩa lượng thông tin hữu ích bằng nhau.
- **Aspect-ratio preserving / dynamic tiling**: ảnh dọc VizWiz có thể cần nhiều tile/token hơn để giữ chữ nhỏ và vật thể lệch tâm. Nếu ép cùng ngân sách token với COCO, mỗi token phải đại diện vùng nội dung lớn hơn hoặc hệ thống phải downsample nhiều hơn, làm giảm khả năng đọc nhãn/OCR.
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

### Các Phát hiện Ngôn ngữ học Nổi bật:
1. **Độ dài câu và đuôi phân phối dài (Long-tail)**:
   - Câu chú thích COCO có độ dài tập trung hẹp xung quanh 10 từ (độ lệch chuẩn chỉ $2.38$).
   - Câu chú thích VizWiz miêu tả có độ dài trung bình lớn hơn ($12.04$ từ) và kéo dài tới **$155$ từ**. Người gán nhãn phải ghi chép chi tiết từng dòng chữ trên bao bì, thương hiệu, màu sắc nút bấm để hỗ trợ tối đa cho người dùng khiếm thị.
2. **Độ phong phú từ vựng (TTR)**:
   - Tỷ lệ Type-Token Ratio của VizWiz ($0.0267$) cao gấp **3.2 lần** so với COCO ($0.0082$). Điều này chứng minh từ vựng trong VizWiz phong phú hơn rất nhiều do bao gồm tên các nhãn hàng, thuật ngữ y tế, loại thuốc, vật dụng gia đình cụ thể.
3. **Từ vựng đặc trưng đối lập (Semantic Shift)**:
   - **Top thực thể COCO**: `man` (24,721), `people` (14,064), `woman` (11,418), `street` (10,374), `table` (10,121), `tennis` (6,904), `dog` (6,238). Ngữ cảnh chủ đạo: con người hoạt động ngoài trời, thể thao, động vật.
   - **Top thực thể VizWiz**: `screen` (3,177), `computer` (3,159), `table` (2,990), `bottle` (2,655), `box` (2,414), `can` (2,297), `package` (1,520), `hand` (1,446), `label` (1,344), `medicine` (1,152). Ngữ cảnh chủ đạo: tay cầm đồ vật, đọc thông tin trên bao bì hộp/chai, màn hình máy tính.

---

## PHẦN 4: PHÂN BỐ DANH MỤC ĐỐI TƯỢNG COCO (OBJECT CATEGORY BIAS)

Tập dữ liệu COCO được xây dựng dựa trên hệ thống phân cấp 80 danh mục đối tượng thuộc 12 siêu danh mục (Supercategories). Phân tích $291,875$ đối tượng được gán nhãn trong tập Val cho thấy:

Trên đúng mẫu COCO-500 ở Phần 1 có 3.719 object instance. Ba siêu danh mục lớn nhất vẫn là `person` (29,52%), `vehicle` (12,34%) và `kitchen` (10,24%), gần sát phân bố toàn tập. Ảnh chụp bảng: [`table4_coco500_supercategories.png`](eda_results/coco500/table4_coco500_supercategories.png).

### 12 Siêu danh mục (Supercategories) trong COCO Val:
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

### Luận điểm Phê phán cho Khóa Luận (Critical Thesis Argument):
- **Sự thiên lệch con người (Person-Centric Bias)**: COCO được thu thập từ ảnh mạng xã hội/Flickr, nơi người chụp có chủ đích chụp người thân, bạn bè hoặc chụp phong cảnh có con người.
- **Sự thiếu vắng các danh mục hỗ trợ thiết yếu**: Người khiếm thị không cần AI phát hiện người chơi tennis hay lướt sóng; nhu cầu hàng ngày của họ là:
  - Phân biệt lọ thuốc cảm cúm ban đêm vs ban ngày.
  - Đọc hạn sử dụng trên hộp sữa.
  - Biết số tiền trên tờ tiền giấy hoặc đọc thông báo lỗi trên màn hình máy tính.
  COCO hoàn toàn không có danh mục văn bản, nhãn hàng, hay thuốc men.

### Giới hạn của CHAIR/POPE khi so sánh COCO với VizWiz:

Không nên đọc điểm CHAIR trên COCO và VizWiz như hai đại lượng hoàn toàn tương đương. CHAIR gốc chủ động giới hạn đối tượng vào **80 category của MS-COCO segmentation**. Trong Top-15 thực thể đặc trưng của VizWiz, nhiều khái niệm phục vụ hỗ trợ như `screen`, `box`, `can`, `package`, `hand`, `label`, `medicine` không phải category COCO; chỉ một phần nhỏ như `bottle`, `laptop/computer` hoặc `table` có thể ánh xạ gần đúng. Vì vậy CHAIR có thể **không đếm** hallucination ngoài ontology và làm điểm VizWiz trông tốt giả tạo.

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

Khảo sát mức độ đồng thuận của 5 annotators trên 7,750 ảnh VizWiz Val:

| Mức độ chất lượng | Số lượng ảnh | Tỷ lệ (%) | Ý nghĩa thực tế |
| :--- | :---: | :---: | :--- |
| **0 / 5 báo lỗi (Rõ ràng hoàn toàn)** | **5,222** | **67.38%** | Ảnh chất lượng tốt, đủ ánh sáng và nét |
| **1 / 5 báo lỗi (Mơ hồ nhẹ)** | 1,297 | 16.74% | Có 1 người thấy khó nhìn nhưng 4 người vẫn đoán được |
| **2 / 5 báo lỗi (Suy giảm đáng kể)** | 480 | 6.19% | Chất lượng suy giảm, bắt đầu gây tranh cãi |
| **3 / 5 báo lỗi (Đa số không nhận ra)**| 286 | 3.69% | Đa số người gán nhãn đầu hàng trước chất lượng ảnh |
| **4 / 5 báo lỗi (Suy giảm nghiêm trọng)** | 273 | 3.52% | Rất mờ hoặc rất tối, chỉ 1 người cố đoán |
| **5 / 5 báo lỗi (Hỏng ảnh hoàn toàn)**| **192** | **2.48%** | **100% không thể nhận diện được hình ảnh** |
| **TỔNG CỘNG ẢNH CÓ LỖI CHẤT LƯỢNG ($\ge 1/5$)** | **2,528** | **32.62%** | **Gần 1/3 dữ liệu thực tế bị suy giảm chất lượng!** |

### 2. Sự Hiện diện của Chữ viết (Text / OCR Relevance):
Khảo sát sự đồng thuận của 5 annotators về việc có xuất hiện văn bản trong ảnh hay không:
- **0 người thấy chữ**: 2,037 ảnh ($26.28\%$)
- **Có từ 1 đến 4 người thấy chữ**: 2,063 ảnh ($26.62\%$)
- **Cả 5 người cùng thấy chữ rõ ràng**: **3,650 ảnh ($47.10\%$)**
- **Tổng số ảnh có chữ ($\ge 1$ annotator)**: **5,713 ảnh ($73.72\%$)**

$\rightarrow$ **Gần 3/4 số ảnh VizWiz đòi hỏi khả năng đọc và hiểu chữ (OCR / Scene Text Recognition)**. Nếu một mô hình Captioning chỉ dựa vào đặc trưng hình ảnh trừu tượng của ResNet/ViT mà không có module OCR tích hợp, mô hình sẽ không thể hỗ trợ người khiếm thị đọc nhãn sản phẩm.

### 3. Đo lường Khách quan bằng Thuật toán Xử lý Ảnh (Image Processing Metrics):
- **Độ nét (Laplacian Variance)**:
  $$\text{Var}(\nabla^2 I) = \frac{1}{HW}\sum_{x,y} \left( L(x,y) - \mu_L \right)^2$$
  - **COCO Val**: Đạt trung bình **$1082.07$**, phân bố lệch mạnh về phía sắc nét, ảnh tĩnh, độ tương phản biên cạnh cao.
  - **VizWiz Val**: Chỉ đạt **$399.99$** (thấp hơn $63\%$). Đồ thị mật độ (KDE) cho thấy một đỉnh lớn nằm ở vùng giá trị cực thấp ($\log_{10} < 1.5$), minh chứng cho hiện tượng mất nét nặng do rung tay hoặc khoảng cách lấy nét (macro focus) quá gần.
- **Độ sáng & Phơi sáng (Mean Luminance)**:
  - COCO có phân bố chuẩn đối xứng quanh mức sáng trung bình lý tưởng ($114.7 \pm 42$).
  - VizWiz có phân bố phân tán mạnh ($104.5$), xuất hiện nhiều ảnh ở 2 cực đoan: cực tối (underexposed do chụp trong góc tủ, phòng tối không bật đèn) hoặc lóa trắng (overexposed do đèn flash điện thoại phản chiếu trực tiếp vào bề mặt nhẵn).

### Mờ/lệch tâm có tương quan với CHAIR/POPE hay không?

**Chưa thể trả lời bằng số liệu hiện có.** EDA mới có độ nét Laplacian và mô tả định tính về framing, trong khi repo chưa có caption/answer do cùng một mô hình sinh ở cấp ảnh và chưa có điểm CHAIR/POPE theo từng ảnh. Chênh lệch trung bình giữa hai dataset không phải là bằng chứng tương quan và không được diễn giải như quan hệ nhân quả.

Kiểm định tối thiểu cần thực hiện trên cùng bộ ảnh và cùng cấu hình decoding:

- $x_1=\log(1+\mathrm{Var}(\nabla^2 I))$ cho độ nét; $x_2$ là khoảng cách chuẩn hóa từ tâm bbox/mask của vật thể chính đến tâm ảnh; thêm độ sáng, số object, độ dài caption và dataset làm biến kiểm soát.
- $y$ gồm CHAIR-I/CHAIR-S ở cấp ảnh và POPE error/F1. Dùng Spearman $\rho$ kèm bootstrap 95% CI; với POPE error nhị phân, dùng logistic regression có interaction `dataset × blur` và `dataset × off-center`.
- Phân tích riêng COCO và VizWiz, đồng thời báo cáo trên ontology chung và ontology mở rộng. Nếu không, hiệu ứng “dataset” và giới hạn 80 category sẽ gây nhiễu kết luận.
- Chỉ kết luận có tương quan khi CI không cắt 0 và hiệu ứng còn tồn tại sau khi kiểm soát độ dài output/ontology coverage. Thực nghiệm làm mờ/crop có kiểm soát trên cùng ảnh sẽ mạnh hơn tương quan quan sát trong việc lập luận nguyên nhân.

---

## PHẦN 6: PHÂN TÍCH CHUYÊN SÂU VỀ DOMAIN GAP COCO ↔ VIZWIZ

Phần phân tích này cung cấp cơ sở lý luận học thuật vững chắc cho chương Đặt vấn đề và Động lực nghiên cứu (Problem Formulation & Motivation) trong khóa luận:

| Chiều không gian so sánh | Chuẩn mực Benchmark (MS-COCO) | Thực tế Đời thực (VizWiz-Captions) | Tác động Kỹ thuật đến Mô hình AI |
| :--- | :--- | :--- | :--- |
| **1. Tác nhân chụp ảnh (Operator)** | Người sáng mắt cầm máy ảnh/smartphone; nhìn thấy màn hình, chủ động căn chỉnh bố cục. | **Người khiếm thị hoàn toàn hoặc suy giảm thị lực nặng**; chụp không nhìn thấy màn hình. | Dữ liệu VizWiz mang tính chất ngẫu nhiên, góc chụp nghiêng, không đối xứng. |
| **2. Bố cục vật thể (Framing & Composition)** | **Photographer / Centered Object Bias**: Vật thể chính nằm giữa khung hình (Rule of Thirds / Center). | **Severe Truncation / Off-center**: Vật thể bị cắt cụt một nửa, lệch góc, góc chụp quá gần (extreme macro). | Mô hình học trên COCO bị phụ thuộc vào trọng tâm ảnh; mất khả năng nhận diện vật thể ở mép ảnh. |
| **3. Môi trường ánh sáng (Lighting)** | Ánh sáng tự nhiên, đủ sáng, tương phản chuẩn mực của ảnh được chọn lọc đăng tải Flickr. | Thiếu sáng, ngược sáng cửa sổ, bóng đổ của người chụp, lóa phản chiếu flash điện thoại. | Cần cơ chế tiền xử lý thích ứng ánh sáng hoặc kiến trúc chống chịu nhiễu phơi sáng. |
| **4. Độ ổn định quang học (Sensor Stability)** | Chụp có chủ đích, tay giữ yên, cảm biến máy ảnh tốt, chống rung quang học. | **Motion blur** do tay run hoặc vừa di chuyển vừa chụp; **defocus blur** do camera không lấy nét được vật thể gần. | Cần biểu diễn đặc trưng bất biến với độ mờ (blur-invariant representations). |
| **5. Che khuất góc nhìn thứ nhất (Egocentric Occlusion)** | Góc nhìn thứ ba (Third-person view), cảnh quan tách biệt với người chụp. | **Egocentric view**: Ngón tay, bàn tay người chụp cầm nắm che khuất $30-50\%$ bề mặt nhãn sản phẩm. | Mô hình COCO dễ nhận nhầm "bàn tay" là chủ thể chính thay vì nhận biết vật thể đang được cầm. |
| **6. Ý đồ sử dụng (User Intent)** | Miêu tả tổng thể bức tranh nghệ thuật ("Một người đàn ông đang cưỡi ngựa trên bãi biển"). | **Giải quyết câu hỏi cụ thể trong sinh hoạt**: "Đây là lon nước ngọt hay lon bia?", "Hộp thuốc này uống mấy viên?". | Đòi hỏi mô hình phải kết hợp OCR đọc chữ và suy luận tri thức thông thường (Commonsense Reasoning). |
| **7. Hành vi khi gặp ảnh hỏng (Failure Mode Handling)** | **Bắt buộc phải sinh câu miêu tả**; giả định mọi ảnh đều có nghĩa. | **Cần khả năng từ chối trả lời (Refusal Mechanism)** khi ảnh quá hỏng để tránh hậu quả nguy hiểm. | Mô hình chỉ huấn luyện trên COCO sẽ mắc lỗi **ảo tưởng nghiêm trọng (Severe Hallucination)** khi gặp ảnh mờ trong thực tế. |

### Hiểm họa Ảo tưởng (The Danger of Hallucination in Assistive Tech):
Nếu một người khiếm thị chụp ảnh một lọ thuốc bị nhòe mờ:
- **Hành vi đúng đắn (VizWiz paradigm)**: Mô hình phải trả lời: *"Chất lượng ảnh quá mờ để nhận biết chính xác loại thuốc, vui lòng chụp lại với khoảng cách xa hơn và bật đèn."*
- **Hành vi tai hại (COCO paradigm)**: Do được huấn luyện luôn luôn sinh caption trên ảnh rõ nét, mô hình COCO sẽ "bịa" ra một câu miêu tả ngẫu nhiên: *"A bottle of vitamin pills on a counter"*. Nếu thực tế đó là lọ thuốc độc hoặc thuốc tim mạch liều cao, sự ảo tưởng này sẽ gây nguy hiểm trực tiếp đến tính mạng người dùng!

---

## PHẦN 7: BỘ MẪU ẢNH ĐỐI SÁNH TIÊU BIỂU (QUALITATIVE COMPARISON)

Để phục vụ thuyết trình và trình bày trực quan trong khóa luận, 8 mẫu ảnh tiêu biểu đã được trích xuất và lưu tại biểu đồ `07_qualitative_samples_grid.png`:

### Nhóm 1: Chuẩn mực Benchmark COCO (Curated Photography)
1. **`COCO_val2014_000000391895.jpg`**:
   - *Hình ảnh*: Người đàn ông đội mũ bảo hiểm đỏ lái xe máy trên đường đất, nền núi mây hùng vĩ.
   - *Đặc điểm*: Ánh sáng hoàn hảo, vật thể cân đối giữa khung hình, độ sâu trường ảnh tốt.
   - *Caption mẫu*: *"A man with a red helmet on a moped on a dirt road."*
2. **`COCO_val2014_000000522418.jpg`**:
   - *Hình ảnh*: Người phụ nữ đeo lưới trùm tóc đang cắt chiếc bánh kem trắng lớn.
   - *Đặc điểm*: Rõ nét từng chi tiết dao cắt, bố cục trong bếp chuẩn mực.
   - *Caption mẫu*: *"A woman wearing a hair net cutting a large sheet cake."*
3. **`COCO_val2014_000000184613.jpg`**:
   - *Hình ảnh*: Cậu bé cầm ô hoa đứng vuốt ve sừng con bò Tây Tạng (yak).
   - *Đặc điểm*: Đầy đủ bối cảnh văn hóa, màu sắc sống động, độ tương phản sắc nét.
   - *Caption mẫu*: *"A young boy with an umbrella touching the horn of a yak."*
4. **`COCO_val2014_000000581357.jpg`**:
   - *Hình ảnh*: Bàn ăn với các đĩa thức ăn và ly rượu vang được sắp xếp ngăn nắp.
   - *Đặc điểm*: Ảnh chụp tĩnh vật trong nhà hàng, ánh sáng ấm, cân bằng trắng chuẩn.
   - *Caption mẫu*: *"A table with plates of food and wine glasses."*

### Nhóm 2: Thực tế Đời thực VizWiz (In-The-Wild Assistive Reality)
1. **`VizWiz_val_00000001.jpg` (Handheld Medicine & OCR)**:
   - *Hình ảnh*: Bàn tay người khiếm thị đang cầm một chai thuốc nhỏ màu đen, nhãn chữ *"NIGHT TIME"*.
   - *Đặc điểm*: Ngón tay che một phần thân chai, góc chụp thẳng đứng, người dùng cần biết đây là thuốc gì.
   - *Caption mẫu*: *"A person is holding a small black bottle of NIGHT TIME medicine."*
2. **`VizWiz_val_00000000.jpg` (Digital Screen & Assistive Prompt)**:
   - *Hình ảnh*: Màn hình laptop chụp nghiêng, hiển thị hộp thoại thông báo sửa lỗi hệ thống máy tính.
   - *Đặc điểm*: Phản chiếu bóng đèn trên màn hình LCD, chữ nhỏ, góc chụp chéo từ trên xuống.
   - *Caption mẫu*: *"A partial computer screen showing a repair prompt popup."*
3. **`VizWiz_val_00000013.jpg` (Severe Motion Blur & Hand Shake)**:
   - *Hình ảnh*: Một vật thể không thể xác định trên mặt bàn, vệt mờ chuyển động kéo dài khắp khung hình.
   - *Đặc điểm*: Camera di chuyển khi bấm chụp, mất hoàn toàn biên cạnh nhận dạng vật thể.
   - *Caption mẫu*: *"A blurry photo of an unrecognizable object on a table."*
4. **`VizWiz_val_00000006.jpg` (Complete Quality Failure - 100% Precanned)**:
   - *Hình ảnh*: Hình ảnh bị đen kịt, cảm biến bị che kín hoàn toàn hoặc chụp trong buồng tối không có ánh sáng.
   - *Đặc điểm*: Cả 5 người gán nhãn độc lập đều đồng thuận tuyệt đối là không thể nhìn thấy bất kỳ nội dung nào.
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
- **Thư mục Đồ thị**: `c:\Users\Admin\Downloads\HGAI_data\eda_results\figures/`
  1. `01_dataset_splits_comparison.png`: So sánh phân chia tập dữ liệu.
  2. `02_resolution_aspect_ratio.png`: Phân bố độ phân giải và tỷ lệ khung hình.
  3. `03_caption_length_and_tokens.png`: Phân bố độ dài từ và từ vựng đặc trưng.
  4. `04_coco_category_distribution.png`: Phân bố đối tượng COCO và sự áp đảo của con người.
  5. `05_vizwiz_quality_and_text_issues.png`: Tỷ lệ ảnh suy giảm chất lượng, chữ viết và độ sắc nét quang học.
  6. `06_domain_gap_characteristics.png`: Biểu đồ radar 7 chiều đối sánh Domain Gap.
  7. `07_qualitative_samples_grid.png`: Lưới ảnh đối sánh trực quan các trường hợp tiêu biểu.
- **Thư mục Bảng số liệu CSV**: `c:\Users\Admin\Downloads\HGAI_data\eda_results\tables/`
- **File Dữ liệu Thống kê Đầy đủ (JSON)**: `c:\Users\Admin\Downloads\HGAI_data\eda_results\eda_summary_metrics.json`
- **Script Thực thi Tái lập (Reproducible Python Script)**: `c:\Users\Admin\Downloads\HGAI_data\eda_coco_vs_vizwiz.py`
