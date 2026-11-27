# BÁO CÁO KẾT QUẢ GIAI ĐOẠN 2 (THÀNH VIÊN A): HUẤN LUYỆN BASELINE `YOLOv8s` & `YOLO11s`
**Bộ dữ liệu:** `DsPCBSD+` (`SEED = 42`: Train 7.181 ảnh | Val 1.539 ảnh | Test 1.539 ảnh)  
**Cấu hình chung đã chốt:** `IMGSZ = 640`, `EPOCHS = 50`, `BATCH = 16`, `SEED = 42`, `cos_lr = True`

---

## 1. Bảng Ước lượng Thời gian / Epoch & Ngân sách GPU (`gpu_budget_estimate.csv`)

| Mô hình | Số Epoch chạy | Thời gian TB / Epoch (s) | Tổng thời gian Train 50 ep (phút) | Ước tính chạy 80 ep GĐ 4 (giờ) | VRAM Đỉnh (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`yolov8s_640_s42`** | 50 | 99.5 s | 84.5 phút (~1.41h) | **2.21 giờ** | 3.53 GB |
| **`yolo11s_640_s42`** | 50 | 96.8 s | 82.3 phút (~1.37h) | **2.15 giờ** | 4.07 GB |

---

## 2. Kết quả Đánh giá trên Tập Validation (`val` — 1.539 ảnh, dùng để chọn Baseline tốt nhất)

| Model | P | R | F1 | mAP50 | mAP50-95 | FPS | Params (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolov8s_640_s42`** | **0.8255** | 0.8017 | **0.8134** | 0.8459 | 0.5071 | 50.26 | 11.14 |
| **`yolo11s_640_s42`** | 0.8154 | **0.8105** | 0.8129 | **0.8552** | **0.5132** | **58.31** | **9.43** |

> **Kết luận chọn Baseline cho Giai đoạn 3**: Chọn **`YOLO11s`** làm nền tảng để cải tiến ở Giai đoạn 3 vì đạt **`mAP50-95 (val) = 0.5132`** và **`mAP50 (val) = 0.8552`** cao hơn `YOLOv8s`, đồng thời nhẹ hơn (`9.43M` vs `11.14M` tham số) và nhanh hơn (**`58.31 FPS`** vs `50.26 FPS`).

---

## 3. Kết quả Đánh giá trên Tập Test Độc lập (`test` — 1.539 ảnh) & Đối chiếu Nhóm 14

| Model | P | R | F1 | mAP50 | mAP50-95 | So với Nhóm 14 (`0.4276`) | FPS | Params (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Tham chiếu Nhóm 14 (`YOLOv8s`, 512, 20ep)* | — | — | — | — | *0.4276* | *Mốc gốc* | — | *11.14* |
| **`yolov8s_640_s42`** | **0.8278** | 0.7919 | 0.8095 | 0.8443 | 0.5032 | **+0.0756 (+7.56%)** | 50.26 | 11.14 |
| **`yolo11s_640_s42`** | 0.8007 | **0.8229** | **0.8117** | **0.8535** | **0.5123** | **+0.0847 (+8.47%)** | **58.31** | **9.43** |

---

## 4. Độ chính xác `AP@0.5` theo Từng Lớp lỗi PCB trên Tập Test (`A_baselines_per_class_ap50.csv`)

| Mã lớp | Mô tả lỗi PCB | `yolov8s_640_s42` | `yolo11s_640_s42` | Chênh lệch (`YOLO11s - YOLOv8s`) | Nhận xét định hướng cho Giai đoạn 3 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`SH`** | Short (Ngắn mạch) | **0.9484** | 0.9447 | -0.0037 | Đạt độ chính xác rất cao (>94%) |
| **`SP`** | Spur (Gai đồng - 35% Small) | 0.8287 | **0.8430** | **+0.0143** | Lỗi kích thước rất nhỏ ($\sim 16\times 14\text{ px}$), cần nhánh `P2/4` |
| **`SC`** | Spurious Copper (Đồng thừa) | 0.8131 | **0.8507** | **+0.0376** | `YOLO11s` cải thiện mạnh (+3.76%) |
| **`OP`** | Open (Hở mạch) | **0.9017** | 0.8943 | -0.0074 | Đạt ~90% AP50 |
| **`MB`** | Mouse Bite (Khuyết cạnh - 33% Small) | 0.8331 | **0.8390** | **+0.0059** | Lỗi khuyết rìa mạch nhỏ ($\sim 16\times 15\text{ px}$), cần nhánh `P2/4` + `NWD Loss` |
| **`HB`** | Hole Breakout (Lệch lỗ khoan) | 0.9731 | **0.9811** | **+0.0080** | Lớp dễ nhận diện nhất (~98.1% AP50) |
| **`CS`** | Conductor Scratch (Xước đường mạch) | 0.7172 | **0.7380** | **+0.0208** | Vết xước mảnh dài, tương phản thấp (`AP50 ~ 73.8%`), cần thêm `Attention` |
| **`CFO`** | Conductor Foreign Object (Dị vật mạch) | 0.7013 | **0.7035** | **+0.0022** | Lớp khó nhất do hình dạng dị vật đa dạng (`AP50 ~ 70.3%`) |
| **`BMFO`** | Base Material Foreign Object (Dị vật nền) | 0.8825 | **0.8869** | **+0.0044** | Đạt ~88.7% AP50 |
| **Trung bình (`mAP50`)** | **Toàn bộ 9 lớp lỗi** | **0.8443** | **0.8535** | **+0.0092** | **`YOLO11s` vượt trội ở 7/9 lớp lỗi** |
