# BÁO CÁO THỐNG KÊ & CHUẨN BỊ DỮ LIỆU GIAI ĐOẠN 1 (THÀNH VIÊN A)
**Bộ dữ liệu:** `DsPCBSD+` (10.259 ảnh AOI bo mạch in PCB, 20.276 Bounding Box, 9 lớp lỗi)

---

## 1. Kết quả Kiểm toán Nhãn YOLO & Chia tập cố định (`SEED = 42`)

Do bộ dữ liệu gốc trên Kaggle chỉ chia sẵn `train` và `val` (chưa tách tập `test` độc lập), Thành viên A đã gom toàn bộ 10.259 ảnh hợp lệ và chia phân tầng cố định **Train (70%) / Val (15%) / Test (15%)** với `SEED = 42`, kiểm chứng bằng hàm `pcb_utils.check_dataset()`:

| Tập dữ liệu (`split`) | Số lượng ảnh (`images`) | Số lượng BBox (`boxes`) | File thiếu nhãn (`missing`) | File nhãn rỗng (`empty`) | Dòng nhãn lỗi (`invalid`) | Rò rỉ giữa các tập (`leaks`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train` (70%)** | 7.181 | 14.138 | 0 | 0 | 0 | **0** |
| **`val` (15%)** | 1.539 | 3.037 | 0 | 0 | 0 | **0** |
| **`test` (15%)** | 1.539 | 3.101 | 0 | 0 | 0 | **0** |
| **Tổng cộng** | **10.259** | **20.276** | **0** | **0** | **0** | **0** |

---

## 2. Thống kê số lượng Ảnh & Kích thước Bounding Box theo 9 Lớp lỗi PCB

| Class ID | Mã lớp | Tên đầy đủ (Mô tả lỗi PCB) | Số ảnh chứa lỗi | Tổng số BBox | Tỷ trọng BBox (%) | Kích thước TB trên ảnh gốc $226\times 226$ ($W\times H$ px) | Tỷ lệ Small ($<32\text{ px}$ ở khung 640) (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **`SH`** | Short (Ngắn mạch) | 604 | 915 | 4.51% | $30.0 \times 27.0$ | 0.1% |
| **1** | **`SP`** | Spur (Gai đồng) | 2.186 | 4.584 | 22.61% | $16.3 \times 14.1$ | **35.0%** |
| **2** | **`SC`** | Spurious Copper (Đồng thừa) | 1.317 | 1.593 | 7.86% | $22.2 \times 23.9$ | 10.0% |
| **3** | **`OP`** | Open (Hở mạch) | 1.420 | 1.770 | 8.73% | $26.0 \times 27.5$ | 5.4% |
| **4** | **`MB`** | Mouse Bite (Khuyết cạnh/Chân chuột) | 1.873 | 2.529 | 12.47% | $16.5 \times 15.7$ | **33.0%** |
| **5** | **`HB`** | Hole Breakout (Lệch/Vỡ lỗ khoan) | 1.290 | 2.883 | 14.22% | $41.1 \times 41.1$ | 0.0% |
| **6** | **`CS`** | Conductor Scratch (Xước đường mạch) | 1.487 | 2.490 | 12.28% | $85.8 \times 74.4$ | 2.7% |
| **7** | **`CFO`** | Conductor Foreign Object (Dị vật trên mạch) | 1.352 | 1.832 | 9.04% | $36.9 \times 37.9$ | 11.1% |
| **8** | **`BMFO`** | Base Material Foreign Object (Dị vật trên nền) | 1.484 | 1.680 | 8.29% | $27.2 \times 29.3$ | 17.7% |

---

## 3. Phân bố số lượng BBox theo Lớp × Tập (`train / val / test`)

| Lớp lỗi | `train` | `val` | `test` | `total` |
| :---: | :---: | :---: | :---: | :---: |
| **`SH`** | 652 | 146 | 117 | **915** |
| **`SP`** | 3.190 | 663 | 731 | **4.584** |
| **`SC`** | 1.116 | 238 | 239 | **1.593** |
| **`OP`** | 1.221 | 276 | 273 | **1.770** |
| **`MB`** | 1.768 | 365 | 396 | **2.529** |
| **`HB`** | 2.030 | 417 | 436 | **2.883** |
| **`CS`** | 1.723 | 388 | 379 | **2.490** |
| **`CFO`** | 1.253 | 301 | 278 | **1.832** |
| **`BMFO`** | 1.185 | 243 | 252 | **1.680** |

---

## 4. Phân bố kích thước BBox theo chuẩn COCO (quy đổi về khung $640\times 640$)

| Nhóm kích thước (tại $640\times 640$) | Số lượng BBox | Tỷ lệ (%) | Nhận xét kỹ thuật cho Giai đoạn 3 |
| :--- | :---: | :---: | :--- |
| **Small ($< 32\times 32\text{ px}$)** | 3.265 | **16.1%** | Tập trung mạnh ở lớp **`SP` (35.0%)** và **`MB` (33.0%)** (trên ảnh gốc $226\times 226$ chỉ rộng $\sim 15\text{ px}$). Đây là cơ sở trực tiếp để bổ sung **nhánh phát hiện độ phân giải cao `P2/4` ($160\times 160$)** và **hàm loss `NWD-WIoU`** ở Giai đoạn 3. |
| **Medium ($32\times 32 .. 96\times 96\text{ px}$)** | 10.482 | **51.7%** | Chiếm đa số trên các lớp `OP`, `SC`, `SH`, `BMFO`, `CFO`. |
| **Large ($\ge 96\times 96\text{ px}$)** | 6.529 | **32.2%** | Chủ yếu thuộc lớp vết xước dài `CS` và vỡ lỗ khoan `HB`. |
