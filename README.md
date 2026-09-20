# Hệ thống Cảnh báo sớm Sinh viên có Nguy cơ Học tập kém bằng Trí tuệ Nhân tạo

> Đề tài Nghiên cứu Khoa học cấp Khoa | Ứng dụng Machine Learning và Large Language Model trong dự báo và can thiệp sớm rủi ro học tập

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Bối cảnh & Tính cấp thiết](#2-bối-cảnh--tính-cấp-thiết)
3. [Mục tiêu nghiên cứu](#3-mục-tiêu-nghiên-cứu)
4. [Câu hỏi nghiên cứu & Giả thuyết](#4-câu-hỏi-nghiên-cứu--giả-thuyết)
5. [Tổng quan tài liệu](#5-tổng-quan-tài-liệu)
6. [Bộ dữ liệu](#6-bộ-dữ-liệu)
7. [Phương pháp luận](#7-phương-pháp-luận)
8. [Khung phân loại mức độ rủi ro (A/B/C)](#8-khung-phân-loại-mức-độ-rủi-ro-abc)
9. [Kiến trúc hệ thống](#9-kiến-trúc-hệ-thống)
10. [Cấu trúc thư mục dự án](#10-cấu-trúc-thư-mục-dự-án)
11. [Cài đặt & Hướng dẫn sử dụng](#11-cài-đặt--hướng-dẫn-sử-dụng)
12. [Lộ trình phát triển](#12-lộ-trình-phát-triển)
13. [Trạng thái hiện tại](#13-trạng-thái-hiện-tại)
14. [Nhóm thực hiện](#14-nhóm-thực-hiện)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)
16. [Giấy phép & Đóng góp](#16-giấy-phép--đóng-góp)

---

## 1. Giới thiệu

**Tên đề tài:** Xây dựng hệ thống cảnh báo sớm sinh viên có nguy cơ học tập kém bằng trí tuệ nhân tạo

Đề tài xây dựng một hệ thống trí tuệ nhân tạo có khả năng **dự báo sớm** sinh viên có nguy cơ trượt học phần hoặc bị cảnh báo học vụ, dựa trên dữ liệu học tập (điểm số, chuyên cần) và dữ liệu hành vi trên hệ thống quản lý học tập (LMS). Hệ thống hướng tới hỗ trợ giảng viên và cố vấn học tập phát hiện sớm các trường hợp cần can thiệp, qua đó nâng cao tỷ lệ hoàn thành học phần và giảm tỷ lệ thôi học.
---

## 2. Bối cảnh & Tính cấp thiết

### 2.1. Thực trạng vấn đề

- Tỷ lệ sinh viên bỏ học, trượt học phần hoặc bị cảnh báo học vụ vẫn là một thách thức phổ biến ở giáo dục đại học, không chỉ tại Việt Nam mà trên toàn thế giới. Việc phát hiện các trường hợp này thường diễn ra **quá muộn** — thường chỉ sau khi có kết quả thi cuối kỳ — khiến các biện pháp can thiệp (gia sư, tư vấn học tập, học lại) không còn kịp thời.
- Hệ thống LMS hiện đại lưu trữ một lượng lớn dữ liệu hành vi học tập (lượt truy cập, thời gian tương tác, mức độ hoàn thành bài tập) nhưng phần lớn **chưa được khai thác** để phục vụ công tác cảnh báo sớm và cố vấn học tập cá nhân hóa.
- Việc cá nhân hóa hỗ trợ học tập — tức là mỗi sinh viên nhận được một khuyến nghị can thiệp phù hợp với đặc điểm rủi ro riêng — vẫn còn hạn chế do khối lượng công việc lớn nếu thực hiện thủ công bởi giảng viên/cố vấn.

### 2.2. Khoảng trống nghiên cứu

- Phần lớn nghiên cứu hiện có tập trung vào **một bộ dữ liệu đơn lẻ**, thiếu kiểm định khả năng khái quát hóa (generalization) của mô hình dự báo khi áp dụng sang bối cảnh giáo dục khác.
- Việc tích hợp **mô hình ngôn ngữ lớn (LLM)** để tự động sinh khuyến nghị hỗ trợ học tập cá nhân hóa — thay vì chỉ dừng ở dự báo — vẫn là hướng khá mới, đặc biệt trong bối cảnh giáo dục đại học Việt Nam.
- Đa số hệ thống cảnh báo sớm hiện tại dùng phân loại **nhị phân** (rủi ro / không rủi ro), trong khi các hệ thống cảnh báo sớm trưởng thành trên thế giới (K-12 và đại học) thường dùng khung **đa mức độ (3 tầng: Cao/Trung bình/Thấp)** để phân bổ nguồn lực can thiệp hợp lý hơn.

### 2.3. Đóng góp dự kiến của đề tài

1. Xây dựng và so sánh mô hình dự báo rủi ro học tập đa lớp (3 mức A/B/C) trên **hai bộ dữ liệu độc lập**, kiểm định khả năng khái quát hóa xuyên bộ dữ liệu (cross-dataset generalization).
2. Đề xuất khung phân loại 3 mức rủi ro có căn cứ khoa học (không chọn ngưỡng tùy tiện), kế thừa từ tổng quan tài liệu đa cấp học (phổ thông và đại học).
3. Tích hợp LLM sinh khuyến nghị hỗ trợ học tập cá nhân hóa, có kiểm soát chất lượng đầu ra (template hóa, tiến tới RAG).
4. Xây dựng dashboard trực quan hỗ trợ giảng viên/cố vấn theo dõi và ra quyết định can thiệp.

---

## 3. Mục tiêu nghiên cứu

| # | Mục tiêu |
|---|---|
| MT1 | Xây dựng mô hình học máy (Random Forest, XGBoost) dự báo sớm mức độ rủi ro học tập của sinh viên (3 mức A/B/C) dựa trên dữ liệu học tập và hành vi LMS |
| MT2 | Kiểm định khả năng khái quát hóa của mô hình giữa hai bộ dữ liệu độc lập (OULAD và UCI ID 697) |
| MT3 | Tích hợp mô hình ngôn ngữ lớn (LLM) để tự động sinh khuyến nghị học tập phù hợp cho từng nhóm/mức độ rủi ro |
| MT4 | Xây dựng Dashboard trực quan hiển thị danh sách và mức độ rủi ro của sinh viên, hỗ trợ giảng viên/cố vấn học tập |
| MT5 | Đánh giá hiệu quả mô hình (Accuracy, Precision, Recall, F1-score) và tính khả dụng của hệ thống qua khảo sát người dùng thực tế |

---

## 4. Câu hỏi nghiên cứu & Giả thuyết

### RQ1
Các đặc trưng dự báo nguy cơ học tập kém (điểm giữa kỳ, chuyên cần/tương tác LMS) có mối quan hệ và tầm quan trọng tương đồng giữa hai bối cảnh giáo dục khác nhau (Open University Anh và các trường đại học Bồ Đào Nha) hay không?

**H1:** Mức độ tương tác/chuyên cần và điểm quá trình là các đặc trưng có tầm quan trọng cao ở cả hai bộ dữ liệu, bất kể khác biệt về bối cảnh giáo dục.

### RQ2
Mô hình học máy (Random Forest/XGBoost) huấn luyện trên một bộ dữ liệu có giữ được hiệu năng dự báo chấp nhận được khi kiểm định trên bộ dữ liệu còn lại (cross-dataset generalization) hay không?

**H2:** Hiệu năng mô hình khi kiểm định chéo (cross-dataset) sẽ giảm so với khi kiểm định cùng bộ dữ liệu huấn luyện, nhưng vẫn ở mức chấp nhận được.

### RQ3
Việc kết hợp/tổng hợp cả hai bộ dữ liệu (qua tập đặc trưng chung) có cải thiện độ chính xác và độ ổn định của mô hình so với chỉ dùng một bộ đơn lẻ hay không?

**H3:** Mô hình huấn luyện trên dữ liệu tổng hợp từ cả hai nguồn sẽ có khả năng khái quát hóa tốt hơn mô hình chỉ huấn luyện trên một bộ.

> Các câu hỏi/giả thuyết trên sẽ được **tinh chỉnh dựa trên kết quả thực nghiệm thật** sau khi hoàn thành notebook phân tích (xem mục 13).

---

## 5. Tổng quan tài liệu

### 5.1. Cảnh báo sớm học tập ở bậc phổ thông (K-12)
Các hệ thống cảnh báo sớm (Early Warning System - EWS) ở bậc phổ thông phổ biến sử dụng khung phân loại 3 mức rủi ro theo màu đèn giao thông (Đỏ/Vàng/Xanh - Cao/Trung bình/Thấp), dựa trên các nhóm chỉ số chính: điểm số, tỷ lệ vắng mặt, và hạnh kiểm/quan sát của giáo viên.

### 5.2. Cảnh báo sớm & dự báo dropout ở bậc đại học
- Nhiều nghiên cứu sử dụng bộ dữ liệu OULAD với các thuật toán cây quyết định, Random Forest, XGBoost để dự báo kết quả học tập và nguy cơ bỏ học, trong đó XGBoost thường cho kết quả tốt (Accuracy ~86-87% trong một số nghiên cứu).
- Kết quả so sánh Random Forest và XGBoost giữa các nghiên cứu **không đồng nhất** — có nghiên cứu cho thấy Random Forest vượt trội hơn, có nghiên cứu khác lại cho kết quả ngược lại — cho thấy hiệu năng phụ thuộc nhiều vào đặc điểm cụ thể của từng bộ dữ liệu, củng cố tính cần thiết của việc tự thực nghiệm so sánh trong đề tài này.

### 5.3. Nghiên cứu khai thác đồng thời nhiều bộ dữ liệu (OULAD + UCI Student Performance/697)
- Một số nghiên cứu gần đây (2025-2026) đã khai thác đồng thời OULAD và bộ dữ liệu UCI Student Performance để phân tích các yếu tố ảnh hưởng đến kết quả học tập ở nhiều bối cảnh giáo dục.
- Nghiên cứu về khung AI lai (hybrid) kiểm định trên đồng thời UCI Student Performance, OULAD và NELS:88 cho thấy việc kiểm định mô hình trên nhiều bộ dữ liệu là hướng tiếp cận đã được công nhận trong lĩnh vực.
- Nghiên cứu về chuyển giao mô hình xuyên tổ chức (cross-institutional transfer learning) trong dự báo bỏ học sinh viên cho thấy việc chuyển giao mô hình giữa các cơ sở đào tạo là khả thi, không đánh đổi đáng kể về hiệu năng hay tính công bằng.

### 5.4. Tích hợp LLM sinh khuyến nghị cá nhân hóa
- Các nghiên cứu gần đây đã thử nghiệm tích hợp LLM (GPT, Llama) vào hệ thống cảnh báo sớm để tự động sinh phản hồi/khuyến nghị hỗ trợ học tập cá nhân hóa.
- Một rủi ro được ghi nhận phổ biến là hiện tượng "hallucination" (LLM sinh thông tin nghe hợp lý nhưng sai lệch thực tế) — gợi ý hướng khắc phục bằng kỹ thuật **structured prompting** (đưa dữ liệu có cấu trúc vào prompt) hoặc **RAG** (tra cứu tài liệu thật trước khi sinh nội dung).

> **Ghi chú:** Danh sách tài liệu tham khảo đầy đủ (tên tác giả, DOI) xem tại [mục 15](#15-tài-liệu-tham-khảo). Cần bổ sung, xác minh đầy đủ thông tin trích dẫn chuẩn APA/IEEE trước khi đưa vào báo cáo chính thức nộp Khoa.

---

## 6. Bộ dữ liệu

Đề tài sử dụng **hai bộ dữ liệu công khai độc lập** để tăng độ tin cậy và kiểm định khả năng khái quát hóa của mô hình:

### 6.1. OULAD — Open University Learning Analytics Dataset

| Thông tin | Chi tiết |
|---|---|
| Nguồn | Open University (Anh), qua Kaggle hoặc [analyse.kmi.open.ac.uk](https://analyse.kmi.open.ac.uk/open_dataset) |
| Quy mô | ~32,000 sinh viên, 7 học phần, nhiều kỳ học |
| Số bảng | 7 file CSV: `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle` |
| Nhãn gốc | `final_result`: Withdrawn / Fail / Pass / Distinction |
| Đặc trưng hành vi | Log tương tác VLE (Virtual Learning Environment) theo ngày — cơ sở để xây đặc trưng "mức độ tương tác LMS" |

### 6.2. UCI ID 697 — Predict Students' Dropout and Academic Success

| Thông tin | Chi tiết |
|---|---|
| Nguồn | [archive.ics.uci.edu/dataset/697](https://archive.ics.uci.edu/dataset/697) (Realinho et al.) |
| Quy mô | ~4,400 sinh viên, các trường đại học Bồ Đào Nha |
| Số bảng | 1 file CSV, 36 thuộc tính |
| Nhãn gốc | `Target`: Dropout / Enrolled / Graduate |
| Đặc trưng | Điểm số theo học kỳ, số tín chỉ đăng ký/đạt, thông tin nhân khẩu học, kinh tế-xã hội |

> **Lý do chọn UCI ID 697 thay vì ID 320 (Cortez):** ID 697 là dữ liệu bậc **đại học** với nhãn kiểu **dropout** — tương thích cao với OULAD về đối tượng và bài toán, phù hợp cho việc kiểm định chéo (cross-dataset) hơn nhiều so với ID 320 (dữ liệu học sinh THPT, nhãn là điểm số liên tục).

---

## 7. Phương pháp luận

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  1. TỔNG QUAN    │────▶│  2. THU THẬP &    │────▶│  3. TIỀN XỬ LÝ &     │
│     TÀI LIỆU     │     │     DỮ LIỆU       │     │     FEATURE ENG.     │
│  (đa cấp học)    │     │  (OULAD + UCI697) │     │  (Pandas)             │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                              │
┌─────────────────┐     ┌──────────────────┐     ┌──────────▼──────────┐
│ 6. DASHBOARD &   │◀────│ 5. TÍCH HỢP LLM   │◀────│  4. MÔ HÌNH HÓA &    │
│    TRIỂN KHAI    │     │    (+ RAG)         │     │     ĐÁNH GIÁ         │
│   (Streamlit)     │     │                    │     │  (RF, XGBoost)       │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                              │
                                                   ┌──────────▼──────────┐
                                                   │ 7. KIỂM ĐỊNH CHÉO    │
                                                   │    (Cross-dataset)   │
                                                   └──────────────────────┘
```

### 7.1. Tiền xử lý & xây dựng đặc trưng (Feature Engineering)

**OULAD:**
- `total_clicks`: tổng lượt tương tác VLE (đại diện mức độ tham gia học tập)
- `avg_tma_score`: điểm trung bình các bài đánh giá quá trình (đại diện "điểm giữa kỳ")
- `num_of_prev_attempts`, `studied_credits`: đặc trưng nhân khẩu học/học vụ sẵn có

**UCI 697:**
- Điểm trung bình học kỳ 1/2, số tín chỉ đăng ký/đạt, độ tuổi nhập học

### 7.2. Mô hình hóa
- Huấn luyện và so sánh **Random Forest** và **XGBoost** trên từng bộ dữ liệu.
- Xử lý mất cân bằng lớp (`class_weight='balanced'`).
- Đánh giá bằng **Accuracy, Precision, Recall, F1-score** (macro-average do bài toán đa lớp).
- Phân tích **feature importance** để xác định yếu tố ảnh hưởng lớn nhất.

### 7.3. Kiểm định chéo (Cross-dataset validation)
- Ánh xạ các đặc trưng có ý nghĩa tương đồng giữa 2 bộ dữ liệu (điểm quá trình, khối lượng học tập).
- Chuẩn hóa (z-score) để đưa về cùng thang đo.
- Huấn luyện trên bộ này, kiểm định trên bộ kia — trả lời RQ2/RQ3.

---

## 8. Khung phân loại mức độ rủi ro (A/B/C)

Đề tài áp dụng khung phân loại **3 mức rủi ro** (thay vì nhị phân), kế thừa từ mô hình phổ biến trong các hệ thống cảnh báo sớm giáo dục:

| Mức | Ý nghĩa | Ánh xạ từ OULAD (`final_result`) | Ánh xạ từ UCI 697 (`Target`) |
|---|---|---|---|
| **A** | Nguy cơ **Cao** — cần can thiệp ngay | Withdrawn, Fail | Dropout |
| **B** | Nguy cơ **Trung bình** — cần theo dõi | Pass (điểm quá trình < ngưỡng median của nhóm Pass) | Enrolled |
| **C** | Nguy cơ **Thấp** — bình thường | Pass (điểm quá trình ≥ ngưỡng median), Distinction | Graduate |

> **Nguyên tắc quan trọng:** Ngưỡng phân chia B/C trong nhóm Pass của OULAD được tính từ **giá trị median thực tế** của chính dữ liệu (`avg_tma_score`), **không chọn tùy tiện**. Đây là điểm cần nhấn mạnh khi bảo vệ — logic giữa bước Tổng quan tài liệu (khung 3 mức đã được kiểm chứng) và bước Dự báo (áp dụng khung đó có căn cứ dữ liệu) phải nhất quán.

---

## 9. Kiến trúc hệ thống

### 9.1. Sơ đồ tổng thể

```
 Dữ liệu (OULAD, UCI 697)
        │
        ▼
 ┌─────────────────┐
 │  Data Pipeline    │   Pandas: làm sạch, join, feature engineering
 └────────┬─────────┘
          ▼
 ┌─────────────────┐
 │  Mô hình ML        │   Random Forest / XGBoost → Nhãn rủi ro A/B/C
 └────────┬─────────┘
          ▼
 ┌─────────────────┐        ┌───────────────────────────┐
 │  Module LLM        │◀─────│  (Mở rộng) RAG:              │
 │  Sinh khuyến nghị   │      │  Quy chế học vụ, hướng dẫn   │
 │  học tập cá nhân    │      │  học tập của trường            │
 └────────┬─────────┘        └───────────────────────────┘
          ▼
 ┌─────────────────┐
 │  Dashboard          │   Streamlit — hiển thị danh sách rủi ro,
 │                     │   biểu đồ, khuyến nghị AI cho từng SV
 └─────────────────┘
```

### 9.2. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Xử lý dữ liệu | Python, Pandas, NumPy |
| Mô hình học máy | scikit-learn (Random Forest), XGBoost |
| Trực quan hóa (phân tích) | Matplotlib, Seaborn |
| LLM sinh khuyến nghị | API Claude/GPT (prompt template hóa; RAG nếu đủ thời gian) |
| RAG (mở rộng) | FAISS/ChromaDB (vector store), sentence-transformers (embedding) |
| Dashboard | Streamlit |
| Môi trường phát triển | Jupyter Notebook (Anaconda) / Google Colab |

---

## 10. Cấu trúc thư mục dự án

```
NCKH_CanhBaoSomHocVu/
│
├── data/
│   ├── oulad/                      # 7 file CSV gốc của OULAD
│   ├── uci_697/                    # File CSV gốc của UCI 697
│   └── processed/                  # Dữ liệu đã qua tiền xử lý (sinh ra từ notebook)
│
├── notebooks/
│   └── OULAD_UCI697_Phan_tich_Tuan1.ipynb   # EDA + mô hình hóa + kiểm định chéo
│
├── src/                             # (Dự kiến, khi module hóa code từ notebook)
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluate.py
│
├── llm_module/                      # (Mở rộng) Module LLM/RAG
│   ├── prompts/
│   └── knowledge_base/              # Tài liệu quy chế học vụ cho RAG
│
├── dashboard/                       # (Mở rộng) Ứng dụng Streamlit
│   └── app.py
│
├── docs/
│   ├── de_cuong_goc.docx            # Đề cương gốc do GVHD cung cấp
│   ├── tai_lieu_trao_doi_GVHD.docx  # Tài liệu trao đổi, mở rộng phạm vi
│   └── related_work.md              # Tổng hợp chi tiết tài liệu tham khảo
│
├── reports/                         # Báo cáo tuần/tháng, kết quả thực nghiệm
│
├── README.md                        # File này
└── requirements.txt                 # Danh sách thư viện Python cần cài
```

---

## 11. Cài đặt & Hướng dẫn sử dụng

### 11.1. Yêu cầu môi trường
- Python ≥ 3.10
- Jupyter Notebook (khuyến nghị qua Anaconda) hoặc Google Colab

### 11.2. Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
pip install kagglehub[pandas-datasets]   # tải OULAD trực tiếp từ Kaggle
pip install ucimlrepo                     # tải UCI 697 trực tiếp qua API
```

### 11.3. Tải dữ liệu

**Cách 1 — Tự động (khuyến nghị):**
```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "anlgrbz/student-demographics-online-education-dataoulad",
    "studentInfo.csv"
)
```

**Cách 2 — Thủ công:** Tải 7 file CSV của OULAD và 1 file CSV của UCI 697, đặt vào đúng thư mục `data/oulad/` và `data/uci_697/` như cấu trúc ở mục 10.

### 11.4. Chạy notebook
1. Mở Jupyter Notebook: `jupyter notebook`
2. Mở file `notebooks/OULAD_UCI697_Phan_tich_Tuan1.ipynb`
3. **Kernel → Restart Kernel and Run All Cells** để đảm bảo chạy đúng thứ tự, tránh lỗi `NameError` do biến chưa được định nghĩa.

---

## 12. Lộ trình phát triển

Lộ trình được chia theo 3 cấp độ mở rộng, để nhóm chủ động lựa chọn tùy theo thời gian còn lại:

### Cấp độ 1 — Lõi bắt buộc (bám sát đề cương GVHD)
- [x] Xây dựng notebook EDA + mô hình hóa OULAD và UCI 697
- [ ] Chạy thực nghiệm hoàn chỉnh, thu thập số liệu Accuracy/Precision/Recall/F1
- [ ] So sánh Random Forest vs XGBoost trên cả 2 bộ
- [ ] Dashboard cơ bản bằng Streamlit

### Cấp độ 2 — Mở rộng chiều sâu khoa học
- [ ] Phân loại đa lớp 3 mức A/B/C có căn cứ khoa học (đã thiết kế, xem mục 8)
- [ ] Kiểm định chéo (cross-dataset validation) — trả lời RQ2/RQ3
- [ ] Kiểm định chéo bằng cross-validation (k-fold) thay vì chỉ 1 lần chia train/test
- [ ] Phân tích công bằng mô hình (fairness) theo giới tính/vùng miền/hoàn cảnh KT-XH
- [ ] Khảo sát định lượng đánh giá chất lượng khuyến nghị LLM (thang Likert)

### Cấp độ 3 — Mở rộng nâng cao (nếu đủ thời gian)
- [ ] Tích hợp RAG cho module LLM (tra cứu quy chế học vụ thật của trường)
- [ ] Dự báo theo thời gian (temporal/early prediction theo tuần học)
- [ ] Giải thích mô hình bằng SHAP values (Explainable AI)
- [ ] Kết hợp SHAP values làm input cho prompt LLM (khuyến nghị có căn cứ định lượng)

### Mốc thời gian dự kiến

| Giai đoạn | Nội dung |
|---|---|
| Tuần 1 | Review đa cấp học, EDA + mô hình hóa 2 bộ dữ liệu, xây dựng câu hỏi nghiên cứu |
| Tuần 2-4 | Hoàn thiện mô hình, kiểm định chéo, viết đề cương chi tiết |
| Tháng 2-3 | Module LLM (template hóa → RAG nếu kịp), Dashboard |
| Tháng 4 | Khảo sát đánh giá hệ thống với giảng viên/cố vấn |
| Tháng 5-6 | Viết báo cáo hoàn chỉnh, chuẩn bị bảo vệ |

> **Lưu ý đồng bộ:** Bản đề cương GVHD gửi ghi mốc 6 tháng (Tháng 1–6), trong khi trao đổi trực tiếp trước đó có đề cập khung 10 tháng (8/2026–6/2027). Cần xác nhận lại chính xác với GVHD để thống nhất lộ trình.

---

## 13. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Tổng quan tài liệu sơ bộ | ✅ Hoàn thành bản nháp |
| Notebook phân tích OULAD + UCI 697 | ✅ Đã viết, đang debug/chạy thực nghiệm |
| Kết quả thực nghiệm (Accuracy/F1...) | ⏳ Đang chờ chạy xong |
| Khung phân loại A/B/C | ✅ Đã thiết kế logic, chờ xác nhận bằng số liệu thật |
| Module LLM | ⏳ Chưa bắt đầu (dự kiến sau khi có kết quả mô hình) |
| RAG | ⏳ Dự kiến mở rộng, đang lên kế hoạch xin tài liệu quy chế học vụ |
| Dashboard | ⏳ Chưa bắt đầu |
| Khảo sát đánh giá | ⏳ Chưa bắt đầu |

*(Cập nhật lần cuối: theo tiến độ làm việc thực tế của nhóm — cần chỉnh sửa thủ công khi có thay đổi.)*

---

## 14. Nhóm thực hiện

| Vai trò | Họ tên | Phụ trách chính |
|---|---|---|
| Thành viên 1 | *(điền tên)* | Dữ liệu, mô hình học máy (RF, XGBoost) |
| Thành viên 2 | *(điền tên)* | Dashboard, tích hợp LLM/RAG |
| GVHD | *(điền tên thầy/cô)* | Định hướng khoa học, phản biện |

**Đơn vị:** *(điền tên khoa/trường)*
**Loại đề tài:** Nghiên cứu khoa học cấp Khoa

---

## 15. Tài liệu tham khảo

> Danh sách dưới đây tổng hợp từ quá trình tìm kiếm tài liệu của nhóm. **Cần xác minh và bổ sung đầy đủ tên tác giả, số DOI/link chính thức theo chuẩn APA hoặc IEEE trước khi đưa vào báo cáo nộp chính thức.**

1. *Predictive modeling of dropout in MOOCs using machine learning techniques* (2024) — So sánh 7 thuật toán ML trên OULAD, XGBoost đạt kết quả tốt nhất.
2. *A Comparative Analysis of Machine Learning Models for Dropout Prediction on the OULAD Dataset* (2025) — So sánh Logistic Regression, SVM, Random Forest, XGBoost trên OULAD.
3. *An Analysis of Datasets for Student Performance Evaluation Based on Machine Learning* (2025, Iraqi Journal of IT Systems) — Phân tích OULAD, xAPI-Edu-Data, UCI Student Performance.
4. *Student Dropout Prediction Using Random Forest and XGBoost Method* (2025) — Random Forest vượt trội XGBoost trên bộ dữ liệu 4,424 sinh viên.
5. *Predicting Student Dropout from Day One: XGBoost-Based Early Warning System* (2025) — XGBoost được ưu tiên trên bộ dữ liệu ~40,000 sinh viên năm nhất.
6. *Predicting student performance: A comprehensive review of machine learning, deep learning, and explainable AI approaches* (ScienceDirect, 2026) — Bài tổng quan, đề cập cả OULAD và UCIMLR.
7. *Artificial intelligence in student management systems to enhance academic performance monitoring and intervention* (Scientific Reports, 2025) — Khung AI lai kiểm định trên UCI, OULAD, NELS:88.
8. *Counterfactual Fairness Evaluation of Machine Learning Models on Educational Datasets* (arXiv, 2025) — Đánh giá công bằng mô hình trên OULAD và Student Performance.
9. *Cross-Institutional Transfer Learning for Educational Models: Implications for Model Performance, Fairness, and Equity* (arXiv:2305.00927, 2023) — Chuyển giao mô hình dự báo bỏ học xuyên tổ chức.
10. *A Generative Feedback System to Support Failure At-risk Online Learners* (2026) — Tích hợp LLM (Llama) vào hệ thống cảnh báo sớm trên nền tảng Canvas LMS.
11. *Leveraging LLMs and wearables to provide personalized recommendations* (2025) — LLM sinh khuyến nghị cá nhân hóa, cảnh báo rủi ro hallucination.
12. *Generating In-Context, Personalized Feedback for Intelligent Tutors with LLMs* (Springer) — Kỹ thuật structured prompting giảm sai lệch LLM.

### Bộ dữ liệu
- **OULAD:** Kuziłek J., Hlosta M., Zdrahal Z. *Open University Learning Analytics dataset.* Sci Data 4, 170171 (2017).
- **UCI 697:** Realinho, V. et al. *Predict Students' Dropout and Academic Success.* UCI Machine Learning Repository (2021).

---

## 16. Giấy phép & Đóng góp

Repo phục vụ mục đích học thuật (Nghiên cứu khoa học cấp Khoa). Dữ liệu OULAD và UCI 697 sử dụng theo giấy phép công khai gốc của từng bộ dữ liệu — vui lòng tham khảo điều khoản sử dụng tại nguồn gốc trước khi tái sử dụng cho mục đích khác.

**Đóng góp:** Đây là dự án nội bộ nhóm 2 thành viên dưới sự hướng dẫn của GVHD. Mọi thắc mắc hoặc góp ý xin liên hệ qua thông tin ở [mục 14](#14-nhóm-thực-hiện).

---

*README này được xây dựng và cập nhật liên tục theo tiến độ thực tế của đề tài — không phải bản tĩnh, cần chỉnh sửa thường xuyên khi có kết quả/quyết định mới từ GVHD.*
