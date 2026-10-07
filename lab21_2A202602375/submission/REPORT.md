# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Văn Sơn  **MSSV**: 2A202602375  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | `250 ticket CSKH → JSON triage` (mặc định: 250 ticket CSKH → JSON triage) |
| Train / val | `225` / `25` (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `30` |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? (Giữ nguyên mặc định vì đã có sẵn block think)

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.75 | 0.0 | 3404.5 |
| (b) base + optimized prompt | 0.6875 | 0.75 | 1.0 | 941.6 |
| (c) LoRA fine-tune | 0.9375 | 0.625 | 1.0 | 1775.5 |

**(b) có thật sự mạnh hơn (a) không?** `có` — nếu không, bạn đã cải thiện (b) thế nào?
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? `Không, giữ nguyên mặc định.`

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6268 | 0.9375 | 396.7 | 8.78 |
| `attn_only` | q,v | *(matched)* | 32,456,704 | 0.0001 | 0.5379 | 0.9375 | 274.4 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0 | 5018.7 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.8438 | 472.2 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
`attn_only` đạt kết quả target bằng (hòa) với cấu hình `correct` (cùng đạt 0.9375). Tuy nhiên, xét theo train loss thì `attn_only` lại có loss thấp hơn hẳn (0.5379 < 0.6268). Điều này cho thấy khi cân bằng ngân sách tham số (tăng rank r để bù đắp việc giảm vị trí layer gắn adapter), mô hình vẫn có không gian chứa đủ thông tin để học tác vụ, khẳng định rank là một đòn bẩy dung lượng (capacity) rất lớn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` cao hơn rất nhiều (1.5702 so với 0.6268 của `correct`) và target rớt xuống 0.0. Nếu chỉ nhìn đường loss tồi tệ này mà không biết rằng LR đang bị setup sai (ở mức 1x scale dành cho Full-FT thay vì 10x của LoRA), ta có thể kết luận nhầm rằng kiến trúc của mô hình kém, dataset chất lượng thấp hoặc bài toán quá khó không thể fine-tune được. Thực tế mô hình chỉ là không "cập nhật" đủ nhanh.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` tiết kiệm được khoảng gần 5GB VRAM (3.86 GB so với 8.78 GB của bf16 LoRA). Tuy nhiên, nó phải trả giá bằng việc điểm target giảm sút nhẹ (0.8438 so với 0.9375) và train loss cao hơn (0.7058 so với 0.6268). Con số thực nghiệm hoàn toàn ủng hộ khuyến nghị của vendor là không nên dùng QLoRA cho Qwen3.5 vì sai số lượng tử hóa làm giảm chất lượng mô hình một cách thấy rõ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.25` · `regression Δ = -0.125` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
Cấu hình fine-tune `correct` đã bị cổng kiểm tra phán quyết là `FAILED`. Lý do không nằm ở việc nó không học được tác vụ mới (thực tế target score đã tăng rất tốt +0.25 so với prompt tối ưu), mà nằm ở việc khả năng tổng quát của mô hình đã bị tổn hại nặng nề (regression giảm tới -0.125, trong khi mức chịu đựng tối đa chỉ là -0.020). Điều này nói lên rằng bài toán fine-tune đã vấp phải hiện tượng "thảm họa quên" (catastrophic forgetting) kinh điển. Khi model quá tập trung vào việc học định dạng JSON cho bài toán phân loại ticket, nó đã hy sinh khả năng trả lời các câu hỏi kiến thức phổ thông. Để khắc phục, ta bắt buộc phải trộn thêm 1-5% dữ liệu phổ thông (replay data) vào tập huấn luyện.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi, mình đặt chuột không dây..." | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | N/A | {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"} | ✅ FT thắng (Chính xác hoàn toàn) |
| 2 | "Shop ơi, mình đặt ốp lưng điện thoại..." | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"} | N/A | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"} | ✅ FT thắng (Chính xác hoàn toàn) |
| 3 | "Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124..." | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "trung_tinh"} | N/A | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": | ❌ **FT thua** (Bị cắt ngang JSON khi generate) |
| 4 | "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548..." | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"} | N/A | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen | ❌ **FT thua** (Bị cắt ngang JSON khi generate) |
| 5 | "Xin chào, mình đặt đèn bàn LED mã đơn VN880807..." | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | N/A | {"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"} | ✅ FT thắng (Chính xác hoàn toàn) |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung của các ca FT thua là mô hình sinh ra chuỗi JSON nhưng bị cắt ngang lưng chừng (bị truncate giữa chừng, mất đuôi `sentiment` và dấu đóng `}`), dẫn đến việc điểm target bị sụt (0.75) và chuỗi không parse được hoàn chỉnh. Điều này xảy ra do mô hình có thể đã chạm ngưỡng giới hạn số token sinh ra (`max_new_tokens`), hoặc do `eval_limit` khi test cắt ngắn câu.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên kết quả thực nghiệm toàn diện, bản fine-tune hiện tại KHÔNG NÊN được đưa thẳng lên production. Dù mô hình đã cho thấy mức độ thích nghi cực kỳ xuất sắc đối với tác vụ đích (nâng điểm độ chính xác phân loại từ 0.6875 lên 0.9375), nó lại phạm vào điều tối kỵ là "thảm họa quên" với độ suy giảm regression nghiêm trọng (-0.125). Nếu deploy, hệ thống sẽ xử lý ticket rất tốt nhưng có thể sẽ trả lời ngớ ngẩn trước các yêu cầu nằm ngoài luồng dữ liệu CSKH. Để giải quyết, ta cần thực thi quy trình trộn dữ liệu replay. Đòn bẩy thực sự và nguy hiểm nhất trong bài lab này chính là Cấu hình Learning Rate và Sự cân bằng Dữ liệu: chỉ cần sai LR, mô hình hoàn toàn mù tịt; và thiếu dữ liệu đa dạng, mô hình sẽ đánh đổi khả năng hiểu biết chung lấy sự tối ưu cục bộ.

**Ba điều tôi học được** (cụ thể, không generic):
1. Loss Mask là sinh tử: Không thể phó mặc nhắm mắt tin vào các flag như `assistant_only_loss=True` của thư viện mà phải trực tiếp in ra token id để chứng minh mô hình không tính loss ngược lên phần câu hỏi.
2. Training Loss không phản ánh chất lượng Task: Một cấu hình sai (`attn_only`) hoàn toàn có thể cho train loss đẹp hơn nhưng thực chất không mang lại kết quả tốt hơn mô hình chuẩn khi test bằng bài toán thực tế.
3. Tác hại của QLoRA: Việc nén xuống 4-bit giúp giảm 50-60% VRAM nhưng với kiến trúc cụ thể như Qwen3.5, mức độ nhiễu lượng tử hóa đủ lớn để kéo sụt hiệu năng target score, giống hệt với khuyến cáo của nhà phát triển.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Thu thập thêm dữ liệu phổ thông và tạo ra bộ replay data (chiếm khoảng 5% tập train) rồi chạy huấn luyện lại cấu hình `correct` để chứng minh tôi có thể vượt qua cổng kiểm tra `regression`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
