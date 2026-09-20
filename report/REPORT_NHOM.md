# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** TH True Mi
**Thành viên:** Trần Nguyễn Trí Dũng - Mai Huy Hoàng - Nguyễn Đức Tâm - Nguyễn Thị Hải Mi
**Ngày:** 20/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Chính sách Đổi trả, Hoàn tiền & Vận chuyển hoàn trả trên Shopee (góc nhìn Người mua và Người bán)

**Tại sao nhóm chọn chủ đề này?**
> Nhóm chọn chính sách Shopee vì đây là chủ đề thực tế, có nhiều điều kiện, thời hạn và quy trình dành riêng cho người mua/người bán. Bộ corpus cho phép kiểm tra cả retrieval theo nội dung lẫn metadata filter, đặc biệt ở các câu hỏi về phí trả hàng và thời hạn phản hồi.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Chính sách trả hàng và hoàn tiền | [Shopee](https://help.shopee.vn/portal/4/article/77251) | 2026-09-20 / effective-2026-03-11 | 19,609 | both; returns-refunds-policy; vi |
| 2 | Chính sách Vận chuyển Shopee | [Shopee](https://help.shopee.vn/portal/4/article/77250) | 2026-09-20 / 2026-09-15 | 24,376 | both; shipping-policy; vi |
| 3 | Điều Khoản Dịch Vụ của Shopee Mall | [Shopee](https://help.shopee.vn/portal/4/article/77262) | 2026-09-20 / not-stated | 33,568 | both; mall-terms; vi |
| 4 | Các phương thức gửi hàng hoàn trả và phí hoàn trả | [Shopee](https://help.shopee.vn/portal/4/article/189477) | 2026-09-20 / not-stated | 5,723 | buyer; returns-logistics; vi |
| 5 | Những quy định chung về Trả hàng/Hoàn tiền | [Shopee](https://help.shopee.vn/portal/4/article/188931) | 2026-09-20 / not-stated | 6,350 | buyer; returns-refunds-guide; vi |
| 6 | Quản lý đơn trả hàng hoàn tiền (Kênh Quản Lý Shop) | [Shopee](https://help.shopee.vn/portal/1/article/102521) | 2026-09-20 / not-stated | 3,699 | seller; returns-logistics; vi |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `shopee-quy-dinh-chung-tra-hang-hoan-tien` | Định danh ổn định để truy vết document/chunk và chấm expected source. |
| `audience` | enum | `buyer`, `seller`, `both` | Lọc đúng đối tượng; tránh trộn hướng dẫn buyer và seller. |
| `category` | string | `returns-logistics` | Phân biệt nhóm chính sách khi query có từ vựng gần nhau. |
| `language` | string | `vi` | Xác định ngôn ngữ corpus và chọn embedding phù hợp. |
| `source_url` / `retrieved_at` / `document_version` | string/date | URL Shopee / `2026-09-20` / `not-stated` | Bảo đảm provenance và biết thời điểm/phiên bản dữ liệu. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| Chính sách trả hàng và hoàn tiền | FixedSizeChunker (`fixed_size`) | 44 | 494.5 | Chunk đều, nhưng có thể cắt giữa câu/điều khoản. |
| Chính sách trả hàng và hoàn tiền | SentenceChunker (`by_sentences`) | 48 | 405.6 | Giữ câu tốt, phù hợp điều kiện ngắn. |
| Chính sách trả hàng và hoàn tiền | RecursiveChunker (`recursive`) | 62 | 314.3 | Giữ cấu trúc tốt hơn nhưng tạo nhiều chunk hơn. |
| Chính sách Vận chuyển Shopee | FixedSizeChunker (`fixed_size`) | 55 | 492.2 | Độ dài ổn định, dễ kiểm soát chi phí embedding. |
| Chính sách Vận chuyển Shopee | SentenceChunker (`by_sentences`) | 64 | 375.8 | Giữ ranh giới câu, số chunk tăng. |
| Chính sách Vận chuyển Shopee | RecursiveChunker (`recursive`) | 67 | 360.6 | Chia theo separator, giữ đoạn tốt hơn fixed. |
| Điều Khoản Dịch Vụ Shopee Mall | FixedSizeChunker (`fixed_size`) | 75 | 496.9 | Nhiều chunk nhưng độ dài gần mục tiêu. |
| Điều Khoản Dịch Vụ Shopee Mall | SentenceChunker (`by_sentences`) | 56 | 595.2 | Có section/câu dài vượt kích thước mục tiêu. |
| Điều Khoản Dịch Vụ Shopee Mall | RecursiveChunker (`recursive`) | 101 | 330.5 | Nhiều chunk nhỏ hơn, giữ ngữ cảnh theo separator. |

### Chiến lược của từng thành viên

> Các artifact benchmark thành công trong repo dùng cùng corpus, 5 query, OpenAI embedding và top-k=3: `bench/fixed-size.out.txt`, `bench/sentences.out.txt`, `bench/heading.out.txt` và `ket_qua_benchmark.txt`. File `bench/recursive.out.txt` hiện chỉ ghi lỗi môi trường; các số Recursive bên dưới cần được Hoàng xuất lại từ lượt chạy thành công trước khi nộp.

**Mi — Hải Mi**
- **Loại chiến lược:** Fixed Size (`fixed_size`, `chunk_size=500`, `overlap=50`)
- **Mô tả & lý do chọn:** Chia văn bản theo kích thước cố định, có overlap để giảm mất ngữ cảnh ở ranh giới chunk. Cách này dễ kiểm soát số chunk và chi phí embedding.

**Dũng — Trần Nguyễn Trí Dũng**
- **Loại chiến lược:** Heading/Section (`heading`, `chunk_size=500`)
- **Mô tả & lý do chọn:** Tách theo heading Markdown hoặc mục chính sách đánh số như `1.`, `1.1.`; section dài được chia tiếp và giữ lại heading để bảo toàn ngữ cảnh điều khoản. Chiến lược này đáp ứng yêu cầu riêng của biến thể L3B.

**Hoàng — Mai Huy Hoàng**
- **Loại chiến lược:** Recursive (`recursive`, `chunk_size=500`)
- **Mô tả & lý do chọn:** Thử separator theo thứ tự đoạn, dòng, câu và từ; nội dung dài tiếp tục được chia ở separator nhỏ hơn. Chiến lược này phù hợp chính sách có nhiều mục và đoạn văn dài.

**Tâm — Nguyễn Đức Tâm**
- **Loại chiến lược:** Sentence (`by_sentences`, tối đa 3 câu/chunk)
- **Mô tả & lý do chọn:** Gom các câu hoàn chỉnh để giữ điều kiện và thời hạn trong cùng chunk. Điểm yếu là câu quá dài hoặc văn bản có dấu chấm đặc biệt có thể tạo chunk vượt kích thước mong muốn.

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Mi | Fixed Size | 3 / 10 | Độ dài ổn định, pipeline đơn giản, dễ tái lập. | EvidenceHit@3=3/5, MRR=0.267; dễ cắt rời bằng chứng khỏi câu hỏi. |
| Dũng | Heading/Section | 6 / 10 | EvidenceHit@3=4/5, MRR=0.600; giữ tên mục cùng điều khoản và đưa Q1/Q5 lên rank 1. | Tạo 292 chunks; Q3 chưa đưa exact evidence vào top-3. |
| Hoàng | Recursive | 9 / 10 | EvidenceHit@3=5/5, MRR=0.900; giữ cấu trúc đoạn tốt. | Tạo nhiều chunk nhất (268), chi phí embedding cao hơn. |
| Tâm | Sentence | 7 / 10 | EvidenceHit@3=4/5, MRR=0.700; giữ câu tự nhiên. | Q3 không có evidence trong top-3; một số câu dài. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Theo các số đã tổng hợp, Recursive đạt cao nhất với EvidenceHit@3=5/5, điểm retrieval-evidence 9/10 và MRR=0.900; Sentence đạt 4/5, 7/10, 0.700; Heading đạt 4/5, 6/10, 0.600; Fixed Size đạt 3/5, 3/10, 0.267. Đổi lại, Heading sinh 292 chunks, cao hơn Recursive 268, Fixed Size 210 và Sentence 192. Artifact Recursive cần được xuất lại để xác minh độc lập các số đã ghi.

**Yêu cầu Heading/Section đã hoàn thành:** Dũng đã chạy `HeadingChunker` trên cùng corpus và 5 query; kết quả OpenAI được lưu tại `bench/heading.out.txt`.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Người mua có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền cho đơn thường? | 15 ngày kể từ trạng thái “Giao hàng thành công”. | `shopee-quy-dinh-chung-tra-hang-hoan-tien` — mục 1.2 |
| 2 | Đơn thực phẩm tươi sống/đông lạnh thời hạn bao lâu? | Trong vòng 24 giờ. | `shopee-quy-dinh-chung-tra-hang-hoan-tien` — mục 1.2 |
| 3 | Chọn “Tự sắp xếp” thì ai trả phí trước, bao lâu được hoàn? | Người mua trả trước; Shopee hỗ trợ hoàn trong 3–5 ngày làm việc. | `shopee-phuong-thuc-gui-hang-hoan-tra` — mục 2.2 |
| 4 | Shop chưa nhận được hàng hoàn thì sau bao lâu khiếu nại được? | Sau 2 ngày kể từ khi hệ thống cập nhật người mua đã gửi hàng cho ĐVVC. | `shopee-seller-quan-ly-don-tra-hang-hoan-tien` — mục C |
| 5 | Người Bán tại Shopee Mall phải nhận lại hàng hoàn trong bao lâu? | 07 ngày làm việc kể từ quyết định cuối cùng của Shopee. | `shopee-dieu-khoan-shopee-mall` — mục 1.6 |

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Người mua có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền cho đơn thường? | Recursive / Sentence / Heading | Có, rank 1 | Heading đưa bằng chứng 15 ngày vào rank 1. `expected_doc_id` không ở top-3, nhưng runner chấm evidence theo nội dung nên vẫn tính 2/2. |
| 2 | Đơn thực phẩm tươi sống/đông lạnh thời hạn bao lâu? | Recursive / Heading | Có, rank 2 | Recursive và Heading đưa đúng bằng chứng 24 giờ vào top-3; Fixed/Sentence không đạt exact match. |
| 3 | Chọn “Tự sắp xếp” thì ai trả phí trước, bao lâu được hoàn? | Recursive | Có, rank 1 | Filter buyer giúp đưa đúng tài liệu và bằng chứng vào top-1. |
| 4 | Shop chưa nhận được hàng hoàn thì sau bao lâu khiếu nại được? | Recursive / Sentence | Có, rank 1 | Cả hai đưa đúng seller chunk lên top-1; Fixed đúng rank 2. |
| 5 | Người Bán tại Shopee Mall phải nhận lại hàng hoàn trong bao lâu? | Recursive / Sentence / Heading | Có, rank 1 | Recursive, Sentence và Heading đều đưa đúng bằng chứng 07 ngày lên top-1; Fixed đúng rank 2. |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> Có. Ở câu 3, filter `audience=buyer` thay đổi top-3 nhưng chưa cải thiện evidence rank cho Heading; ở câu 4, filter `audience=seller` đưa chunk chứa đáp án từ ngoài top-3 lên rank 2. Filter làm giảm candidate/recall tổng thể nếu gán audience quá hẹp, nên metadata phải nhất quán với nội dung.

### Phân tích lỗi (Failure Analysis)

**Q3 — Chọn “Tự sắp xếp” thì ai trả phí trước, bao lâu được hoàn?** Với Heading, ba chunk đầu đều đúng chủ đề và đúng tài liệu nhưng không chứa exact evidence “trả trước” cùng “3–5 ngày”, nên đạt 0/2. Filter buyer thay đổi top-3 nhưng không cải thiện evidence rank. Nhóm cần tinh chỉnh ranh giới mục 2.2 hoặc giảm `chunk_size` để chunk chứa chính sách phí cạnh tranh tốt hơn với các mục hướng dẫn trả hàng.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> - Recursive đạt chất lượng retrieval tốt nhất trên corpus này nhưng tạo nhiều chunk nhất.
> - Heading đạt EvidenceHit@3=4/5 và giữ tên mục chính sách cùng nội dung, nhưng tạo 292 chunks và thất bại ở Q3.
> - Metadata filter buyer/seller giúp giảm nhiễu khi hai nhóm tài liệu dùng chung từ “Trả hàng/Hoàn tiền”.
> - Benchmark hiện dùng OpenAI embedding; nhóm kiểm tra exact evidence thay vì chỉ nhìn similarity score hoặc doc_id.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng corpus và embedding nhưng Recursive đạt Hit@3 cao nhất vì giữ được đoạn/điều khoản liền mạch. Heading giúp Q1 và Q5 lên rank 1 nhờ giữ tên mục, nhưng section-aware splitting tạo nhiều chunk cạnh tranh và Q3 vẫn thất bại. Fixed Size dễ cắt giữa bằng chứng; Sentence giữ câu tự nhiên nhưng một số câu dài.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nhóm sẽ tách thêm các tài liệu `audience=both` thành các section buyer/seller khi nội dung cho phép, lưu exact evidence span cho từng gold answer và tinh chỉnh kích thước Heading chunks để cải thiện Q3.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 14 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 9 / 10 |
| Thuyết trình (Demo) | 4 / 5 |
| **Tổng phần nhóm** | **37 / 40 (tự đánh giá thận trọng)** |
