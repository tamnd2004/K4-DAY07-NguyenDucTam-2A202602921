# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Tâm
**Nhóm:** THTrueMi
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = 60

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Hai vector embedding chỉ về gần cùng một hướng trong không gian nhiều chiều, nghĩa là mô hình xếp hai đoạn văn bản vào cùng một vùng ngữ nghĩa. Nó nói rằng hai câu *nói về cùng một chuyện*, chứ không phải hai câu *dùng cùng những từ*.

**Ví dụ có độ tương tự CAO:**

- Câu A: "Người mua có thể yêu cầu trả hàng trong vòng 15 ngày."
- Câu B: "Khách hàng được phép hoàn sản phẩm trong hai tuần kể từ khi nhận."
- Tại sao tương đồng: Hai câu **không dùng chung một từ khoá nội dung nào** — "người mua"/"khách hàng", "trả hàng"/"hoàn sản phẩm", "15 ngày"/"hai tuần" — nhưng diễn đạt đúng một quy định. Nếu chỉ so khớp chuỗi thì điểm gần bằng 0; embedding vẫn cho điểm cao vì nó mã hoá nghĩa. Đây chính là chỗ chứng minh embedding hiểu nghĩa chứ không đếm từ trùng (điểm đo thực tế: xem cặp 1 mục 4).

**Ví dụ có độ tương tự THẤP:**

- Câu A: "Chính sách đổi trả hàng hóa trên sàn thương mại điện tử."
- Câu B: "Hướng dẫn nấu phở bò truyền thống Hà Nội."
- Tại sao khác: Hai câu thuộc hai miền chủ đề không giao nhau — quy định thương mại và ẩm thực. Cùng là tiếng Việt, cùng cấu trúc danh ngữ, nhưng không chia sẻ nội dung ngữ nghĩa nào (điểm đo thực tế: xem cặp 3 mục 4).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Cosine chỉ đo **góc**, bỏ qua **độ dài** vector. Với văn bản, độ dài vector thường phản ánh độ dài đoạn văn chứ không phải nội dung, nên khoảng cách Euclid sẽ coi một đoạn 50 từ và một đoạn 500 từ *cùng chủ đề* là xa nhau chỉ vì chênh lệch độ lớn. Trong lab này cả `MockEmbedder` lẫn `LocalEmbedder` đều trả vector đã chuẩn hoá (‖v‖ = 1), nên tích vô hướng bằng đúng cosine — đó là lý do docstring của `search` cho phép dùng dot product cho gọn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> _Trình bày phép tính:_
>
> `số_chunk = ceil((độ_dài − overlap) / (chunk_size − overlap))`
> `= ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
>
> _Đáp án:_ **23 chunks**

Tôi không tin công thức suông mà kiểm lại bằng chính `FixedSizeChunker` trong repo:

```bash
python -c "
from src.chunking import FixedSizeChunker
print(len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)))
"
# -> 23
```

Công thức và code khớp nhau.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> `ceil((10000 − 100) / (500 − 100)) = ceil(9900 / 400) = ceil(24.75) = 25` chunks — chạy lại `FixedSizeChunker` với `overlap=100` cũng ra đúng **25**. Tăng overlap 50 → 100 làm số chunk tăng 23 → 25 (~9%), vì bước nhảy giảm từ 450 xuống 400 ký tự.
>
> Lý do đôi khi vẫn muốn overlap lớn hơn dù tốn thêm chunk: ranh giới cắt là **mù ngữ nghĩa** — nó rơi vào đâu thì cắt ở đó. Một câu quy định bị cắt đôi giữa hai chunk sẽ không chunk nào chứa trọn ý, và cả hai đều truy xuất kém. Overlap lớn hơn làm tăng xác suất ít nhất một chunk giữ được trọn vẹn câu đó. Với corpus của nhóm tôi điều này rất cụ thể: các mốc như *"trong vòng 03 – 05 ngày làm việc kể từ ngày Người Mua đã gửi Sản Phẩm Hoàn Trả"* dài gần 100 ký tự — cắt trúng giữa là mất luôn đáp án.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

> Tôi dùng `re.compile(r"(?<=[.!?])\s+")` — một **lookbehind** thay vì `[.!?]\s+`. Khác biệt then chốt: split bằng `[.!?]\s+` sẽ *nuốt mất* dấu câu và mọi chunk thành câu cụt; lookbehind cắt ở vị trí **sau** dấu câu nên dấu chấm ở lại với câu mà nó kết thúc. Một pattern này bao trọn cả `". "`, `"! "`, `"? "` và `".\n"` vì `\s+` khớp cả dấu cách lẫn xuống dòng. Sau khi tách, tôi `strip()` từng câu, bỏ chuỗi rỗng, rồi gom `max_sentences_per_chunk` câu thành một chunk; text rỗng trả `[]` chứ không crash.
>
> **Edge case tôi biết là mình chưa xử lý được:** regex này cắt sai ở **chữ viết tắt** ("TS.", "v.v.", "TP.") và ở **số thập phân kiểu Anh** ("1.5 triệu") — mọi dấu chấm theo sau bởi khoảng trắng đều bị coi là hết câu. Với corpus Shopee của nhóm thì "v.v..." xuất hiện thật trong mục quy định hàng hoá đặc thù, nên lỗi này có xảy ra chứ không phải giả định. Cách sửa đúng là dùng thư viện tách câu tiếng Việt (underthesea, pyvi) hoặc thêm danh sách ngoại lệ, nhưng đề bài yêu cầu regex nên tôi giữ nguyên và ghi nhận hạn chế.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

> Thuật toán đi **hai chiều**, và chiều thứ hai mới là chỗ dễ bỏ sót:
>
> 1. **Đệ quy xuống sâu** — thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split` với phần separator còn lại. Cắt bằng ranh giới "to" trước để giữ ngữ nghĩa.
> 2. **Gom lên** — các mảnh nhỏ liền kề được nối lại bằng chính separator đó cho tới sát `chunk_size`. Thiếu bước này, một file nhiều dòng ngắn — đúng như tài liệu Shopee với rất nhiều dòng "Bước 1", "Bước 2" — sẽ sinh ra hàng trăm chunk vụn 5–10 ký tự và retrieval hỏng hoàn toàn.
>
> Tôi thêm một nhánh nhỏ: nếu separator hiện tại **không xuất hiện** trong text (`len(pieces) == 1`) thì bỏ qua, hạ thẳng xuống separator kế tiếp, thay vì đệ quy vô ích trên cùng một chuỗi.
>
> **Ba base case:**
>
> - Text sau `strip()` rỗng → trả `[]`
> - `len(text) <= chunk_size` → trả `[text]`, không cắt nữa
> - Hết separator **hoặc** separator là `""` → `_hard_split`, cắt cứng theo `chunk_size`
>
> Base case thứ ba là cái mà test `test_empty_separators_falls_back_gracefully` nhắm vào: nó truyền thẳng `separators=[]`, thiếu nhánh đó là `IndexError`.

**`HeadingChunker.chunk`** — hướng tiếp cận:

> Đây không phải chiến lược thi đấu của tôi (của tôi là `by_sentences`), nhưng `bench.py` import nó để chạy được cả bốn chiến lược trong một lượt, nên nó phải tồn tại trong `src`. Biến thể L3B cũng yêu cầu nhóm có ít nhất một người chunk theo heading.
>
> Ý tưởng: tách trước mỗi dòng tiêu đề, mỗi section thành một chunk, section nào dài quá `chunk_size` thì hạ xuống `RecursiveChunker`.
>
> **Chỗ khó nhất là nhận diện đâu là tiêu đề.** Corpus Shopee không chỉ dùng heading Markdown (`# ...`) mà chủ yếu dùng mục đánh số (`1.`, `2.3.`). Nhưng cùng một dạng `2.1.` lại vừa là tiêu đề mục *vừa là* một điều khoản dài mấy trăm ký tự — ví dụ `2.1. Theo các điều khoản và điều kiện được quy định trong...` chạy suốt một đoạn văn. Nếu bắt mọi dòng bắt đầu bằng số thì mỗi điều khoản thành một "section" riêng và chunker mất hết tác dụng. Tôi phân biệt bằng **độ dài dòng**: một dòng đánh số chỉ được coi là tiêu đề khi nó ngắn hơn `MAX_HEADING_LENGTH = 120` ký tự — tức nó là cái tên, không phải cái nội dung.
>
> **Chi tiết dễ bỏ sót:** khi một section dài phải cắt nhỏ, tôi **gắn lại tiêu đề vào từng mảnh con** (`f"{heading}\n{piece}"`). Không có nó, mảnh thứ hai trở đi mất ngữ cảnh "đây là mục nói về cái gì" — mà với văn bản quy định thì mất tiêu đề mục là mất luôn thông tin điều khoản này áp dụng cho ai. Vì tiêu đề chiếm chỗ, tôi trừ trước độ dài tiêu đề khỏi ngân sách của phần thân, và chặn sàn ở `chunk_size // 2` để một tiêu đề dài bất thường không bóp phần thân xuống còn vài chữ.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

> Tôi **bỏ hẳn nhánh ChromaDB**, chỉ dùng list in-memory. Ba lý do: không test nào cần nó, `requirements.txt` không cài nó, và code khởi tạo sẵn có một cái bẫy — `self._use_chroma = True` được gán **trước khi** client được tạo, nên nếu máy chấm bài tình cờ có `chromadb` thì mọi method sẽ rẽ vào nhánh chưa cài đặt và cả 14 test sập. Bỏ đi thì code ngắn hơn và hành vi giống nhau trên mọi máy.
>
> `add_documents` **không tự chunk**: 1 `Document` = 1 record, đúng như `test_add_documents_increases_size` mong đợi. Việc chia nhỏ do tầng ngoài làm, mỗi chunk thành một `Document` riêng.
>
> Hai chi tiết trong `_make_record`: (a) **copy** `metadata` bằng `dict(...)` thay vì dùng thẳng object của người gọi, để họ sửa dict sau đó không âm thầm ghi đè dữ liệu đã lưu; (b) record id là `f"{doc.id}::{self._next_index}"` vì **cùng một `doc.id` có thể được add hai lần** và không được dedupe — `test_add_more_increases_further` thêm 2 rồi 3 document trùng id và mong đợi size = 5.
>
> `search` chấm điểm bằng **cosine** (`compute_similarity`) chứ không phải dot product thuần. Với vector đã chuẩn hoá hai cách cho kết quả y hệt, nhưng cosine vẫn đúng nếu sau này đổi sang backend không chuẩn hoá. Kết quả trả về **bỏ `embedding` đi** — vector 384 chiều mỗi hit làm output không đọc nổi khi in ra terminal.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

> **Lọc trước, xếp hạng sau.** Nếu lấy top-k rồi mới bỏ cái không khớp, k slot có thể đã bị chiếm hết bởi tài liệu sai audience và ta còn lại 0 kết quả *dù store vẫn còn đầy tài liệu hợp lệ*. Với corpus của nhóm tôi rủi ro này rất thật: 3/6 tài liệu là `audience: both`, chúng dài và bám sát chủ đề, hoàn toàn có thể chiếm trọn top-3 và đẩy tài liệu `seller` duy nhất ra ngoài.
>
> Tôi tách riêng helper `_search_records(query, records, top_k)` chạy trên **một tập record bất kỳ**, rồi cho cả `search()` lẫn `search_with_filter()` đi qua đúng đường code đó — chỉ khác nhau ở tập ứng viên đầu vào. Nhờ vậy hai hàm không thể lệch kết quả, và `test_no_filter_returns_all_candidates` pass do thiết kế chứ không phải nhờ may.
>
> `delete_document` xoá mọi record có `metadata['doc_id']` khớp, trả `True/False` tuỳ có xoá được gì không. Trong `_make_record` tôi đặt `metadata.setdefault("doc_id", doc.id.split("#")[0])` — cắt hậu tố chunk, vì khi một file sinh ra `"file#0"`, `"file#1"` thì `doc_id` phải trỏ về **file gốc** chứ không phải id của chunk; nếu không `delete_document("file")` sẽ không xoá được gì.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

> Ba nhịp: truy xuất top-k → dựng prompt có ngữ cảnh → gọi `llm_fn`.
>
> Phần tôi đầu tư nhất là **cách dựng ngữ cảnh**. Mỗi chunk được đánh số và kèm nguồn: `[1] nguồn=<doc_id> (score=0.612)` rồi mới đến nội dung. Prompt yêu cầu model trích dẫn đúng số đó trong câu trả lời. Nhờ vậy mỗi ý trong câu trả lời truy ngược được về đúng chunk và đúng file — đây là tiêu chí **Source Traceability** trong `docs/EVALUATION.md`, và với corpus là văn bản quy định thì nó không phải tính năng phụ: một câu trả lời về thời hạn trả hàng mà không chỉ được nguồn thì không dùng được.
>
> Hai ràng buộc chống bịa: prompt nói rõ **chỉ dùng ngữ cảnh được cung cấp**, không có thì phải nói là không tìm thấy. Và khi store rỗng, `answer` **trả thẳng câu thông báo mà không gọi `llm_fn`** — không có ngữ cảnh thì mọi câu trả lời đều là bịa, gọi LLM chỉ tốn một request để sinh ra điều đó. Tôi kiểm lại bằng cách đếm số lần `llm_fn` được gọi: 0.
>
> Prompt viết bằng tiếng Việt vì toàn bộ corpus và 5 câu hỏi đánh giá đều là tiếng Việt.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.12.8, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Tamnd\AI_thuc_chien\K4-DAY07-NguyenDucTam-2A202602921
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
...  (34 dòng PASSED lược bớt cho gọn)  ...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

Checkpoint 3 (`pytest tests/ -k "Chunker or Similarity or Compare" -v`): **23 passed, 19 deselected** — gồm 7 test `FixedSizeChunker` có sẵn cộng 16 test của phần tôi viết.

`python main.py "Chunking là gì?"` chạy trọn vẹn, exit code 0. Dòng `Skipping missing file: data\customer_support_playbook.txt` là bình thường, repo không có file đó.

**Số lượng bài test vượt qua (pass):** **42** / 42

---

> ### ⚠️ Ghi chú về backend embedding cho mục 4 và 5
>
> Hai mục dưới đây chạy trên **hai backend khác nhau**, và tôi giữ nguyên như vậy một cách có chủ ý:
>
> | Mục | Backend | Nguồn số liệu |
> | --- | --- | --- |
> | 4. Dự đoán độ tương tự | `MockEmbedder` (băm MD5 → 64 chiều giả ngẫu nhiên) | đo ở checkpoint đầu buổi, khi chưa có API key |
> | 5. Kết quả truy xuất | **OpenAI embedding** (`EMBEDDING_PROVIDER=openai`) | `ket_qua_benchmark.txt` |
>
> Tôi không xoá số liệu mock ở mục 4, vì đặt cạnh mục 5 thì hai mục thành một phép đối chứng trực tiếp: cùng corpus, cùng công thức cosine, chỉ khác mô hình sinh vector. Mục 4 cho thấy phép đo trả về cái gì khi vector **không** mã hoá ngữ nghĩa; mục 5 cho thấy nó trả về cái gì khi có. Phản ngẫm cuối mục 4 so sánh trực tiếp hai dải số này.
>
> Lệnh tái tạo mục 5:
>
> ```bash
> echo "EMBEDDING_PROVIDER=openai" > .env
> echo "OPENAI_API_KEY=sk-..." >> .env
> python bench.py --strategy by_sentences --provider openai --top-k 3
> ```

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán được **ghi cứng trong script trước khi chạy** (biến `PAIRS`), không điền ngược sau khi biết kết quả.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (mock) | Đúng? |
| --- | ----- | ----- | ------- | ------------------- | ----- |
| 1 | Người mua có thể yêu cầu trả hàng trong vòng 15 ngày. | Khách hàng được phép hoàn sản phẩm trong hai tuần kể từ khi nhận. | cao | **+0.067** | ✗ |
| 2 | Người bán phải phản hồi yêu cầu trả hàng trong 2 ngày. | Nhà bán hàng cần trả lời đề nghị hoàn trả trong vòng 48 giờ. | cao | **−0.032** | ✗ |
| 3 | Chính sách đổi trả hàng hóa trên sàn thương mại điện tử. | Hướng dẫn nấu phở bò truyền thống Hà Nội. | thấp | **−0.170** | ✓ |
| 4 | Shopee hoàn phí trả hàng trong 3-5 ngày làm việc. | Shopee hoàn phí vận chuyển trong 3-5 ngày làm việc. | cao | **+0.230** | ✗ |
| 5 | Người bán bị khóa tài khoản do tạo đơn hàng ảo. | Người mua được miễn phí trả hàng khi gửi tại bưu cục. | thấp | **+0.015** | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Bất ngờ nhất là **cặp 4**. Hai câu chỉ khác đúng **một từ** — "phí trả hàng" so với "phí vận chuyển" — mà chỉ đạt +0.230. Trong khi đó cặp 3, hai câu chẳng liên quan gì nhau (chính sách đổi trả so với công thức nấu phở), lại được −0.170. Khoảng cách giữa "gần như trùng khớp" và "hoàn toàn vô quan" chỉ có 0,4.
>
> Điều này nói lên một chuyện quan trọng: **`MockEmbedder` không biểu diễn ý nghĩa gì cả.** Nó băm MD5 chuỗi rồi sinh 64 số giả ngẫu nhiên, nên hai chuỗi *khác nhau dù chỉ một ký tự* cho hai vector độc lập thống kê. Cosine giữa hai vector ngẫu nhiên 64 chiều dao động quanh 0 với độ lệch chuẩn ~1/√64 = 0,125 — và đúng như vậy, cả 5 điểm đo đều nằm trong khoảng ±0,25, tức **hoàn toàn phù hợp với nhiễu thuần tuý**. Hai lần tôi "dự đoán đúng" (cặp 3, 5) chỉ là ăn may vì tôi đoán "thấp" mà mọi thứ đều thấp.
>
> **Mục 5 cho tôi đúng nhóm đối chứng mà mục này thiếu.** Cùng loại phép đo cosine, nhưng chạy bằng OpenAI embedding trên corpus thật, dải điểm dịch hẳn lên: chunk chứa đáp án của Q1 đạt **0,7041**, Q5 đạt **0,7331**, và ngay cả một chunk *sai* nhưng cùng chủ đề như `chinh-sach-van-chuyen#15` ở Q2 cũng còn **0,7004**. Toàn bộ dải đo có nghĩa nằm ở vùng 0,47–0,73 — nơi mà mock không bao giờ chạm tới, vì trần của nhiễu 64 chiều chỉ quanh 0,25.
>
> Đối chiếu đó làm rõ chính điều mà bài tập muốn dạy. Cặp 1 và cặp 2 là hai cặp **đồng nghĩa nhưng không chung từ khoá** ("15 ngày"/"hai tuần", "2 ngày"/"48 giờ"); mock chấm +0,067 và −0,032, tức không phân biệt nổi "cùng nghĩa khác chữ" với "chẳng liên quan". Còn ở mục 5, đúng kiểu quan hệ đó lại hoạt động: câu hỏi Q1 viết là *"có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền"*, còn chunk thắng cuộc viết là *"trong vòng 15 (mười lăm) ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công"* — **không chung một từ khoá nội dung nào** với câu hỏi, vẫn lên hạng 1 với 0,7041. Đó chính là phần "ngữ nghĩa" mà mô hình đóng góp, và là toàn bộ lý do RAG hoạt động được.
>
> Một hệ quả thực tế tôi rút ra: **điểm cosine tuyệt đối không tự nó nói lên chất lượng.** Ở mục 5, chunk sai của Q2 (0,7004) còn cao hơn chunk đúng của Q4 (0,6132). Nếu tôi đặt một ngưỡng kiểu "chỉ nhận kết quả trên 0,65" thì tôi sẽ nhận nhầm câu sai và loại đúng câu đúng. Score chỉ dùng để **xếp hạng trong cùng một truy vấn**, không dùng để so giữa các truy vấn khác nhau.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

**Chiến lược của tôi trong nhóm:** `SentenceChunker(max_sentences_per_chunk=3)` — chiến lược `by_sentences`.

```bash
python bench.py --strategy by_sentences --provider openai --top-k 3
```

**Cấu hình:** 6 tài liệu Shopee của nhóm → **192 chunks**, backend **OpenAI embedding**, top-k = 3, bộ 5 câu hỏi chung của nhóm trong `data/ecommerce/benchmark.json`. Output đầy đủ: `ket_qua_benchmark.txt`.

| Tài liệu | audience | Số ký tự thân bài | chunks | Độ dài TB |
| --- | --- | --- | --- | --- |
| shopee-chinh-sach-tra-hang-hoan-tien | both | 19.442 | 47 | 413 |
| shopee-chinh-sach-van-chuyen | both | 24.373 | 64 | 380 |
| shopee-dieu-khoan-shopee-mall | both | 33.568 | 56 | 599 |
| shopee-phuong-thuc-gui-hang-hoan-tra | buyer | 5.723 | 9 | 635 |
| shopee-quy-dinh-chung-tra-hang-hoan-tien | buyer | 6.145 | 9 | 682 |
| shopee-seller-quan-ly-don-tra-hang-hoan-tien | seller | 3.698 | 5 | 739 |

> **Ghi chú tái lập:** bảng trên tính lại bằng code trong repo và ra **190 chunks**, trong khi artifact ghi **192**. Nguyên nhân: tôi có sửa `bench.py` sau lượt chạy OpenAI. Tôi đã đối chiếu lại 7 chunk được trích trong artifact — **6/7 khớp nguyên văn**, riêng chunk artifact gọi là `shopee-quy-dinh-chung-tra-hang-hoan-tien#3` nay nằm ở `#2`. Tức nội dung truy xuất không đổi, chỉ nhãn chỉ số và tổng số chunk lệch 2. Tôi giữ nguyên số của artifact ở mọi chỗ bên dưới thay vì sửa tay cho khớp.

### Bảng kết quả top-3

| # | Câu hỏi (Query) | Top-1 chunk truy xuất được | Score | Hạng chunk chứa đáp án | Có liên quan? | Điểm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Người mua có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền cho đơn thường? | `chinh-sach-tra-hang-hoan-tien#7` — điều 3.2, đúng câu "15 (mười lăm) ngày" | 0.7041 | **1** | ✓ | **2/2** |
| 2 | Đơn thực phẩm tươi sống/đông lạnh thời hạn bao lâu? | `chinh-sach-van-chuyen#15` — danh mục hàng cần bảo quản đặc biệt | 0.7004 | 2 (`chinh-sach-tra-hang-hoan-tien#7`, 0.4719) | ~ | **1/2** |
| 3 | Chọn "Tự sắp xếp" thì ai trả phí trước, bao lâu được hoàn? *(filter `audience=buyer`)* | `phuong-thuc-gui-hang-hoan-tra#2` — lưu ý khi chọn ĐVVC đến lấy hàng | 0.5246 | không có trong top-3 | ✗ | **0/2** |
| 4 | Shop chưa nhận được hàng hoàn thì sau bao lâu khiếu nại được? *(filter `audience=seller`)* | `seller-quan-ly-don-tra-hang-hoan-tien#2` — bảng hạn phản hồi | 0.6132 | **1** | ✓ | **2/2** |
| 5 | Người Bán tại Shopee Mall phải nhận lại hàng hoàn trong bao lâu? | `dieu-khoan-shopee-mall#17` — đúng câu "07 (bảy) ngày làm việc" | 0.7331 | **1** | ✓ | **2/2** |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4** / 5

**Tổng kết từ runner:** `chunks=192, DocHit@3=4/5, EvidenceHit@3=4/5, Score=7/10, MRR=0.700`

### Hai mức chấm lệch nhau ở đâu — và lệch theo chiều nào

Tôi chấm hai mức như `docs/SCORING.md` yêu cầu: **hạng của tài liệu gold** và **hạng của chunk thật sự chứa đáp án**. Hai mức này không trùng nhau ở Q1 và Q2:

| # | Hạng tài liệu gold | Hạng chunk chứa đáp án | Chấm theo `doc_id` | Chấm theo nội dung |
| --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1/2 | **2/2** |
| 2 | không có trong top-3 | 2 | 0/2 | **1/2** |

Bài lab cảnh báo rằng chấm theo `doc_id` sẽ **thổi phồng** kết quả. Trên corpus của nhóm tôi nó lại sai theo **chiều ngược lại**: nó *hạ thấp* kết quả. Lý do là cùng một quy định được viết lại ở hai văn bản có thẩm quyền ngang nhau — mốc "15 ngày" và mốc "24 giờ" đều xuất hiện cả trong `quy-dinh-chung-tra-hang-hoan-tien` (tài liệu nhóm khai là gold) lẫn trong điều 3.2 của `chinh-sach-tra-hang-hoan-tien`. Retrieval lấy về bản thứ hai. Với người dùng thật thì câu trả lời đó **đúng và trích dẫn được**, nhưng thước đo `doc_id` chấm 0.

Kết luận tôi rút ra không phải "mức nào đúng hơn", mà là: **`doc_id` đo sai bản chất của thứ cần đo.** Câu hỏi thực sự là "ngữ cảnh trả về có chứa câu trả lời không", và chỉ kiểm ở mức nội dung mới trả lời được. `expected_doc_id` nên dùng để chẩn đoán, không nên dùng để chấm.

### A/B bắt buộc — có filter so với không filter

| # | Filter | Có filter | Không filter | Kết luận runner |
| --- | --- | --- | --- | --- |
| 3 | `audience=buyer` | tài liệu gold hạng 1, đáp án **không** vào top-3 → 0/2 | tài liệu gold hạng 2, đáp án **không** vào top-3 → 0/2 | `CHANGED` |
| 4 | `audience=seller` | đáp án **hạng 1** → **2/2** | đáp án hạng 2 → 1/2 | `IMPROVED` |

**Q4 là bằng chứng rõ nhất cho việc metadata filter có ích.** Khi không lọc, top-1 là `chinh-sach-van-chuyen#43` với score **0.6378** — cao hơn cả chunk chứa đáp án (0.6132). Đoạn đó nói về *"Khiếu nại sai phí vận chuyển… trong vòng 7 ngày"*: cùng từ "khiếu nại", cùng cấu trúc "trong vòng N ngày", cùng miền tiền bạc — nhưng là một thời hạn **khác** cho một tình huống **khác**. Đây đúng là cái bẫy mà đề bài mô tả: hai tài liệu cùng chủ đề, cùng từ vựng, khác đối tượng, khác đáp án. Lọc `audience=seller` loại đúng đoạn nhiễu đó và đẩy đáp án lên hạng 1, điểm từ 1/2 lên 2/2.

**Q3 cho thấy mặt còn lại:** filter **không** phải thuốc chữa cho một chunk tồi. Lọc `buyer` thu tập ứng viên từ 192 xuống 18 chunk thuộc 2 tài liệu, kéo tài liệu gold lên hạng 1 — mà chunk chứa đáp án vẫn không lọt top-3. Filter thu hẹp được *không gian tìm*, nhưng nếu đáp án bị chôn trong một chunk loãng thì nó vẫn thua ngay trong không gian đã thu hẹp.

### Phân tích lỗi (Failure case) — Q3

**Hỏng ở đâu.** Câu hỏi: *"Chọn 'Tự sắp xếp' thì ai trả phí trước, bao lâu được hoàn?"*. Đáp án nằm nguyên văn trong `shopee-phuong-thuc-gui-hang-hoan-tra.md`, mục 2.2: *"Nếu bạn trả hàng qua hình thức Tự sắp xếp, bạn cần thanh toán trước phí trả hàng. Shopee sẽ hỗ trợ bạn phí trả hàng trong vòng 3 - 5 ngày làm việc"*. Với `by_sentences`, câu đó rơi vào chunk `shopee-phuong-thuc-gui-hang-hoan-tra#6`, và chunk này **dài 951 ký tự** — gần gấp đôi độ dài trung bình 485 của chiến lược tôi.

**Vì sao.** 951 ký tự đó là 3 câu, nhưng là 3 câu rất dài, gộp chung ít nhất năm chủ đề: miễn phí trả hàng qua ĐVVC, hình thức Tự sắp xếp, đơn Shopee Mall, hoàn Shopee Xu, và mức 25.000/40.000 Xu theo cùng hay khác tỉnh. Vector của chunk là **trung bình ngữ nghĩa của cả năm thứ**, nên tín hiệu "ai trả trước / bao lâu được hoàn" bị pha loãng. Nó thua hai chunk hướng dẫn thao tác từng bước (0.5246 và 0.5049) vốn khớp với lối diễn đạt "chọn… thì…" của câu hỏi.

Gốc rễ là một khuyết điểm thiết kế của `SentenceChunker`: **nó đếm câu chứ không giới hạn độ dài.** Đo trên corpus này, chunk dài nhất của tôi là **2.493 ký tự**, trong khi cả ba chiến lược còn lại đều chặn cứng ở 500. Số câu là xấp xỉ rất tệ cho kích thước ngữ nghĩa khi độ dài câu chênh nhau hàng chục lần — mà văn bản quy định thì đúng là loại văn bản có câu dài bất thường.

**Một sắc thái phải nói cho sòng phẳng.** Ở lượt **không** lọc, top-1 là `chinh-sach-tra-hang-hoan-tien#34` (0.5857): *"Theo hình thức 'Tự sắp xếp': Người Mua cần thanh toán trước chi phí vận chuyển cho việc trả hàng. Shopee sẽ hỗ trợ hoàn lại một phần chi phí…"*. Đoạn này **trả lời đúng vế "ai trả trước"**, chỉ diễn đạt khác và thiếu mốc 3–5 ngày. Tức retrieval không mù hoàn toàn; thước đo khớp chuỗi chính xác chấm 0 vì chunk gold được chỉ định chưa bao giờ nổi lên. Điểm 0/2 là đúng với câu hỏi "chunk gold có lên không", nhưng nghiêm khắc hơn thực tế nếu câu hỏi là "agent có trả lời được không".

**Đề xuất sửa — tôi đo chứ không đoán.** Phản xạ đầu tiên là "chặn thêm độ dài cho chunk". Tôi thử và **nó làm hỏng chính Q3**:

| Cấu hình | chunks | Độ dài TB | Số evidence còn nguyên vẹn |
| --- | --- | --- | --- |
| 3 câu, không cap *(hiện tại)* | 190 | 485 | **5/5** |
| 3 câu, cap 800 ký tự | 209 | 441 | 4/5 — **Q3 bị cắt đôi** |
| 3 câu, cap 400 ký tự | 283 | 325 | 4/5 — **Q3 bị cắt đôi** |
| 2 câu, cap bất kỳ | 284–372 | 247–324 | 4/5 — **Q3 bị cắt đôi** |

Lý do: đáp án Q3 trải qua **hai câu liền nhau**, và cả hai nằm cuối một nhóm 3 câu. Hạ trần câu xuống 2, hoặc chèn một ranh giới ký tự vào giữa, đều rơi đúng vào khe giữa hai câu đó và xoá sổ đáp án khỏi mọi chunk. Chunk nhỏ hơn thì đặc hơn, nhưng mỗi ranh giới mới là một cơ hội mới để cắt trúng đáp án — và ở đây nó cắt trúng thật.

Phương án thực sự dùng được là **cửa sổ trượt có chồng lấn** thay vì cắt nhỏ hơn, vì `SentenceChunker` hiện tại có overlap bằng 0 nên mỗi thông tin chỉ có **đúng một cơ hội** lọt top-k:

| Cấu hình | chunks | Độ dài TB | evidence còn nguyên | Số bản sao của evidence (Q1…Q5) |
| --- | --- | --- | --- | --- |
| 3 câu, bước 3 *(hiện tại)* | 190 | 485 | 5/5 | 1 · 1 · 1 · 1 · 1 |
| 3 câu, bước 1 | 552 | 488 | 5/5 | 3 · 3 · **2** · 3 · 3 |
| **2 câu, bước 1** | 558 | **326** | **5/5** | 2 · 2 · **1** · 2 · 2 |

Cấu hình `2 câu, bước 1` **hạ độ dài trung bình từ 485 xuống 326 mà vẫn giữ trọn cả 5 đáp án** — đúng thứ mà cách cap ký tự không làm được. Cái giá là 558 chunk, gấp 2,9 lần, tức chi phí embedding gấp 2,9 lần. Với corpus 6 tài liệu thì đổi chác này rõ ràng là đáng.

### Điều hay nhất tôi học được từ thành viên khác

Hoàng chạy `RecursiveChunker` trên **đúng corpus, đúng 5 câu hỏi, đúng backend OpenAI** và đạt EvidenceHit@3 = 5/5, điểm 9/10, MRR 0.900 — tức hơn tôi đúng ở câu Q3 mà tôi trượt. Tôi không muốn dừng ở kết luận "recursive tốt hơn", nên đo lại xem **chunk chứa đáp án Q3 dài bao nhiêu ở từng chiến lược**:

| Chiến lược | chunks | Độ dài TB | Chunk dài nhất | Độ dài chunk chứa đáp án Q3 |
| --- | --- | --- | --- | --- |
| Sentence *(của tôi)* | 190 | 485 | **2.493** | **951** |
| Recursive *(Hoàng)* | 265 | 348 | 500 | **336** |
| Fixed Size *(Mi)* | 210 | 491 | 500 | 500 |
| Heading *(Dũng)* | 314 | 330 | 500 | 493 |

Đây mới là lời giải thích thật. Cả ba chiến lược kia đều **có trần độ dài**; chỉ chiến lược của tôi là không. Với Q3, Recursive gói đáp án vào 336 ký tự còn tôi gói vào 951 — cùng một câu văn, nhưng vector của Hoàng nói gần như chỉ về chuyện đó, còn vector của tôi nói về năm chuyện cùng lúc. Chênh lệch 5/5 so với 4/5 giữa hai chúng tôi quy về đúng con số này.

Bài học tôi mang đi: **trên văn bản quy định, bố cục (xuống dòng, đoạn, mục) mang nhiều cấu trúc hơn dấu chấm câu.** `RecursiveChunker` cắt theo `\n\n` và `\n` trước rồi mới tới `". "`, nên nó ăn theo cách người soạn thảo đã chia nội dung. `SentenceChunker` chỉ thấy dấu chấm, nên ở một danh sách điều kiện viết liền thành một câu dài, nó không thấy ranh giới nào cả. Nếu làm lại, tôi vẫn gom theo câu để không cắt giữa mệnh đề, nhưng sẽ thêm trần độ dài **hạ xuống theo ranh giới dòng chứ không phải theo số ký tự** — giữ ưu điểm của cả hai mà tránh đúng cái bẫy đã xoá sổ đáp án Q3 ở bảng đo phía trên.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
| --- | --- |
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 4 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **55 / 60** |

**Giải thích phần tự trừ điểm:**

- **Hướng tiếp cận (−1):** `SentenceChunker` còn lỗi đã biết ở chữ viết tắt và số thập phân, chưa khắc phục. Mục 5 còn cho thấy một khuyết điểm thứ hai mà lúc viết code tôi chưa lường: chiến lược này **không có trần độ dài**, nên một chunk có thể phình tới 2.493 ký tự.
- **Dự đoán similarity (−1):** chỉ 2/5 dự đoán đúng, và cả hai đều là ăn may chứ không phải nhờ hiểu đúng — do backend mock. Tôi bù lại bằng phần đối chiếu với dải điểm OpenAI thật ở mục 5, nhưng bản thân 5 cặp câu thì vẫn chưa được đo lại bằng embedder có ngữ nghĩa.
- **Kết quả truy xuất (−3):** runner chấm **7/10** (EvidenceHit@3 = 4/5, MRR = 0.700). Tôi lấy đúng con số runner trả về, không tự nâng. Câu trượt là Q3, và tôi đã truy được nguyên nhân tới mức chunk cụ thể (951 ký tự) cùng một đề xuất sửa đã đo kiểm.
