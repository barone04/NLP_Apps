Nhận xét

### Lab 1: Text Tokenization

- **Mục tiêu và triển khai**: Lab 1 tập trung vào việc triển khai hai bộ tách từ: `SimpleTokenizer` và `RegexTokenizer`.  
  - `SimpleTokenizer`: chuyển văn bản thành chữ thường, tách dựa trên khoảng trắng và xử lý dấu câu cơ bản (., !, ?) bằng cách chèn khoảng trắng.  
  - `RegexTokenizer`: sử dụng biểu thức chính quy (`\w+|[^\w\s]`) để tách từ chi tiết hơn, bao gồm dấu nháy và dấu gạch ngang.  
  - Nhiệm vụ cuối cùng áp dụng cả hai trên mẫu UD_English-EWT.
- **Điểm mạnh**: 
  - `SimpleTokenizer` đơn giản, hiệu quả với văn bản cơ bản, dễ bảo trì.
  - `RegexTokenizer` linh hoạt, xử lý tốt các trường hợp phức tạp như từ rút gọn (e.g., "isn't" → ["isn", "'", "t"]).
  - Kết quả trên mẫu UD_English-EWT cho thấy khả năng áp dụng thực tế, đặc biệt khi so sánh hai cách tiếp cận.
- **Hạn chế**: 
  - `SimpleTokenizer` không xử lý tốt từ ghép không có khoảng trắng (e.g., "well-known") và tách dấu câu lặp thành nhiều token (e.g., "..." → [".", ".", "."]).
  - `RegexTokenizer` có thể phân mảnh từ quá mức (e.g., "let's" → ["let", "'", "s"]), cần điều chỉnh tùy ngữ cảnh.
  - Dữ liệu mock thay vì thực tế do không truy cập được file gốc, có thể ảnh hưởng đến độ chính xác.
- **Khó khăn**: Lỗi `ModuleNotFoundError` do đường dẫn module sai (giải quyết bằng `sys.path.append`), và phụ thuộc vào mẫu dữ liệu giả lập.

---

### Lab 2: Count Vectorization

- **Mục tiêu và triển khai**: Lab 2 tập trung vào biểu diễn văn bản thành vector số bằng mô hình Bag-of-Words thông qua `CountVectorizer`.  
  - Giao diện `Vectorizer` được định nghĩa với các phương thức `fit`, `transform`, và `fit_transform`.  
  - `CountVectorizer` sử dụng tokenizer từ Lab 1 để xây dựng từ vựng và ma trận tần số, được kiểm tra trên một corpus mẫu.
- **Điểm mạnh**: 
  - `CountVectorizer` hoạt động tốt với tokenizer, tạo từ vựng sắp xếp và ma trận phản ánh tần suất từ chính xác (e.g., "love" xuất hiện 1 lần ở tài liệu 1, 2).
  - Các bài kiểm tra đơn vị trong `test_lab02.py` (sau khi sửa) xác nhận chức năng đúng, như kiểm tra kích thước ma trận và tần suất từ.
  - Tích hợp tốt với `RegexTokenizer`, xử lý dấu câu và từ phức tạp.
- **Hạn chế**: 
  - Chưa áp dụng trên UD_English-EWT thực tế do dữ liệu mock, dẫn đến từ vựng hạn chế.
  - Thiếu các tính năng nâng cao như loại bỏ stop words hoặc giới hạn tần suất tối thiểu, cần bổ sung cho tập dữ liệu lớn.
  - Ban đầu không có test, gây khó khăn trong việc xác nhận (đã sửa bằng cách thêm `test_*`).
- **Khó khăn**: Lỗi `ModuleNotFoundError` tương tự Lab 1 (giải quyết bằng `sys.path.append`), và cần cấu hình test thủ công để PyTest nhận diện.

---
