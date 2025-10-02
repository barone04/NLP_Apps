# Báo cáo Lab – NLP Pipeline với Spark

## 1. Các bước triển khai

Quy trình triển khai được thực hiện như sau:

1. **Khởi tạo Spark Session**

   * Tạo Spark session với `SparkSession.builder` để chạy Spark ML pipeline trên môi trường local.
   * Bật Spark UI tại [http://localhost:4040](http://localhost:4040) để theo dõi quá trình thực thi.

2. **Đọc dữ liệu**

   * Dữ liệu sử dụng là **một tập con của bộ C4** (`c4-train.00000-of-01024.json.gz`).
   * Nạp dữ liệu vào Spark DataFrame bằng `spark.read.json()`.
   * Giới hạn 1000 dòng để giảm thời gian xử lý trong bài lab.

3. **Định nghĩa pipeline xử lý văn bản**

   * **Tokenization**: sử dụng `RegexTokenizer` để tách văn bản thành token, dựa trên khoảng trắng và dấu câu.
   * **Loại bỏ stop words**: áp dụng `StopWordsRemover` để loại bỏ các từ dừng thông dụng trong tiếng Anh.
   * **Vector hóa (Term Frequency)**: sử dụng `HashingTF` để ánh xạ token sang vector đặc trưng cố định (20,000 chiều).
   * **Vector hóa (Inverse Document Frequency)**: sử dụng `IDF` để điều chỉnh trọng số dựa trên độ quan trọng của từ trong toàn bộ văn bản.

4. **Huấn luyện và áp dụng pipeline**

   * Gộp toàn bộ các bước trên vào `Pipeline`.
   * Tiến hành `fit` (huấn luyện) pipeline với dữ liệu, sau đó `transform` để sinh vector TF-IDF.

5. **Ghi log và xuất kết quả**

   * Ghi lại các chỉ số hiệu năng (thời gian huấn luyện, thời gian transform, kích thước từ vựng…) vào `../log/lab17_metrics.log`.
   * Ghi kết quả của 20 bản ghi đầu tiên vào `../results/lab17_pipeline_output.txt`.

---

## 2. Cách chạy chương trình và ghi log

* **Chuẩn bị môi trường**:

  * Cài đặt Java 17, Spark.
  * Cài `sbt` (Scala Build Tool).
  * Tải tập dữ liệu [C4 dataset trên Hugging Face](https://huggingface.co/datasets/allenai/c4) và đặt vào thư mục `NLP&Apps/data/`.

* **Chạy chương trình**:

  ```bash
  cd NLP&Apps/spark_labs
  sbt "runMain com.baro.spark.Lab17_NLPPipeline"
  ```

* **Kết quả đầu ra**:

  * **Log**: các thông tin về hiệu năng được ghi vào:

    ```
    NLP&Apps/log/lab17_metrics.log
    ```
  * **Kết quả**: dữ liệu sau khi transform được ghi vào:

    ```
    NLP&Apps/results/lab17_pipeline_output.txt
    ```
  * Trong khi chạy có thể theo dõi Spark jobs tại [http://localhost:4040](http://localhost:4040).

---

## 3. Giải thích kết quả

* Pipeline đã thực hiện thành công các bước:

  * Tách văn bản thành token.
  * Loại bỏ stop words.
  * Sinh vector TF-IDF cho mỗi văn bản.

* **Kết quả mẫu** bao gồm:

  * Đoạn text gốc (tối đa 100 ký tự đầu).
  * Vector TF-IDF dạng thưa (sparse vector).

* **Thông số hiệu năng**:

  * Ghi lại thời gian huấn luyện và transform (tính bằng giây).
  * Tính toán kích thước từ vựng thực tế sau khi loại bỏ stop words.
  * Nếu từ vựng thực tế lớn hơn 20,000 (số chiều của HashingTF), log sẽ cảnh báo có thể xảy ra **hash collision**.

---

## 4. Khó khăn gặp phải và cách giải quyết

1. **Kích thước dữ liệu quá lớn**

   * Bộ C4 có dung lượng hàng trăm GB.
   * Giải pháp: chỉ lấy 1000 dòng để xử lý thử nghiệm trong bài lab.

2. **Lỗi đường dẫn dữ liệu**

   * Ban đầu Spark báo lỗi `No such file or directory`.
   * Giải pháp: đặt file vào `NLP&Apps/data/` và chỉnh lại đường dẫn trong code.

3. **Regex trong Tokenizer**

   * Regex cấu hình ban đầu chưa chính xác, gây tách token sai.
   * Giải pháp: sửa regex thành `\\s+|[.,;!?()\"']`.

4. **Lỗi ClassNotFoundException trong sbt**

   * Xuất hiện lỗi `com.baro.spark.Lab17_NLPPipeline not found`.
   * Giải pháp: đảm bảo file được đặt trong `src/main/scala/com/baro/spark/` và chạy bằng fully qualified name.

---

## 5. Tài liệu tham khảo

* [Apache Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Hugging Face – C4 Dataset](https://huggingface.co/datasets/allenai/c4)
* [Apache Spark RegexTokenizer API](https://spark.apache.org/docs/latest/ml-features.html#tokenizer)

---

## 6. Sử dụng mô hình pre-trained

* **Ghi chú**: Bài lab này **không sử dụng mô hình huấn luyện sẵn (pre-trained model)**.
* Chỉ dùng các công cụ xử lý NLP cơ bản của Spark (RegexTokenizer, StopWordsRemover, HashingTF, IDF).

---

# --- Update Request ---

Dưới đây là các cập nhật đã thực hiện cho mã nguồn Lab17_NLPPipeline.scala dựa trên các yêu cầu được đề xuất:

1. Tùy chỉnh giới hạn tài liệu
- Mô tả: Thêm biến limitDocuments để dễ dàng thay đổi số lượng tài liệu được xử lý thay vì sử dụng giá trị cố định.
- Thực hiện: Biến limitDocuments đã được khai báo và sử dụng trong lệnh limit(limitDocuments) khi đọc dữ liệu từ tệp JSON. Điều này cho phép người dùng điều chỉnh số tài liệu xử lý bằng cách thay đổi giá trị của biến này.

2. Đo lường hiệu suất chi tiết
- Mô tả: Thêm mã để đo và in thời gian thực thi cho từng giai đoạn chính như đọc/ghi dữ liệu, huấn luyện pipeline, và xử lý dữ liệu.
- Thực hiện:
* Thời gian đọc dữ liệu được đo bằng System.nanoTime() từ khi bắt đầu đọc đến khi hoàn thành initialDF.count().
* Thời gian huấn luyện pipeline được đo trong quá trình gọi pipeline.fit(initialDF).
* Thời gian xử lý dữ liệu được đo trong quá trình gọi pipelineModel.transform(initialDF).
* Kết quả thời gian được in ra dưới dạng giây với độ chính xác hai chữ số thập phân.

3. Chuẩn hóa vector
- Mô tả: Thêm lớp Normalizer vào pipeline để chuẩn hóa vector đặc trưng TF-IDF.
- Thực hiện: Lớp Normalizer đã được thêm vào pipeline với cột đầu vào là features (vector TF-IDF) và cột đầu ra là normFeatures. Chuẩn L2 (Euclidean norm) được sử dụng với tham số setP(2.0) để đảm bảo các vector có độ dài bằng 1, giúp cải thiện tính toán độ tương đồng cosine.

4. Tìm kiếm tài liệu tương đồng
- Mô tả:Chọn bất kỳ tài liệu nào từ tập dữ liệu đã xử lý. Tính toán độ tương đồng cosine giữa vector của tài liệu đó và tất cả các tài liệu khác. In ra 5 tài liệu có độ tương đồng cao nhất.
- Thực hiện:
Tài liệu truy vấn được chọn là tài liệu có id = 0 (sau khi thêm cột id bằng monotonically_increasing_id()).
Một UDF được định nghĩa để tính độ tương đồng cosine dựa trên tích vô hướng của các vector đã chuẩn hóa.
Kết quả được sắp xếp theo thứ tự giảm dần của độ tương đồng và in ra 5 tài liệu tương đồng nhất.


- Kết quả:
![Query Document](image/query_document.png)
![Result Documents](image/result_documents.png)


### Kết luận
Các cập nhật đã được tích hợp thành công vào mã nguồn Lab17_NLPPipeline.scala, cho phép tùy chỉnh số lượng tài liệu, đo lường hiệu suất chi tiết, chuẩn hóa vector, và tìm kiếm tài liệu tương đồng một cách hiệu quả. Hình ảnh minh họa cho thấy quá trình chọn tài liệu truy vấn và danh sách 5 tài liệu tương đồng nhất, với độ tương đồng cosine được tính toán chính xác.

---