package com.baro.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Normalizer}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
import breeze.linalg.{DenseVector => BreezeVector}

object Lab17_NLPPipeline {

  def main(args: Array[String]): Unit = {

    // Khởi tạo Spark Session
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // Cấu hình
    val dataPath = "../data/c4-train.00000-of-01024.json.gz"
    val limitDocuments = 2000

    // 1. --- Đọc Dữ liệu ---
    val readStartTime = System.nanoTime()

    // Đọc dữ liệu và thêm cột id
    val initialDF = spark.read.json(dataPath)
      .limit(limitDocuments)
      .withColumn("id", monotonically_increasing_id())

    // Kích hoạt việc đọc dữ liệu (do tính chất Lazy của Spark)
    val readCount = initialDF.count()

    val readDuration = (System.nanoTime() - readStartTime) / 1e9d
    println(f"\n--> Data reading and initial count of $readCount records took $readDuration%.2f seconds.")

    // --- Xây dựng Pipeline NLP ---
    // 2. Tokenization
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // 3. Stop Words Removal
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. HashingTF (Term Frequency)
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000)

    // 5. IDF (Inverse Document Frequency)
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features") // Vector TF-IDF

    // 6. Normalizer (Vector Normalization - Chuẩn hóa vector TF-IDF)
    val normalizer = new Normalizer()
      .setInputCol(idf.getOutputCol)
      .setOutputCol("normFeatures") // Output: Vector TF-IDF đã được chuẩn hóa (độ dài = 1)
      .setP(2.0) // Sử dụng chuẩn L2 (Euclidean norm)

    // 7. Assemble the Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // --- Huấn luyện Pipeline và Đo thời gian ---
    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline **fitting (training)** took $fitDuration%.2f seconds.")

    // --- Biến đổi Dữ liệu và Đo thời gian ---
    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()

    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()

    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data **transformation (processing)** of $transformCount records took $transformDuration%.2f seconds.")

    // --- Tính toán Vocabulary Size và Đo thời gian ---
    val vocabStartTime = System.nanoTime()
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    val vocabDuration = (System.nanoTime() - vocabStartTime) / 1e9d
    println(f"--> Vocabulary size calculation took $vocabDuration%.2f seconds.")
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // Hiển thị mẫu
    println("\nSample of transformed data (TF-IDF vs. Normalized TF-IDF):")
    transformedDF.select("text", "features", "normFeatures").show(5, truncate = 50)

    // --- 8. Tìm kiếm Tài liệu Tương đồng ---
    println("\n" + "="*80)
    println("--- 8. Finding Similar Documents (Cosine Similarity) ---")

    // Chọn Tài liệu truy vấn (ID 0)
    val queryDocument = transformedDF
      .select("id", "text", "normFeatures")
      .filter($"id" === 0)
      .first()

    val queryText = queryDocument.getAs[String]("text")
    val queryVector = queryDocument.getAs[org.apache.spark.ml.linalg.Vector]("normFeatures")

    println(s"\nQuery Document (ID 0):")
    println("="*20)
    println(s"${queryText.substring(0, Math.min(queryText.length, 150))}...")
    println("="*80)

    // Định nghĩa UDF tính Cosine Similarity
    val calculateSimilarity = udf((v1: org.apache.spark.ml.linalg.Vector) => {
      val v1Array = v1.toArray
      val queryArray = queryVector.toArray
      // Tính tích vô hướng
      val dotProduct = v1Array.zip(queryArray).map { case (x, y) => x * y }.sum
      // Vì các vector đã được chuẩn hóa (L2 norm = 1), cosine similarity = tích vô hướng
      dotProduct
    })

    // Áp dụng UDF và Tìm kiếm Top 5
    val similarityDF = transformedDF
      .withColumn("similarity", calculateSimilarity($"normFeatures"))
      // Loại bỏ tài liệu gốc
      .filter($"id" =!= 0)
      .select("id", "text", "similarity")

    println("\nTop 5 Most Similar Documents:")
    println("="*80)

    val topSimilarDocs = similarityDF
      .orderBy(col("similarity").desc)
      .limit(5)
      .collect()

    // In kết quả Top 5
    topSimilarDocs.zipWithIndex.foreach { case (row, index) =>
      val sim = row.getAs[Double]("similarity")
      val docText = row.getAs[String]("text")
      val docId = row.getAs[Long]("id")

      println(f"${index + 1}. ID: $docId | Similarity: $sim%.4f")
      println(s"   Text: ${docText.substring(0, Math.min(docText.length, 100))}...")
    }
    println("="*80)

    // --- Ghi Metrics và Kết quả ra File ---
    val n_results = 20
    val writeStartTime = System.nanoTime()
    // Lấy 20 kết quả đã được chuẩn hóa để ghi ra file
    val results = transformedDF.select("text", "normFeatures").take(n_results)

    // Ghi Metrics vào log folder
    val log_path = "../log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"1. Data Reading Duration: $readDuration%.2f seconds")
      logWriter.println(f"2. Pipeline Fitting (Training) Duration: $fitDuration%.2f seconds")
      logWriter.println(f"3. Data Transformation (Processing) Duration: $transformDuration%.2f seconds")
      logWriter.println(f"4. Vocabulary Calculation Duration: $vocabDuration%.2f seconds")
      logWriter.println(s"Total documents processed: $transformCount (out of $limitDocuments requested)")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      logWriter.println(s"Vector Normalization: L2 (Euclidean) used.")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Ghi Dữ liệu đã chuẩn hóa vào results folder
    val result_path = "../results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results - Normalized TF-IDF) ---")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val normFeatures = row.getAs[org.apache.spark.ml.linalg.Vector]("normFeatures")
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Normalized TF-IDF Vector: ${normFeatures.toString}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }

    val writeDuration = (System.nanoTime() - writeStartTime) / 1e9d
    println(f"--> Data writing/logging took $writeDuration%.2f seconds.")

    spark.stop()
    println("Spark Session stopped.")
  }
}