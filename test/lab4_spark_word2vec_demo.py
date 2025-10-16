import re
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split


def main():
    # Initialize Spark Session
    print("Initializing Spark session...")
    spark = (SparkSession.builder.appName("Word2VecC4Training").master("local[*]").config("spark.driver.memory",
                                                                                          "4g").getOrCreate())

    # Define the path to the dataset
    data_path = "data/c4-train.00000-of-01024.json.gz"
    print(f"Loading dataset from '{data_path}'...")

    # Load the dataset
    # The C4 dataset is in JSON format, with each line being a JSON object.
    # We are interested in the 'text' field.
    try:
        df = spark.read.json(data_path)
    except Exception as e:
        print(f"\nError loading data file: {e}")
        print("Please ensure the dataset 'c4-train.00000-of-01024-30K.json' is in the same directory.")
        spark.stop()
        return

    print("Dataset loaded successfully.")
    df.printSchema()

    # Preprocessing.
    print("\nPreprocessing text data...")
    # 1. Select the text column and convert to lowercase
    # 2. Remove punctuation and special characters (keep only letters and spaces)
    # 3. Split the text into an array of words

    # Using DataFrame transformations for preprocessing
    processed_df = df.select(
        split(
            regexp_replace(lower(col("text")), r'[^a-z\s]', ''),
            ' '
        ).alias("words")
    )

    # Remove rows where the 'words' array is empty after cleaning
    processed_df = processed_df.filter(col("words").isNotNull())
    print("Preprocessing complete.")
    print("Sample of processed data:")
    processed_df.show(5, truncate=50)

    # Configure and train the Word2Vec model
    print("\nConfiguring and training the Word2Vec model...")
    word2vec = Word2Vec(
        vectorSize=100,  # Dimension of the word vectors
        minCount=5,  # Minimum word frequency to be included
        inputCol="words",  # Input column name (array of strings)
        outputCol="vectors"  # Output column name (vector representation)
    )

    start = time.perf_counter()
    model = word2vec.fit(processed_df)
    end = time.perf_counter()
    print("Model training complete.")
    print(f"Training time: {end - start}")

    # Demonstrate the model
    # Find synonyms for a word
    search_word = "computer"
    num_synonyms = 5
    print(f"\nFinding {num_synonyms} synonyms for the word '{search_word}'...")

    try:
        synonyms = model.findSynonyms(search_word, num_synonyms)
        print(f"Top {num_synonyms} words similar to '{search_word}':")
        synonyms.show()
    except Exception as e:
        print(f"\nCould not find synonyms for '{search_word}'. It might not be in the vocabulary.")
        print(f"(This can happen if it appears fewer than 'minCount' times).")

    # Stop the Spark session
    print("\nStopping the Spark session.")
    spark.stop()


if __name__ == "__main__":
    main()
