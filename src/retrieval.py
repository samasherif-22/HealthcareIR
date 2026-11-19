import os
import sys
import json
import csv
import math
from pathlib import Path

from pyspark.sql import SparkSession

# Configure Python for Spark workers
python_executable = sys.executable
os.environ["PYSPARK_PYTHON"] = python_executable
os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "output" / "positional_index.json"
OUTPUT_DIR = BASE_DIR / "output"

DOCS = [f"d{i}" for i in range(1, 11)]
N = len(DOCS)

# Create Spark Session
spark = SparkSession.builder \
    .appName("Healthcare IR Retrieval") \
    .config("spark.pyspark.python", python_executable) \
    .config("spark.pyspark.driver.python", python_executable) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Load Positional Index
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    positional_index = json.load(file)

terms = sorted(positional_index.keys())

# Convert index into RDD
positional_rdd = sc.parallelize(
    list(positional_index.items()), 4
)

# Calculate Raw TF
def calculate_tf(record):
    term, documents = record
    tf_values = {
        doc_id: len(positions)
        for doc_id, positions in documents.items()
    }
    return term, tf_values

tf_rdd = positional_rdd.map(calculate_tf)

# Calculate Weighted TF
def calculate_weighted_tf(record):
    term, doc_tf = record

    weighted_values = {
        doc_id: (
            1 + math.log10(tf)
            if tf > 0 else 0.0
        )
        for doc_id, tf in doc_tf.items()
    }

    return term, weighted_values

weighted_tf_rdd = tf_rdd.map(calculate_weighted_tf)

# Calculate DF and IDF
def calculate_df_idf(record):
    term, documents = record

    df = len(documents)
    idf = math.log10(N / df)

    return term, (df, idf)

stats_rdd = positional_rdd.map(calculate_df_idf)

# Collect results
tf_results = tf_rdd.collect()
weighted_results = weighted_tf_rdd.collect()
stats_results = stats_rdd.collect()

# Build complete matrices
tf_matrix = {}
weighted_matrix = {}
term_statistics = {}

for term, doc_tf in tf_results:
    tf_matrix[term] = {
        doc: doc_tf.get(doc, 0)
        for doc in DOCS
    }

for term, doc_wtf in weighted_results:
    weighted_matrix[term] = {
        doc: doc_wtf.get(doc, 0.0)
        for doc in DOCS
    }

for term, (df, idf) in stats_results:
    term_statistics[term] = {
        "DF": df,
        "IDF": idf
    }

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Write TF Matrix
with open(OUTPUT_DIR / "tf_matrix.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["term"] + DOCS)

    for term in terms:
        writer.writerow(
            [term] + [tf_matrix[term][doc] for doc in DOCS]
        )

# Write Weighted TF Matrix
with open(OUTPUT_DIR / "weighted_tf.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["term"] + DOCS)

    for term in terms:
        writer.writerow(
            [term] + [
                f"{weighted_matrix[term][doc]:.4f}"
                if weighted_matrix[term][doc] else "0"
                for doc in DOCS
            ]
        )

# Write DF and IDF
with open(OUTPUT_DIR / "term_statistics.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["term", "DF", "IDF"])

    for term in terms:
        writer.writerow([
            term,
            term_statistics[term]["DF"],
            f'{term_statistics[term]["IDF"]:.4f}'
        ])


# =====================================================================
# MEMBER 3: TF-IDF, Document Vector Lengths, L2 Normalization
# (uses weighted_matrix and term_statistics from Member 2's section above,
#  at full precision, not the rounded CSV values)
# =====================================================================

# TF-IDF = Weighted TF x IDF
tfidf_matrix = {
    term: {
        doc: weighted_matrix[term][doc] * term_statistics[term]["IDF"]
        for doc in DOCS
    }
    for term in terms
}

# Document vector length = sqrt(sum of squared TF-IDF values of the document)
doc_lengths = {
    doc: math.sqrt(sum(tfidf_matrix[term][doc] ** 2 for term in terms))
    for doc in DOCS
}

# Normalized TF-IDF = TF-IDF / document length
normalized_matrix = {
    term: {
        doc: tfidf_matrix[term][doc] / doc_lengths[doc]
        for doc in DOCS
    }
    for term in terms
}

# Write TF-IDF
with open(OUTPUT_DIR / "tfidf.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["term"] + DOCS)

    for term in terms:
        writer.writerow(
            [term] + [
                f"{tfidf_matrix[term][doc]:.4f}"
                if tfidf_matrix[term][doc] else "0"
                for doc in DOCS
            ]
        )

# Write Document Vector Lengths
with open(OUTPUT_DIR / "vector_lengths.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["doc", "length"])

    for doc in DOCS:
        writer.writerow([doc, f"{doc_lengths[doc]:.4f}"])

# Write Normalized TF-IDF
with open(OUTPUT_DIR / "normalized_tfidf.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["term"] + DOCS)

    for term in terms:
        writer.writerow(
            [term] + [
                f"{normalized_matrix[term][doc]:.4f}"
                if normalized_matrix[term][doc] else "0"
                for doc in DOCS
            ]
        )

# Member 3 validation
print("\nDocument lengths:")
for doc in DOCS:
    print(doc, f"{doc_lengths[doc]:.4f}")

print("\nCheck: sum of squares of each normalized column (should be 1):")
for doc in DOCS:
    print(doc, round(sum(normalized_matrix[t][doc] ** 2 for t in terms), 4))

# =====================================================================
# END OF MEMBER 3 SECTION
# =====================================================================


# Validation summary
total_tokens = sum(
    sum(row.values()) for row in tf_matrix.values()
)

print("Number of terms:", len(terms))
print("Total tokens:", total_tokens)

print("\nSample Weighted TF values:")
for term, doc in [("patient", "d5"), ("care", "d5"),
                  ("monitoring", "d9")]:
    print(
        term, doc,
        f"{weighted_matrix[term][doc]:.4f}"
    )

print("\nSample DF / IDF values:")
for term in ["patient", "medical", "care"]:
    print(
        term,
        "DF:", term_statistics[term]["DF"],
        "IDF:", f'{term_statistics[term]["IDF"]:.4f}'
    )

print("\nFiles written:")
print("output/tf_matrix.csv")
print("output/weighted_tf.csv")
print("output/term_statistics.csv")
print("output/tfidf.csv")
print("output/vector_lengths.csv")
print("output/normalized_tfidf.csv")

spark.stop()