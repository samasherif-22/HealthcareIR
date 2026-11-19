
# ============================================
# Healthcare Information Retrieval Project
# Part A: Spark Positional Index
# ============================================

# 1. Import required libraries

import os
import sys
import json

# os: Used to work with file paths and names.
# sys: Used to get the current Python executable path.
# json: Used to save the positional index as a JSON file.


# 2. Configure Python for Spark

# Tell Spark to use the Python interpreter
# from the current virtual environment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# 3. Create Spark Session

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("HealthcareIR_PositionalIndex")
    .getOrCreate()
)

# Reduce unnecessary Spark log messages.
spark.sparkContext.setLogLevel("ERROR")

# Get Spark Context to work with RDDs.
sc = spark.sparkContext


# 4. Read all documents using wholeTextFiles()

# Each file is read as a pair:
# (file_path, file_content)
documents = sc.wholeTextFiles("data/*.txt")

# Calculate the word count for each document.

document_word_counts = (
    documents
    .map(
        lambda doc: (
            os.path.splitext(os.path.basename(doc[0]))[0],
            len(doc[1].split())
        )
    )
    .sortBy(
        lambda item: int(item[0][1:])
)

)

print("\nDocument Word Counts:\n")

for doc_id, word_count in document_word_counts.collect():
    print(f"{doc_id}: {word_count}")

print(
    "\nTotal Word Count:",
    sum(
        count
        for _, count in document_word_counts.collect()
    )
)


# 5. Extract terms and their positions

def extract_positions(document):

    # Get the file path and its content.
    path, content = document

    # Extract the document ID from the file name.
    # Example: data/d5.txt -> d5
    doc_id = os.path.splitext(os.path.basename(path))[0]

    # Convert all terms to lowercase and split the content.
    terms = content.lower().split()

    # Assign a position to each term, starting from 1.
    # Output format: ((term, doc_id), position)
    return [
        ((term, doc_id), position)
        for position, term in enumerate(terms, start=1)
    ]


# 6. Apply extract_positions to all documents

# flatMap creates one output record for every term.
term_positions = documents.flatMap(extract_positions)


# 7. Group all positions of the same term in the same document

# Example:
# ((patient, d5), [1, 3, 7])
term_doc_positions = (
    term_positions
    .groupByKey()
    .mapValues(lambda positions: sorted(positions))
)


# 8. Group documents belonging to the same term

# Convert:
# ((term, doc_id), positions)
#
# Into:
# (term, (doc_id, positions))
term_documents = term_doc_positions.map(
    lambda item: (
        item[0][0],
        (item[0][1], item[1])
    )
)

# Group all documents for each term.
# Sort document IDs numerically (d1, d2, ..., d10).
# Sort terms alphabetically.
positional_index_rdd = (
    term_documents
    .groupByKey()
    .mapValues(
        lambda docs: dict(
            sorted(
                docs,
                key=lambda item: int(item[0][1:])
            )
        )
    )
    .sortByKey()
)


# 9. Collect the final index from Spark

# Convert the RDD result into a Python dictionary.
positional_index = dict(positional_index_rdd.collect())


# 10. Save the positional index as a JSON file

# Create the output folder if it does not exist.
os.makedirs("output", exist_ok=True)

# Save the index in JSON format.
with open(
    "output/positional_index.json",
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        positional_index,
        output_file,
        indent=4,
        ensure_ascii=False
    )


# 11. Display the results

print("Total Documents:", documents.count())
print("Total Unique Terms:", len(positional_index))

print("\nHealthcare Positional Index:\n")

print(
    json.dumps(
        positional_index,
        indent=4,
        ensure_ascii=False
    )
)


# 12. Stop Spark Session

spark.stop()

print("\nPositional Index created successfully!")
