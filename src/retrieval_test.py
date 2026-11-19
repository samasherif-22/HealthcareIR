
import csv
import os

OUTPUT_DIR = "output"


def read_csv(name):
    with open(
        os.path.join(OUTPUT_DIR, name),
        encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


# Load output files
tf_rows = {
    row["term"]: row
    for row in read_csv("tf_matrix.csv")
}

wtf_rows = {
    row["term"]: row
    for row in read_csv("weighted_tf.csv")
}

stats = {
    row["term"]: row
    for row in read_csv("term_statistics.csv")
}


# Test 1: Number of terms
assert len(tf_rows) == 28
assert len(wtf_rows) == 28
assert len(stats) == 28

print("Test 1 Passed: 28 terms in all three files.")


# Test 2: Raw TF reference values
assert tf_rows["care"]["d5"] == "2"
assert tf_rows["patient"]["d5"] == "3"
assert tf_rows["monitoring"]["d9"] == "2"
assert tf_rows["patient"]["d9"] == "2"

print("Test 2 Passed: TF reference values are correct.")


# Test 3: Weighted TF reference values
assert wtf_rows["patient"]["d5"] == "1.4771"
assert wtf_rows["monitoring"]["d9"] == "1.3010"
assert wtf_rows["care"]["d5"] == "1.3010"
assert wtf_rows["patient"]["d9"] == "1.3010"
assert wtf_rows["patient"]["d3"] == "0"

print("Test 3 Passed: Weighted TF reference values are correct.")


# Test 4: DF / IDF reference values
expected_values = {
    "patient": (9, "0.0458"),
    "medical": (5, "0.3010"),
    "care": (4, "0.3979"),
    "clinical": (3, "0.5229"),
    "research": (3, "0.5229"),
    "safety": (3, "0.5229"),
    "treatment": (3, "0.5229"),
    "monitoring": (2, "0.6990"),
    "records": (2, "0.6990"),
    "recovery": (2, "0.6990"),
    "supports": (2, "0.6990")
}

for term, (expected_df, expected_idf) in expected_values.items():
    assert int(stats[term]["DF"]) == expected_df
    assert stats[term]["IDF"] == expected_idf

for term, row in stats.items():
    if int(row["DF"]) == 1:
        assert row["IDF"] == "1.0000"

print("Test 4 Passed: DF and IDF reference values are correct.")


# Test 5: Total number of tokens
total_tokens = sum(
    int(value)
    for row in tf_rows.values()
    for column, value in row.items()
    if column != "term"
)

assert total_tokens == 60

print("Test 5 Passed: total token count is 60.")

print("\nAll Member 2 validation tests passed successfully!")
