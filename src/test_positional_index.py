
# ============================================
# HealthcareIR - Positional Index Validation
# ============================================

import json
import os

# Load the generated positional index.
with open(
    "output/positional_index.json",
    "r",
    encoding="utf-8"
) as file:
    index = json.load(file)


# Test 1: Check the number of unique terms.
assert len(index) == 28, "Incorrect number of unique terms!"

print("Test 1 Passed: 28 unique terms found.")


# Test 2: Check the expected patient positions.
expected_patient = {
    "d1": [1],
    "d2": [1],
    "d4": [4],
    "d5": [1, 3, 7],
    "d6": [4],
    "d7": [4],
    "d8": [4],
    "d9": [1, 3],
    "d10": [5]
}

assert index["patient"] == expected_patient

print("Test 2 Passed: Patient positions are correct.")


# Test 3: Check the expected care positions.
expected_care = {
    "d1": [2],
    "d5": [2, 8],
    "d6": [5],
    "d9": [6]
}

assert index["care"] == expected_care

print("Test 3 Passed: Care positions are correct.")


# Test 4: Check alphabetical term ordering.
terms = list(index.keys())

assert terms == sorted(terms)

print("Test 4 Passed: Terms are sorted alphabetically.")


# Test 5: Check numeric document ordering.
for term, documents in index.items():

    doc_ids = list(documents.keys())

    expected_order = sorted(
        doc_ids,
        key=lambda doc: int(doc[1:])
    )

    assert doc_ids == expected_order
    # Test 6: Check the word count of each document.

expected_word_counts = {
    "d1": 2,
    "d2": 5,
    "d3": 4,
    "d4": 5,
    "d5": 8,
    "d6": 5,
    "d7": 7,
    "d8": 8,
    "d9": 6,
    "d10": 10
}

for doc_id, expected_count in expected_word_counts.items():

    file_path = f"data/{doc_id}.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    actual_count = len(content.split())

    assert actual_count == expected_count, (
        f"Word count mismatch in {doc_id}: "
        f"Expected {expected_count}, got {actual_count}"
    )

print("Test 6 Passed: All document word counts are correct.")

print("Test 5 Passed: Documents are sorted numerically.")


print("\nAll validation tests passed successfully!")
