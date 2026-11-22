import json
import csv
import math

with open("output/positional_index.json", "r", encoding="utf-8") as f:
    positional_index = json.load(f)

idf = {}

with open("output/idf.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        idf[row["term"]] = float(row["idf"])

normalized_tfidf = {}

with open("output/normalized_tfidf.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    docs = reader.fieldnames[1:]

    for row in reader:
        term = row["term"]
        normalized_tfidf[term] = {}

        for doc in docs:
            normalized_tfidf[term][doc] = float(row[doc])

def search_phrase(phrase):
    words = phrase.lower().split()

    first_word = words[0]

    if first_word not in positional_index:
        return {}

    result = {}

    first_docs = positional_index[first_word]

    for doc_id, positions in first_docs.items():

        starts = []

        for start in positions:

            ok = True

            for i in range(1, len(words)):

                word = words[i]

                if word not in positional_index:
                    ok = False
                    break

                if doc_id not in positional_index[word]:
                    ok = False
                    break

                if (start + i) not in positional_index[word][doc_id]:
                    ok = False
                    break

            if ok:
                starts.append(start)

        if starts:
            result[doc_id] = starts

    return result

def create_query_vector(phrase):
    words = phrase.lower().split()

    query_vector = {}

    for word in words:
        if word not in idf:
            continue

        query_vector[word] = idf[word]

    length = math.sqrt(sum(value ** 2 for value in query_vector.values()))

    if length == 0:
        return {}

    for word in query_vector:
        query_vector[word] = query_vector[word] / length

    return query_vector

def cosine_similarity(query_vector, doc_id):
    score = 0

    for word, weight in query_vector.items():
        score += weight * normalized_tfidf[word][doc_id]

    return score

def rank_results(phrase):
    matches = search_phrase(phrase)

    query_vector = create_query_vector(phrase)

    results = []

    for doc_id in matches:
        score = cosine_similarity(query_vector, doc_id)
        results.append((doc_id, score))

    results.sort(key=lambda item: item[1], reverse=True)

    return results

queries = [
    "patient care",
    "medical research"
]


for query in queries:

    print("\nPhrase Query:", query)

    matches = search_phrase(query)

    print("\nMatching Documents and Starting Positions:")

    for doc_id, starts in matches.items():
        print(f"{doc_id}: {starts}")

    query_vector = create_query_vector(query)

    print("\nQuery Vector:")

    for word, weight in query_vector.items():
        print(f"{word}: {weight}")

    print("\nSimilarity Scores and Ranking:")

    results = rank_results(query)

    for doc_id, score in results:
        print(f"{doc_id}: {score}")


def boolean_queries():

    phrase1 = "patient care"
    phrase2 = "medical research"

    p1 = search_phrase(phrase1)
    p2 = search_phrase(phrase2)

    docs1 = set(p1.keys())
    docs2 = set(p2.keys())

    and_docs = docs1 & docs2
    and_not_docs = docs1 - docs2

    print("\nBoolean Query: patient care AND medical research")
    print("Matching Documents:")
    print(sorted(and_docs))

    query_terms = phrase1.split() + phrase2.split()
    query_vector = create_query_vector(" ".join(query_terms))

    print("Query Vector:")
    for word, weight in query_vector.items():
        print(f"{word}: {weight}")

    print("Similarity Scores and Ranking:")

    results = []

    for doc_id in and_docs:
        score = cosine_similarity(query_vector, doc_id)
        results.append((doc_id, score))

    results.sort(key=lambda item: item[1], reverse=True)

    for doc_id, score in results:
        print(f"{doc_id}: {score}")

    print("\nBoolean Query: patient care AND NOT medical research")
    print("Matching Documents:")
    print(sorted(and_not_docs))

    query_terms = phrase1.split()
    query_vector = create_query_vector(" ".join(query_terms))

    print("Query Vector:")
    for word, weight in query_vector.items():
        print(f"{word}: {weight}")

    print("Similarity Scores and Ranking:")

    results = []

    for doc_id in and_not_docs:
        score = cosine_similarity(query_vector, doc_id)
        results.append((doc_id, score))

    results.sort(key=lambda item: item[1], reverse=True)

    for doc_id, score in results:
        print(f"{doc_id}: {score}")


if __name__ == "__main__":
    boolean_queries()