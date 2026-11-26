
import streamlit as st
import json
import csv
import math
import re
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

st.set_page_config(
    page_title="Healthcare IR",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# DOCUMENT SORTING
# ==================================================

def document_sort_key(name):
    """
    Natural sorting:
    D1, D2, D3 ... D10
    Also supports names such as patient1, patient2, patient10.
    """
    parts = re.split(r"(\d+)", name)

    return tuple(
        (0, int(part)) if part.isdigit()
        else (1, part.casefold())
        for part in parts
    )


# ==================================================
# DESIGN SYSTEM
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F5F8;
    color: #111827;
    font-family: Inter, sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.3rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 0;
}

h1, h2, h3 {
    color: #111827 !important;
    font-weight: 750 !important;
}

p {
    color: #64748B;
}

.hero {
    background: #111827;
    border-radius: 18px;
    padding: 35px;
    margin-bottom: 28px;
    color: white;
}

.hero-tag {
    color: #60A5FA;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.25;
    margin: 13px 0;
}

.hero-title span {
    color: #F472B6;
}

.hero-desc {
    color: #CBD5E1;
    font-size: 14px;
    max-width: 650px;
    line-height: 1.8;
}

.metric {
    border-radius: 14px;
    padding: 22px;
    color: white;
    min-height: 120px;
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    opacity: .9;
}

.metric-number {
    font-size: 34px;
    font-weight: 800;
    margin-top: 9px;
}

.blue { background: #2563EB; }
.pink { background: #DB2777; }
.red { background: #E5484D; }

.section-tag {
    color: #2563EB;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

.result {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 17px;
    margin: 10px 0;
}

.result-name {
    color: #111827;
    font-weight: 750;
    font-size: 16px;
}

.result-score {
    color: #2563EB;
    font-weight: 800;
    font-size: 19px;
}

.result-position {
    color: #64748B;
    font-size: 12px;
    margin-top: 5px;
}

div.stButton > button {
    background: #2563EB;
    color: white;
    border: 0;
    border-radius: 8px;
    font-weight: 700;
    padding: 9px 20px;
}

div.stButton > button:hover {
    background: #1D4ED8;
    color: white;
    border: 0;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    border-radius: 8px;
}

hr {
    border-color: #E2E8F0;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    documents = {}

    if DATA_DIR.exists():

        file_paths = sorted(
            DATA_DIR.glob("*.txt"),
            key=lambda path: document_sort_key(path.stem)
        )

        for path in file_paths:
            documents[path.stem] = path.read_text(
                encoding="utf-8"
            )

    index_path = OUTPUT_DIR / "positional_index.json"

    if index_path.exists():

        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)

    else:
        index = {}

    idf = {}

    idf_path = OUTPUT_DIR / "idf.csv"

    if idf_path.exists():

        with open(idf_path, encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:
                idf[row["term"]] = float(row["idf"])

    return documents, index, idf


documents, index, idf = load_data()

# Use one consistent order throughout the application.
docs = sorted(
    documents.keys(),
    key=document_sort_key
)


# ==================================================
# INFORMATION RETRIEVAL FUNCTIONS
# ==================================================

def search_term(term):

    term = term.lower().strip()

    return index.get(term, {})


def search_phrase(phrase):

    words = phrase.lower().split()

    if not words:
        return {}

    first_word = words[0]

    if first_word not in index:
        return {}

    matches = {}

    for doc_id, positions in index[first_word].items():

        starts = []

        for start in positions:

            found = True

            for i in range(1, len(words)):

                word = words[i]

                if (
                    word not in index
                    or doc_id not in index[word]
                    or start + i not in index[word][doc_id]
                ):
                    found = False
                    break

            if found:
                starts.append(start)

        if starts:
            matches[doc_id] = starts

    return matches


# ==================================================
# TF-IDF AND NORMALIZATION
# ==================================================

def calculate_normalized_tfidf():

    weighted_tf = {}

    for term, postings in index.items():

        weighted_tf[term] = {}

        for doc in docs:

            tf = len(postings.get(doc, []))

            weighted_tf[term][doc] = (
                1 + math.log10(tf) if tf > 0 else 0
            )

    tfidf = {}

    for term in index:

        tfidf[term] = {}

        for doc in docs:

            tfidf[term][doc] = (
                weighted_tf[term][doc] * idf.get(term, 0)
            )

    lengths = {}

    for doc in docs:

        lengths[doc] = math.sqrt(
            sum(
                tfidf[term][doc] ** 2
                for term in tfidf
            )
        )

    normalized = {}

    for term in tfidf:

        normalized[term] = {}

        for doc in docs:

            normalized[term][doc] = (
                tfidf[term][doc] / lengths[doc]
                if lengths[doc] > 0
                else 0
            )

    return normalized


normalized_tfidf = calculate_normalized_tfidf()


# ==================================================
# QUERY VECTOR AND SIMILARITY
# ==================================================

def create_query_vector(query):

    vector = {}

    for word in query.lower().split():

        if word in idf:
            vector[word] = idf[word]

    length = math.sqrt(
        sum(value ** 2 for value in vector.values())
    )

    if length == 0:
        return {}

    return {
        word: value / length
        for word, value in vector.items()
    }


def similarity(query_vector, doc_id):

    score = sum(
        weight * normalized_tfidf.get(word, {}).get(doc_id, 0)
        for word, weight in query_vector.items()
    )

    return max(0.0, min(1.0, score))


def rank_documents(query, matching_docs):

    vector = create_query_vector(query)

    results = [
        (doc, similarity(vector, doc))
        for doc in matching_docs
    ]

    return sorted(
        results,
        key=lambda item: (
            -item[1],
            document_sort_key(item[0])
        )
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:white'>HEALTHCARE / IR</h2>",
        unsafe_allow_html=True
    )

    st.caption("INFORMATION RETRIEVAL SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Documents",
            "Search",
            "Boolean Search"
        ]
    )

    st.divider()

    st.caption("PROJECT STATUS")

    st.markdown(
        "Dataset loaded" if documents
        else "No documents found"
    )

    st.markdown(
        "Positional index available" if index
        else "Positional index not found"
    )

    st.markdown(
        "TF-IDF data available" if idf
        else "IDF data not found"
    )


# ==================================================
# HEADER
# ==================================================

# Important: HTML lines start at the beginning of the line.
# This prevents Streamlit from interpreting indented HTML as code.

st.markdown("""
<div class="hero">
<div class="hero-tag">HEALTHCARE INFORMATION RETRIEVAL</div>
<div class="hero-title">
Healthcare knowledge,<br>
<span>within reach.</span>
</div>
<div class="hero-desc">
Explore medical documents, search for relevant information,
and understand how the system retrieves and ranks results.
</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.markdown(
        '<div class="section-tag">PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.header("Your document collection")

    total_words = sum(
        len(text.split())
        for text in documents.values()
    )

    c1, c2, c3 = st.columns(3)

    metrics = [
        (c1, len(documents), "DOCUMENTS", "blue"),
        (c2, total_words, "TOTAL WORDS", "pink"),
        (c3, len(index), "UNIQUE TERMS", "red")
    ]

    for col, number, label, color in metrics:

        with col:

            st.markdown(f"""
<div class="metric {color}">
<div class="metric-label">{label}</div>
<div class="metric-number">{number}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("How does the system work?")

    st.write(
        "The system processes text documents and builds an index "
        "that helps retrieve relevant information."
    )

    steps = [
        (
            "01",
            "Document Collection",
            "Read the healthcare text files."
        ),
        (
            "02",
            "Positional Index",
            "Store every term and its position in each document."
        ),
        (
            "03",
            "TF-IDF",
            "Calculate term importance and represent documents as vectors."
        ),
        (
            "04",
            "Query Processing",
            "Find matching terms, phrases, and Boolean results."
        ),
        (
            "05",
            "Ranking",
            "Compare query and document vectors using cosine similarity."
        )
    ]

    for number, title, description in steps:

        with st.container(border=True):

            a, b = st.columns([1, 5])

            with a:

                st.markdown(
                    f"<h2 style='color:#2563EB'>{number}</h2>",
                    unsafe_allow_html=True
                )

            with b:

                st.markdown(f"**{title}**")
                st.caption(description)


# ==================================================
# DOCUMENTS
# ==================================================

elif page == "Documents":

    st.markdown(
        '<div class="section-tag">DOCUMENT EXPLORER</div>',
        unsafe_allow_html=True
    )

    st.header("Explore healthcare documents")

    if documents:

        selected = st.selectbox(
            "Choose a document",
            docs
        )

        content = documents[selected]
        words = content.split()

        # Use a Streamlit container instead of an HTML div.
        with st.container(border=True):

            st.subheader(selected.upper())

            st.write(content)

            st.divider()

            a, b = st.columns(2)

            a.metric("Word count", len(words))

            b.metric(
                "Unique words",
                len(set(word.lower() for word in words))
            )

        st.subheader("Document collection")

        # Iterate through the sorted list, not documents.items().
        for doc in docs:

            with st.expander(doc.upper()):

                st.write(documents[doc])

    else:

        st.warning("No documents were found.")


# ==================================================
# SEARCH
# ==================================================

elif page == "Search":

    st.markdown(
        '<div class="section-tag">QUERY PROCESSING</div>',
        unsafe_allow_html=True
    )

    st.header("Search healthcare information")

    st.write(
        "A term search finds a single word. A phrase search finds "
        "words appearing consecutively and in the same order."
    )

    query = st.text_input(
        "Enter your query",
        placeholder="Try: patient care"
    )

    search_type = st.radio(
        "Search type",
        ["Single Term", "Phrase Search"],
        horizontal=True
    )

    if st.button("Search", type="primary"):

        words = query.lower().split()

        if not words:

            st.warning("Enter a search query.")

        elif search_type == "Single Term" and len(words) != 1:

            st.warning("Enter one word for term search.")

        else:

            if search_type == "Single Term":
                matches = search_term(query)

            else:
                matches = search_phrase(query)

            st.divider()

            st.subheader("Search results")

            if matches:

                st.success(
                    f"{len(matches)} matching document(s) found."
                )

                ranking = rank_documents(
                    query,
                    list(matches.keys())
                )

                for doc_id, score in ranking:

                    positions = matches[doc_id]

                    with st.container(border=True):

                        a, b = st.columns([3, 1])

                        with a:

                            st.markdown(f"**{doc_id.upper()}**")

                            st.caption(
                                f"Matching positions: {positions}"
                            )

                        with b:

                            st.markdown(
                                f"<div class='result-score'>{score:.4f}</div>",
                                unsafe_allow_html=True
                            )

                            st.caption("Similarity")

                        st.progress(score)

                        with st.expander("View document"):

                            st.write(
                                documents.get(doc_id, "")
                            )

            else:

                st.info("No matching documents found.")


# ==================================================
# BOOLEAN SEARCH
# ==================================================

elif page == "Boolean Search":

    st.markdown(
        '<div class="section-tag">BOOLEAN RETRIEVAL</div>',
        unsafe_allow_html=True
    )

    st.header("Combine search phrases")

    st.write(
        "Boolean operators let you combine two phrase searches "
        "to control which documents are returned."
    )

    phrase1 = st.text_input(
        "First phrase",
        value="patient care"
    )

    phrase2 = st.text_input(
        "Second phrase",
        value="medical research"
    )

    operator = st.selectbox(
        "Boolean operator",
        ["AND", "AND NOT"]
    )

    if st.button("Run Boolean Search", type="primary"):

        if not phrase1.strip() or not phrase2.strip():

            st.warning("Enter both phrases.")

        else:

            first_matches = set(
                search_phrase(phrase1).keys()
            )

            second_matches = set(
                search_phrase(phrase2).keys()
            )

            if operator == "AND":

                matching_docs = first_matches & second_matches

                query_for_ranking = phrase1 + " " + phrase2

            else:

                matching_docs = first_matches - second_matches

                query_for_ranking = phrase1

            st.divider()

            st.subheader("Boolean results")

            st.markdown(
                f"**{phrase1} {operator} {phrase2}**"
            )

            if matching_docs:

                st.success(
                    f"{len(matching_docs)} matching document(s) found."
                )

                ranking = rank_documents(
                    query_for_ranking,
                    matching_docs
                )

                for doc_id, score in ranking:

                    with st.container(border=True):

                        a, b = st.columns([3, 1])

                        with a:

                            st.markdown(f"**{doc_id.upper()}**")

                            st.caption(
                                documents.get(doc_id, "")
                            )

                        with b:

                            st.markdown(
                                f"<div class='result-score'>{score:.4f}</div>",
                                unsafe_allow_html=True
                            )

                            st.caption("Similarity")

                        st.progress(score)

            else:

                st.info(
                    "No documents satisfy this Boolean query."
                )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "HEALTHCARE INFORMATION RETRIEVAL SYSTEM  |  "
    "Python · PySpark · Streamlit"
)
