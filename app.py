import streamlit as st
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read text from a Word document
def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to calculate cosine similarity between two texts with n-grams and optional length penalty
def calculate_similarity(text1, text2, ngram_range=(1, 3), apply_length_penalty=True):
    documents = [text1, text2]

    # Use TfidfVectorizer with the specified n-gram range
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    # Initial similarity percentage based on cosine similarity
    similarity_percentage = similarity_matrix[0][1] * 100

    # Optionally penalize based on the difference in document lengths
    if apply_length_penalty:
        len_diff_penalty = min(len(text1), len(text2)) / max(len(text1), len(text2))
        similarity_percentage *= len_diff_penalty

    return similarity_percentage

# Function to count words in a text
def word_count(text):
    return len(text.split())

# Streamlit interface
st.title("IPA Document Similarity Checker")
st.write("Upload two MS Word documents (.docx) to compare their similarity.")

# File uploader for two documents
file1 = st.file_uploader("Choose the first Word file", type="docx")
file2 = st.file_uploader("Choose the second Word file", type="docx")

# Allow users to select the n-gram range
st.sidebar.header("N-Gram Settings")
min_gram = st.sidebar.number_input("Minimum n-gram", min_value=1, max_value=5, value=1)
max_gram = st.sidebar.number_input("Maximum n-gram", min_value=1, max_value=5, value=3)

# Checkbox to apply or ignore length penalty
apply_length_penalty = st.sidebar.checkbox("Apply length penalty", value=True)

# Process the files and display the results
if file1 and file2:
    # Read the contents of both documents
    text1 = read_docx(file1)
    text2 = read_docx(file2)

    # Display word counts
    st.write(f"Number of words in file 1: {word_count(text1)}")
    st.write(f"Number of words in file 2: {word_count(text2)}")

    # Calculate similarity with the specified n-gram range and optional length penalty
    similarity_percentage = calculate_similarity(text1, text2, ngram_range=(min_gram, max_gram), apply_length_penalty=apply_length_penalty)

    # Display similarity percentage
    st.subheader(f"Similarity percentage: {similarity_percentage:.2f}%")
else:
    st.warning("Please upload both Word files to perform the comparison.")
