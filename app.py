import streamlit as st
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Function to read text from a Word document
def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to split text into sentences by new lines
def split_into_sentences(text):
    return [sentence.strip() for sentence in text.split('\n') if sentence.strip()]

# Function to calculate similarity between two sentences
def sentence_similarity(sent1, sent2, threshold=0.85):
    return SequenceMatcher(None, sent1, sent2).ratio() >= threshold

# Function to classify sentences into new, deleted, slightly changed, or common
def compare_sentences(sentences1, sentences2, threshold=0.85):
    new_sentences = []
    deleted_sentences = []
    slightly_changed_sentences = []
    common_sentences = []

    # First pass: Check for common and slightly changed sentences
    for sent1 in sentences1:
        matched = False
        for sent2 in sentences2:
            if sent1 == sent2:
                common_sentences.append(sent1)
                matched = True
                break
            elif sentence_similarity(sent1, sent2, threshold):
                slightly_changed_sentences.append((sent1, sent2))
                matched = True
                break
        if not matched:
            deleted_sentences.append(sent1)

    # Second pass: Find new sentences in sentences2 that are not in sentences1
    for sent2 in sentences2:
        if sent2 not in common_sentences and not any(sentence_similarity(sent1, sent2, threshold) for sent1 in sentences1):
            new_sentences.append(sent2)

    return sorted(new_sentences), sorted(deleted_sentences), sorted(slightly_changed_sentences), sorted(common_sentences)

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

# Function to highlight sentences with different colors for comparison
def highlight_sentence(text, color):
    return f'<span style="background-color:{color}">{text}</span>'

# Streamlit interface
st.title("IPA AI Use Cases")
st.title("Tool One: Document Similarity Checker")
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
    st.write(f"Number of words in **{file1.name}**: {word_count(text1)}")
    st.write(f"Number of words in **{file2.name}**: {word_count(text2)}")

    # Calculate similarity with the specified n-gram range and optional length penalty
    similarity_percentage = calculate_similarity(text1, text2, ngram_range=(min_gram, max_gram), apply_length_penalty=apply_length_penalty)

    # Display similarity percentage
    st.subheader(f"Similarity percentage: {similarity_percentage:.2f}%")

    # Split both documents into sentences
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    # Compare the sentences
    new_sentences, deleted_sentences, slightly_changed_sentences, common_sentences = compare_sentences(sentences1, sentences2)

    # Calculate counts for percentages
    total_new = len(new_sentences)
    total_deleted = len(deleted_sentences)
    total_changed = len(slightly_changed_sentences)
    total_common = len(common_sentences)

    # Total sentences considered for percentage calculation
    total_sentences = total_new + total_deleted + total_changed + total_common

    # Avoid division by zero
    if total_sentences == 0:
        total_sentences = 1

    # Calculate percentages for each category
    new_percentage = (total_new / total_sentences) * 100
    deleted_percentage = (total_deleted / total_sentences) * 100
    updated_percentage = (total_changed / total_sentences) * 100
    common_percentage = (total_common / total_sentences) * 100

    # Display the percentages
    # st.subheader("Breakdown of Sentence Changes:")
    # st.write(f"New Sentences: {new_percentage:.2f}%")
    # st.write(f"Deleted Sentences: {deleted_percentage:.2f}%")
    # st.write(f"Updated Sentences: {updated_percentage:.2f}%")
    # st.write(f"Common Sentences: {common_percentage:.2f}%")

    # Color legends
    st.write("### Color Legends:")
    st.write(f'<span style="color:red">Deleted Sentences</span> | '
             f'<span style="color:green">New Sentences</span> | '
             f'<span style="color:gold">Updated Sentences</span>', unsafe_allow_html=True)

    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)

    # Prepare highlighted texts for both files
    highlighted_text1 = []
    highlighted_text2 = []

    # Highlight sentences in file1 and file2
    for sent1 in sentences1:
        if sent1 in deleted_sentences:
            highlighted_text1.append(highlight_sentence(sent1, "red"))
        elif any(sent1 == orig for orig, _ in slightly_changed_sentences):
            highlighted_text1.append(highlight_sentence(sent1, "gold"))
        else:
            highlighted_text1.append(sent1)

    for sent2 in sentences2:
        if sent2 in new_sentences:
            highlighted_text2.append(highlight_sentence(sent2, "green"))
        elif any(sent2 == changed for _, changed in slightly_changed_sentences):
            highlighted_text2.append(highlight_sentence(sent2, "gold"))
        else:
            highlighted_text2.append(sent2)

    # Display file1 on the right and file2 on the left with dynamic filenames
    with col1:
        st.subheader(f"{file2.name} (Left)")
        st.write('<div style="text-align: left; direction: rtl;">' + '<br>'.join(highlighted_text2) + '</div>', unsafe_allow_html=True)

    with col2:
        st.subheader(f"{file1.name} (Right)")
        st.write('<div style="text-align: left; direction: rtl;">' + '<br>'.join(highlighted_text1) + '</div>', unsafe_allow_html=True)

else:
    st.warning("Please upload both Word files to perform the comparison.")
