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

# Streamlit interface
st.title("فاحص تشابه المستندات")
st.write("قم بتحميل ملفين MS Word (.docx) للمقارنة بينهما.")

# Custom CSS to right-align the tabs
st.markdown("""
    <style>
    .css-1uixwzi.e1fqkh3o3 { justify-content: flex-end; } /* Right-align the tabs */
    </style>
""", unsafe_allow_html=True)

# File uploader for two documents
file1 = st.file_uploader("اختر الملف الأول", type="docx")
file2 = st.file_uploader("اختر الملف الثاني", type="docx")

# Allow users to select the n-gram range
st.sidebar.header("إعدادات النماذج اللغوية")
min_gram = st.sidebar.number_input("النموذج اللغوي الأدنى (Minimum n-gram)", min_value=1, max_value=5, value=1)
max_gram = st.sidebar.number_input("النموذج اللغوي الأقصى (Maximum n-gram)", min_value=1, max_value=5, value=3)

# Checkbox to apply or ignore length penalty
apply_length_penalty = st.sidebar.checkbox("تطبيق عقوبة طول المستند", value=True)

# Process the files and display the results
if file1 and file2:
    # Read the contents of both documents
    text1 = read_docx(file1)
    text2 = read_docx(file2)

    # Display word counts
    st.write(f"عدد الكلمات في الملف الأول: {word_count(text1)}")
    st.write(f"عدد الكلمات في الملف الثاني: {word_count(text2)}")

    # Calculate similarity with the specified n-gram range and optional length penalty
    similarity_percentage = calculate_similarity(text1, text2, ngram_range=(min_gram, max_gram), apply_length_penalty=apply_length_penalty)

    # Display similarity percentage
    st.subheader(f"نسبة التشابه: {similarity_percentage:.2f}%")

    # Split both documents into sentences
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    # Compare the sentences
    new_sentences, deleted_sentences, slightly_changed_sentences, common_sentences = compare_sentences(sentences1, sentences2)

    # Create tabs for each section (in Arabic)
    tab1, tab2, tab3, tab4 = st.tabs(["الجمل الجديدة", "الجمل المحذوفة", "الجمل المتغيرة", "الجمل المشتركة"])

    # Tab 1: New sentences
    with tab1:
        st.subheader("الجمل الجديدة (في الملف الثاني ولكن ليست في الأول):")
        st.write("\n".join(new_sentences) if new_sentences else "لا توجد جمل جديدة.")

    # Tab 2: Deleted sentences
    with tab2:
        st.subheader("الجمل المحذوفة (في الملف الأول ولكن ليست في الثاني):")
        st.write("\n".join(deleted_sentences) if deleted_sentences else "لا توجد جمل محذوفة.")

    # Tab 3: Slightly changed sentences
    with tab3:
        st.subheader("الجمل المتغيرة:")
        if slightly_changed_sentences:
            for original, changed in slightly_changed_sentences:
                st.write(f"الأصلية: {original}")
                st.write(f"المتغيرة: {changed}")
                st.write("---")
        else:
            st.write("لا توجد جمل متغيرة.")

    # Tab 4: Common sentences
    with tab4:
        st.subheader("الجمل المشتركة (في كلا الملفين):")
        st.write("\n".join(common_sentences) if common_sentences else "لا توجد جمل مشتركة.")

else:
    st.warning("يرجى تحميل كلا الملفين لإجراء المقارنة.")
