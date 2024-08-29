# -*- coding: utf-8 -*-
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize SQLite database
conn = sqlite3.connect('summaries.db')
cursor = conn.cursor()

# Create the table with 'name' column if it does not exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    name TEXT,
    summary TEXT
)
''')
conn.commit()
conn.close()

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

def summarize_text(text, min_length=70, max_length=120):
    preprocessed_text = text.strip().replace('\n', ' ')
    t5_input_text = 'summarize: ' + preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(tokenized_text, min_length=min_length, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return summarize_text(full_text)

def summarize_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return summarize_text(text)

def save_summary_to_db(name, summary):
    conn = sqlite3.connect('summaries.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO summaries (name, summary) VALUES (?, ?)', (name, summary))
    conn.commit()
    conn.close()

def get_all_summaries_from_db():
    conn = sqlite3.connect('summaries.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, summary FROM summaries')
    summaries = cursor.fetchall()
    conn.close()
    return summaries

def is_duplicate_summary(new_summary):
    summaries = [row[1] for row in get_all_summaries_from_db()] + [new_summary]
    
    if len(summaries) == 0 or all(not summary.strip() for summary in summaries):
        return False
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(summaries)
    
    if tfidf_matrix.shape[0] > 1:
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        max_similarity = max(similarities[0])
        return max_similarity > 0.7
    
    return False

def handle_summarization(input_type, input_value, summary_name):
    if input_type == "text":
        summary = summarize_text(input_value)
    elif input_type == "pdf":
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(input_value.getbuffer())
        summary = summarize_pdf("uploaded_pdf.pdf")
    elif input_type == "url":
        summary = summarize_url(input_value)
    
    if is_duplicate_summary(summary):
        st.warning("This summary is similar to an existing one. Please consider revising the content.")
    else:
        save_summary_to_db(summary_name, summary)
        st.success("Summary saved successfully!")
        st.markdown("**Summary:**")
        st.write(summary)

# Streamlit interface
st.title("Text Summarization App")

choice = st.sidebar.selectbox(
    "Select Summarization Technique",
    ["Summarize Text", "Summarize PDF", "Summarize URL"]
)

summary_name = st.text_input("Enter a name or title for this summary")

if choice == "Summarize Text":
    st.subheader("Summarize Direct Text")
    input_text = st.text_area("Enter your text here")
    if st.button("Summarize and Save Text"):
        if not input_text.strip():
            st.error("Please enter some text to summarize.")
        else:
            handle_summarization("text", input_text, summary_name)

elif choice == "Summarize PDF":
    st.subheader("Summarize PDF Document")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None and st.button("Summarize PDF"):
        handle_summarization("pdf", uploaded_file, summary_name)

elif choice == "Summarize URL":
    st.subheader("Summarize Webpage Text")
    url = st.text_input("Enter the URL of the webpage")
    if st.button("Summarize URL"):
        if not url.strip():
            st.error("Please enter a valid URL.")
        else:
            handle_summarization("url", url, summary_name)

# Display summary history
if st.sidebar.button("Show Summary History"):
    summaries = get_all_summaries_from_db()
    for name, summary in summaries:
        st.markdown(f"**{name}:**")
        st.write(summary)
        st.markdown("---")
