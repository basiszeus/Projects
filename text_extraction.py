import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
from textblob import TextBlob
from transformers import pipeline
import re
import torch
from collections import Counter
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
device = 0 if torch.cuda.is_available() else -1 # Check if GPU is available

# Extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

# Function to count specific keywords
def count_keyword_occurrences(text, keyword):
    return text.lower().count(keyword.lower())

# Function to get top N relevant keywords using TF-IDF
def get_relevant_keywords(text, top_n=10):
    # Get stop words from NLTK
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    # Use TF-IDF to find relevant keywords
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
    
    # Get the feature names and their corresponding TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    
    # Combine feature names with their scores and sort them
    keyword_scores = list(zip(feature_names, tfidf_scores))
    relevant_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    
    return relevant_keywords[:top_n]

# Function to summarize text with chunking
def summarize_text(summarizer, text, max_length=60, min_length=30):
    max_chunk_size = 1000  # Adjust based on the model's max token limit
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []
    for chunk in text_chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            if summary:
                summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            return ""
    return " ".join(summaries)

# Function to extract top 3 companies using spaCy NER
def extract_top_companies(ner_pipeline, text, top_n=3):
    try:
        ner_results = ner_pipeline(text)
        org_entities = [ent['word'].replace('##', '').strip() for ent in ner_results if ent['entity_group'] == 'ORG' and len(ent['word'].replace('##', '').strip()) > 1]
        org_counts = Counter(org_entities)
        top_companies = org_counts.most_common(top_n)
        return top_companies
    except Exception as e:
        st.error(f"An error occurred during NER: {e}")
        return []

# Initialize summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", device=device)

# Initialize NER pipeline
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True, device=device)

# Streamlit app layout
st.set_page_config(page_title="📄 Text Analysis App", layout="wide")
st.title("📄 Text Analysis App")
st.write("Upload a PDF file or enter a URL to analyze text.")

# Sidebar for input options
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("📁 Choose a PDF file", type=["pdf"])
url_input = st.sidebar.text_input("🔗 Or enter a URL")

# Process the input
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
elif url_input:
    text = extract_text_from_url(url_input)
else:
    st.warning("Please upload a PDF file or enter a URL to proceed.")
    st.stop()

# Check if text extraction was successful
if not text.strip():
    st.error("No text could be extracted from the provided source.")
    st.stop()

# Display the extracted text (first 1000 characters)
st.subheader("📄 Extracted Text")
st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

# Keyword counting
st.subheader("🔍 Keyword Analysis")
keyword = st.text_input("Enter a keyword to count its occurrences:")
if keyword:
    keyword_count = count_keyword_occurrences(text, keyword)
    st.write(f"The keyword **'{keyword}'** is mentioned **{keyword_count}** times.")

# Get and display top relevant keywords with graph
st.subheader("📈 Relevant Keywords")
top_keywords = get_relevant_keywords(text, top_n=10)

# Prepare data for plotting
words, scores = zip(*top_keywords)

# Plotting the bar chart
plt.figure(figsize=(7, 4))
plt.bar(words, scores, color='skyblue')
plt.xlabel('Keywords')
plt.ylabel('TF-IDF Score')
plt.title('Top Relevant Keywords')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Load summarizer and NER pipeline
summarizer = load_summarizer()
ner_pipeline = load_ner_pipeline()

# Summarization
st.subheader("📝 Summarization")
if st.button("Generate Summary"):
    with st.spinner("Summarizing the text..."):
        summary = summarize_text(summarizer, text)
    if summary:
        st.write(summary)
    else:
        st.write("Could not generate a summary.")

# Sentiment analysis
st.subheader("😊 Sentiment Analysis")
if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing sentiment..."):
        sentiment = TextBlob(text).sentiment
    st.write(f"**Polarity**: {sentiment.polarity}")
    st.write(f"**Subjectivity**: {sentiment.subjectivity}")
    if sentiment.polarity > 0:
        st.success("The sentiment of the text is **positive**.")
    elif sentiment.polarity < 0:
        st.error("The sentiment of the text is **negative**.")
    else:
        st.info("The sentiment of the text is **neutral**.")

# Named Entity Recognition (NER) for top 3 companies
st.subheader("🏢 Top 3 Company Entities (NER)")
if st.button("Extract Top 3 Companies"):
    with st.spinner("Extracting top 3 companies..."):
        top_companies = extract_top_companies(ner_pipeline, text, top_n=3)
    if top_companies:
        st.write("**Top 3 Companies Found:**")
        for company, freq in top_companies:
            st.write(f"- **{company}**: mentioned **{freq}** times")
    else:
        st.write("No company entities found in the text.")
