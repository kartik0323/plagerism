import streamlit as st
import pandas as pd
import nltk
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from transformers import BertTokenizer, BertModel
import torch
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px

# Download the NLTK 'punkt' package for sentence tokenization
nltk.download('punkt')

# Initialize BERT tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
except Exception as e:
    st.error(f"Error loading BERT model: {e}")
    st.stop()

def get_bert_embeddings(text):
    """Compute BERT embeddings for the input text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def get_sentences(text):
    """Tokenize the input text into sentences."""
    return tokenize.sent_tokenize(text)

def get_url(sentence):
    """Retrieve the first search result URL for a given sentence from Google."""
    base_url = 'https://www.google.com/search?q='
    query = '+'.join(sentence.split())
    url = base_url + query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        divs = soup.find_all('div', class_='yuRUbf')
        urls = [div.find('a')['href'] for div in divs if div.find('a')]
        return urls[0] if urls else None
    except Exception as e:
        st.warning(f"Error fetching URL for '{sentence[:30]}...': {e}")
        return None

def read_text_file(file):
    """Read content from a text file."""
    with io.open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx_file(file):
    """Extract text from a .docx file."""
    return docx2txt.process(file)

def read_pdf_file(file):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_from_file(uploaded_file):
    """Get text from the uploaded file based on its type."""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            return read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx_file(uploaded_file)
    return ""

def get_text(url):
    """Retrieve and extract text from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(map(lambda p: p.text, soup.find_all('p')))
    except Exception as e:
        st.warning(f"Error fetching text from URL: {e}")
        return ""

def get_similarity(text1, text2):
    """Calculate cosine similarity between two text documents using BERT embeddings."""
    embeddings1 = get_bert_embeddings(text1)
    embeddings2 = get_bert_embeddings(text2)
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
    return similarity

def get_similarity_list(texts, filenames=None):
    """Get pairwise similarity scores between texts."""
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list

def get_similarity_list2(text, url_list):
    """Calculate similarity between the input text and the text retrieved from URLs."""
    similarity_list = []
    total_similarity = 0
    valid_url_count = 0

    for url in url_list:
        if url is not None:
            text2 = get_text(url)
            similarity = get_similarity(text, text2)
            similarity_list.append(similarity)
            total_similarity += similarity
            valid_url_count += 1
        else:
            similarity_list.append(0)

    avg_similarity = total_similarity / valid_url_count if valid_url_count > 0 else 0
    return similarity_list, avg_similarity

def plot_results(df):
    """Generate plots for the similarity results."""
    plot_types = {
        "scatter": px.scatter(df, x='File 1', y='File 2', color='Similarity', title='Similarity Scatter Plot'),
        "bar": px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart'),
        "histogram": px.histogram(df, x='Similarity', title='Similarity Histogram')
    }
    
    for plot in plot_types.values():
        st.plotly_chart(plot, use_container_width=True)

st.set_page_config(page_title='Plagiarism Detection with BERT')
st.title('Plagiarism Detector using BERT')

option = st.radio("Select input option:", ('Enter text', 'Upload file', 'Find similarities between files'))

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    text = get_text_from_file(uploaded_file) if uploaded_file else ""
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload multiple files (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts, filenames = [], []
    for uploaded_file in uploaded_files:
        if uploaded_file:
            text = get_text_from_file(uploaded_file)
            texts.append(text)
            filenames.append(uploaded_file.name)
    text = " ".join(texts)

if st.button('Check for plagiarism or find similarities'):
    st.write("### Checking for plagiarism or finding similarities...")
    if not text:
        st.write("### No text found for plagiarism check or finding similarities.")
        st.stop()
    
    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity']).sort_values(by=['Similarity'], ascending=False)
        plot_results(df)
    else:
        sentences = get_sentences(text)
        urls = [get_url(sentence) for sentence in sentences]

        if not any(urls):
            st.write("### No URLs found for plagiarism detection.")
            st.stop()

        similarity_list, avg_similarity = get_similarity_list2(text, urls)
        df = pd.DataFrame({'Sentence': sentences, 'URL': urls})
        df = df.reset_index(drop=True)

        percentage_similarity = (avg_similarity * 100)
        st.write(f"### Average Percentage Similarity with Internet Content: {percentage_similarity:.2f}%")

        if 'URL' in df.columns:
            df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>' if x else '')
        
        df_html = df.to_html(escape=False)
        if 'URL' in df.columns:
            df_html = df_html.replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
        st.write(df_html, unsafe_allow_html=True)
