import streamlit as st
import numpy as np
import re
import pandas as pd
import pickle
import os
import time
import ssl
import requests
from io import StringIO, BytesIO

# Try to import NLTK with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    
    # Fix SSL certificate issues for NLTK download
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download stopwords with error handling
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    NLTK_AVAILABLE = True
except ImportError:
    st.error("NLTK is not installed. Please install it using: pip install nltk")
    NLTK_AVAILABLE = False

# Try to import sklearn with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("Scikit-learn is not installed. Please install it using: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Try to import PyPDF2 with fallback
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    st.error("PyPDF2 is not installed. Please install it using: pip install PyPDF2")
    PYPDF2_AVAILABLE = False

# Check if all required libraries are available
if not all([NLTK_AVAILABLE, SKLEARN_AVAILABLE, PYPDF2_AVAILABLE]):
    st.stop()

# Google Drive file IDs (from your links)
FAKE_PDF_ID = "1LB3I0x3oi_SEI6JAD58znaWgyRoh50Jm"
TRUE_PDF_ID = "1ah9uVYS_yMoj7pzMhQKgw3eD6oci6GrI"

def extract_text_from_pdf_url(pdf_url):
    """Extract text from a PDF file given its Google Drive URL"""
    try:
        # Download the PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Read PDF content
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def download_and_process_pdfs():
    """Download PDFs from Google Drive and extract text to create datasets"""
    try:
        # Create direct download URLs
        fake_pdf_url = f"https://drive.google.com/uc?export=download&id={FAKE_PDF_ID}"
        true_pdf_url = f"https://drive.google.com/uc?export=download&id={TRUE_PDF_ID}"
        
        st.sidebar.info("📥 Downloading and processing PDF files...")
        
        # Extract text from PDFs
        fake_text = extract_text_from_pdf_url(fake_pdf_url)
        true_text = extract_text_from_pdf_url(true_pdf_url)
        
        if fake_text is None or true_text is None:
            st.error("Failed to process PDF files")
            return None, None
        
        # Split text into articles (assuming each article is separated by multiple newlines)
        fake_articles = [article.strip() for article in fake_text.split('\n\n') if len(article.strip()) > 100]
        true_articles = [article.strip() for article in true_text.split('\n\n') if len(article.strip()) > 100]
        
        # Create DataFrames
        fake_df = pd.DataFrame({"text": fake_articles})
        fake_df["label"] = 0  # Fake news
        
        true_df = pd.DataFrame({"text": true_articles})
        true_df["label"] = 1  # True news
        
        # Save for future use
        fake_df.to_csv("fake_articles.csv", index=False)
        true_df.to_csv("true_articles.csv", index=False)
        
        return true_df, fake_df
        
    except Exception as e:
        st.error(f"Error downloading PDFs: {str(e)}")
        return None, None

# Check if model is already saved to avoid retraining
@st.cache_resource
def load_model():
    model_path = 'fake_news_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    # If model exists, load it
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        st.sidebar.success("✅ Loading pre-trained model...")
        model = pickle.load(open(model_path, 'rb'))
        vector = pickle.load(open(vectorizer_path, 'rb'))
        return model, vector
    
    # Otherwise, train and save the model
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Update progress
    def update_progress(step, total_steps=7):
        progress = (step / total_steps)
        progress_bar.progress(progress)
        return progress
    
    # Step 1: Loading data
    status_text.text("📂 Step 1/7: Loading datasets...")
    update_progress(1)
    time.sleep(0.5)
    
    try:
        # Try to load from previously created CSV files first
        if os.path.exists("true_articles.csv") and os.path.exists("fake_articles.csv"):
            true_df = pd.read_csv("true_articles.csv")
            fake_df = pd.read_csv("fake_articles.csv")
            st.sidebar.info("📁 Loaded data from processed CSV files")
        else:
            # Download and process PDFs
            true_df, fake_df = download_and_process_pdfs()
            if true_df is None or fake_df is None:
                st.error("❌ Failed to load datasets")
                st.stop()
        
        # Debug: Check columns
        print("True data columns:", list(true_df.columns))
        print("Fake data columns:", list(fake_df.columns))
        print(f"True articles: {len(true_df)}, Fake articles: {len(fake_df)}")
        
        # Combine them
        news_df = pd.concat([true_df, fake_df], ignore_index=True)
        news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Debug information
        print(f"Dataset loaded with {len(news_df)} rows")
        print("Columns available:", list(news_df.columns))
        
        status_text.text(f"✅ Loaded {len(news_df)} articles")
        update_progress(2)
        time.sleep(0.5)
        
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        st.stop()
    
    # Fill missing values
    news_df = news_df.fillna(' ')
    
    # Create content column
    if "text" in news_df.columns:
        news_df['content'] = news_df['text'].fillna("")
    else:
        st.error("❌ Error: No text column found in dataset")
        st.stop()
    
    # Define stemming function
    ps = PorterStemmer()
    
    # Get stopwords with error handling
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    
    # Apply stemming
    status_text.text("⚡ Step 4/7: Processing text...")
    update_progress(5)
    news_df['content'] = news_df['content'].apply(stemming)
    
    # Vectorize data
    status_text.text("🔢 Step 5/7: Vectorizing text...")
    update_progress(6)
    X = news_df['content'].values
    y = news_df['label'].values
    vector = TfidfVectorizer(max_features=10000)
    vector.fit(X)
    X = vector.transform(X)
    time.sleep(0.5)
    
    # Split and train model
    status_text.text("🤖 Step 6/7: Training model...")
    update_progress(7)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    time.sleep(0.5)
    
    # Save model and vectorizer
    status_text.text("💾 Step 7/7: Saving model...")
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vector, open(vectorizer_path, 'wb'))
    time.sleep(0.5)
    
    # Complete
    status_text.text("✅ Model trained successfully!")
    progress_bar.progress(1.0)
    time.sleep(1)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return model, vector

# Prediction function
def prediction(input_text):
    # Preprocess input text
    ps = PorterStemmer()
    
    # Get stopwords with error handling
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    
    input_text_processed = re.sub('[^a-zA-Z]',' ', input_text)
    input_text_processed = input_text_processed.lower()
    words = input_text_processed.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    input_text_processed = ' '.join(words)
    
    # Vectorize and predict
    input_data = vector.transform([input_text_processed])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    
    return prediction[0], probability

# Streamlit app UI
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title('🔍 Fake News Detector')
st.markdown("Check if a news article is likely to be **true** or **fake** using AI")

# Display loading message
loading_placeholder = st.empty()
loading_placeholder.info("⏳ Loading model and preparing analysis...")

# Load model
try:
    model, vector = load_model()
    loading_placeholder.empty()
    
except Exception as e:
    loading_placeholder.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main input area
st.divider()
input_text = st.text_area(
    '📝 Paste news article content here:',
    height=200,
    placeholder="Type or paste the complete news article content...",
    help="The more text you provide, the more accurate the prediction will be"
)

# Prediction button
if st.button('🔎 Analyze News', type='primary', use_container_width=True):
    if input_text.strip():
        with st.spinner('🔍 Analyzing content...'):
            pred, prob = prediction(input_text)
            
            # Show results
            st.divider()
            if pred == 1:
                st.success(f'✅ **Real News** (Confidence: {prob[1]*100:.1f}%)')
            else:
                st.error(f'🚨 **Fake News** (Confidence: {prob[0]*100:.1f}%)')
            
            # Probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real News Probability", f"{prob[1]*100:.1f}%")
            with col2:
                st.metric("Fake News Probability", f"{prob[0]*100:.1f}%")
    else:
        st.warning('⚠️ Please enter some text to analyze')

# Info section
st.sidebar.header("ℹ️ App Status")
st.sidebar.success("✅ Model loaded successfully!")
st.sidebar.info("📊 Ready to analyze news articles")

# Footer
st.divider()
st.caption("🔄 First run will download and process PDF files. Future runs will be faster!")
