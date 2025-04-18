### Importing dependencies
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding='utf-8'),
        logging.StreamHandler()
    ]  
)

logging.info("Starting preprocessing...")

### Text cleaning
stop_words = set(stopwords.words('english'))

### Load dataset
try:
    df = pd.read_csv('../data/movies.csv')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise e

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower()  
    tokens = word_tokenize(text) 
    tokens = [word for word in tokens if word not in stop_words]  

    return " ".join(tokens)

### Filter the required column for recommandation
required_columns = ['genres', 'keywords', 'title', 'overview']
df = df[required_columns]
df = df.dropna().reset_index(drop=True)

df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']

logging.info("Cleaning Text...")
df['cleaned_text'] = df['combined'].apply(preprocess_text)
logging.info("Text cleaned successfully.")

### Vectorization
logging.info("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

### Cosine Similarity
logging.info("Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("Cosine similarity calculated successfully.")

### Save everything
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("Data Saved to disk.")

logging.info("Preprocessing completed successfully.")