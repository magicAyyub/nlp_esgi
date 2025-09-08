from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text with NLTK: lowercase, remove punctuation, stem, remove stopwords"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stopwords (French stopwords for French video titles)
    stop_words = set(stopwords.words('french'))
    words = [word for word in words if word not in stop_words]
    
    # Stem words
    stemmer = SnowballStemmer('french')
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def make_features(df, vectorizer=None, fit_vectorizer=True):
    y = df["is_comic"]

    # Preprocess video names
    processed_texts = [preprocess_text(text) for text in df["video_name"]]

    # Use TfidfVectorizer instead of CountVectorizer for better performance
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words=None,  # We already removed stopwords
            max_features=1000,  # Limit features
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
    
    if fit_vectorizer:
        X = vectorizer.fit_transform(processed_texts).toarray()
        # Save the vectorizer for later use
        joblib.dump(vectorizer, "models/vectorizer.pkl")
    else:
        X = vectorizer.transform(processed_texts).toarray()

    return X, y, vectorizer
