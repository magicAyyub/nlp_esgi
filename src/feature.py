from sklearn.feature_extraction.text import CountVectorizer
import joblib

def make_features(df, vectorizer=None, fit_vectorizer=True):
    y = df["is_comic"]

    # Simple CountVectorizer as requested
    if vectorizer is None:
        vectorizer = CountVectorizer(
            lowercase=True,
            max_features=1000  # Limit features to avoid overfitting
        )
    
    if fit_vectorizer:
        X = vectorizer.fit_transform(df["video_name"]).toarray()
        # Save the vectorizer for later use
        joblib.dump(vectorizer, "models/vectorizer.pkl")
    else:
        X = vectorizer.transform(df["video_name"]).toarray()

    return X, y, vectorizer
