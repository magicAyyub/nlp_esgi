from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def make_model():
    # Simple Random Forest as requested
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

def make_linear_model():
    # Alternative linear model 
    return LogisticRegression(
        random_state=42,
        max_iter=1000
    )
