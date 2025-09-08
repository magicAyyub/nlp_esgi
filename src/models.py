from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def make_model():
    # Use SVM which often works better for text classification
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Enable probability estimates
        random_state=42
    )

def make_model_with_gridsearch():
    """Alternative model with hyperparameter tuning"""
    # Create a pipeline with grid search
    model = SVC(probability=True, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    return GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1
    )
