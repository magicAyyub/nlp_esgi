import click
import joblib

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y, vectorizer = make_features(df, fit_vectorizer=True)

    model = make_model()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/processed/test.csv", help="File with data to predict")
@click.option("--model_dump_filename", default="models/dump.json", help="File to load model from")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    import pandas as pd
    
    # Load the trained model and vectorizer
    model = joblib.load(model_dump_filename)
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    # Load test data
    df = make_dataset(input_filename)
    
    # Make features for prediction (don't fit the vectorizer)
    X, y_true, _ = make_features(df, vectorizer=vectorizer, fit_vectorizer=False)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create output dataframe with predictions
    output_df = df.copy()
    output_df['predicted_is_comic'] = y_pred
    
    # Save predictions
    output_df.to_csv(output_filename, index=False)
    
    # Calculate accuracy if we have true labels
    if 'is_comic' in df.columns:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
    
    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, vectorizer = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    
    # Run k-fold cross validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
