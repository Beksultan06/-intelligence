import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Загрузка модели spaCy
nlp = spacy.load("en_core_web_sm")

def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda text: [token.lemma_ for token in nlp(text) if not token.is_punct and not token.is_stop],
    )
    X = tfidf_vectorizer.fit_transform(texts)
    return X, tfidf_vectorizer

def train_classifier(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    return classifier

def predict_sentiment(classifier, tfidf_vectorizer, text):
    transformed_text = tfidf_vectorizer.transform([text])
    sentiment = classifier.predict(transformed_text)
    return sentiment[0]
