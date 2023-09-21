from model import extract_features, train_classifier, predict_sentiment
# Здесь вы можете использовать функции и классы из model.py
import nltk, spacy, spacy.cli
from googletrans import Translator
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Создайте объект Translator
nltk.download('punkt')
spacy.cli.download("en_core_web_sm")
translator = Translator()
# model_path = "/Desktop/интелект/main.py`/en_core_web_sm"
# nlp = spacy.load(model_path)

# Загрузите данные для обучения классификатора на тональность текста
# Замените это на вашими данными
texts = [
    "Этот фильм просто великолепен! Я наслаждался каждой минутой просмотра.",
    "Ужасный опыт. Фильм был скучным и непонятным.",
    "Книга очень увлекательна. Я не мог оторваться от нее.",
    "Сервис в этом ресторане ужасен. Официанты были невежливыми и невнимательными.",
    "Отличная компания и веселая атмосфера на вечеринке.",
    "Этот продукт оказался довольно разочарованием. Низкое качество и высокая цена.",
]

labels = ["позитивный", "негативный", "позитивный", "негативный", "позитивный", "негативный"]


# Преобразуйте тексты в числовые признаки с использованием TF-IDF и более сложных функций извлечения признаков с spaCy
nlp = spacy.load("en_core_web_sm")
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=lambda text: [token.lemma_ for token in nlp(text) if not token.is_punct and not token.is_stop],
)
X = tfidf_vectorizer.fit_transform(texts)

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Обучите классификатор
classifier = LinearSVC()
classifier.fit(X_train, y_train)

while True:
    # Запросите пользователя ввести текст
    text_to_translate = input("Введите текст для перевода (или 'выход' для завершения): ")

    # Проверьте, если пользователь ввел "выход", то завершите программу
    if text_to_translate.lower() == "выход":
        break

    # Определите исходный язык автоматически
    detected_language = translator.detect(text_to_translate).lang

    # Определите целевой язык (если исходный - русский, то целевой - английский, и наоборот)
    target_language = "en" if detected_language == "ru" else "ky"
    
    # Перевод с исходного языка на целевой
    translation = translator.translate(text_to_translate, src=detected_language, dest=target_language)

    # Оцените тональность переведенного текста
    translated_text = translation.text
    translated_text_features = tfidf_vectorizer.transform([translated_text])
    sentiment = classifier.predict(translated_text_features)

    # Выведите перевод и тональность
    print(f"Перевод на {target_language}: {translated_text}")
    print(f"Тональность: {sentiment[0]}")

extract_features()
train_classifier()
predict_sentiment()
