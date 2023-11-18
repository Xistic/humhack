from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Загрузка подготовленных данных
df = pd.read_csv('train_data.csv', sep=',')

# Разделение данных на признаки (X) и целевую переменную (y)
X = df['utterance']
y_category = df['request']
y_importance = df['importance']

# Преобразование текстовых данных в числовой формат с использованием TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_category_train, y_category_test, y_importance_train, y_importance_test = train_test_split(
    X_tfidf, y_category, y_importance, test_size=0.2, random_state=42
)

# Создание модели SVM для классификации категорий
category_model = SVC(kernel='linear')
category_model.fit(X_train, y_category_train)

# Создание модели SVM для классификации важности
importance_model = SVC(kernel='linear')
importance_model.fit(X_train, y_importance_train)

# Оценка точности моделей на тестовой выборке
y_category_pred = category_model.predict(X_test)
y_importance_pred = importance_model.predict(X_test)

print("Accuracy for category classification:", accuracy_score(y_category_test, y_category_pred))
print("Accuracy for importance classification:", accuracy_score(y_importance_test, y_importance_pred))

# Сохранение моделей для дальнейшего использования
import joblib
joblib.dump(category_model, 'Категории/category_model.joblib')
joblib.dump(importance_model, 'Категории/importance_model.joblib')
