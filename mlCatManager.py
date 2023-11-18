import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from trainingModel import df
# Загрузка обученных моделей
category_model = joblib.load('Категории/category_model.joblib')
importance_model = joblib.load('Категории/importance_model.joblib')

# Преобразование категориальных признаков в числовые
category_mapping = {
    'change_order': 0,
    'delete_account': 1,
    'get_invoice': 2,
    'track_order': 3,
    'set_up_shipping_address': 4,
    'payment_issue': 5,
    'switch_account': 6,
    'delivery_period': 7,
    'create_account': 8,
    'complaint': 9,
    'contact_customer_service': 10,
    'get_refund': 11,
    'recover_password': 12,
    'delivery_options': 13,
    'contact_human_agent': 14,
    'check_invoice': 15,
    'check_refund_policy': 16,
    'check_invoices': 17,
    'newsletter_subscription': 18,
    'edit_account': 19,
    'check_cancellation_fee': 20,
    'check_payment_methods': 21,
    'track_refund': 22,
    'review': 23,
    'change_shipping_address': 24,
    'registration_problems': 25,
    'place_order': 26,
    'cancel_order': 27
}

importance_mapping = {
    'high_priority': 0,
    'standard_priority': 1,
    'medium_priority': 2
}

# Пример нового запроса
new_query = "Can you help me track my order?"

# Создайте TfidfVectorizer с теми же параметрами, что и при обучении модели
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')


# Преобразование текстового запроса в числовой формат с использованием TF-IDF
new_query_tfidf = vectorizer.transform(df[new_query])

# Обратные отображения для категорий и важности
category_mapping_reverse = {v: k for k, v in category_mapping.items()}
importance_mapping_reverse = {v: k for k, v in importance_mapping.items()}

# Предсказание категории и важности
predicted_category = category_model.predict(new_query_tfidf)[0]
predicted_importance = importance_model.predict(new_query_tfidf)[0]

# Обратное отображение числовых предсказаний в текстовый формат
predicted_category_text = category_mapping_reverse[predicted_category]
predicted_importance_text = importance_mapping_reverse[predicted_importance]

# Вывод результатов предсказания
print("Predicted Category:", predicted_category_text)
print("Predicted Importance:", predicted_importance_text)
