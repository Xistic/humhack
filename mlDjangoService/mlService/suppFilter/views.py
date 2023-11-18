from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Загрузка моделей и маппингов
category_model = joblib.load('C:/Users/egork/dev/humhack/mlDjangoService/mlService/suppFilter/mlModels/category_model.joblib')
importance_model = joblib.load('C:/Users/egork/dev/humhack/mlDjangoService/mlService/suppFilter/mlModels/importance_model.joblib')
vectorizer = joblib.load("C:/Users/egork/dev/humhack/mlDjangoService/mlService/suppFilter/mlModels/tfidf_vectorizer.joblib")
category_mapping_reverse = {
    0:'change_order',
    1:'delete_account',
    2:'get_invoice',
    3:'track_order',
    4:'set_up_shipping_address',
    5:'payment_issue',
    6:'switch_account',
    7:'delivery_period',
    8:'create_account',
    9:'complaint',
    10:'contact_customer_service',
    11:'get_refund',
    12:'recover_password',
    13:'delivery_options',
    14:'contact_human_agent',
    15:'check_invoice',
    16:'check_refund_policy',
    17:'check_invoices',
    18:'newsletter_subscription',
    19:'edit_account',
    20:'check_cancellation_fee',
    21:'check_payment_methods',
    22:'track_refund',
    23:'review',
    24:'change_shipping_address',
    25:'registration_problems',
    26:'place_order',
    27:'cancel_order'
}

importance_mapping_reverse = {
    0: 'high_priority',
    1: 'standard_priority',
    2: 'medium_priority'
}
import json

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Получение данных в формате JSON из тела запроса
            json_data = json.loads(request.body.decode('utf-8'))
            user_query = json_data.get('user_query', '')

            # Преобразование текстового запроса в числовой формат с использованием TF-IDF
            user_query_tfidf = vectorizer.transform([user_query])

            # Предсказание категории и важности
            predicted_category = category_model.predict(user_query_tfidf)[0]
            predicted_importance = importance_model.predict(user_query_tfidf)[0]

            # Обратное отображение числовых предсказаний в текстовый формат
            predicted_category_text = category_mapping_reverse[predicted_category]
            predicted_importance_text = importance_mapping_reverse[predicted_importance]

            # Возвращение предсказанных значений в формате JSON
            response_data = {
                'predicted_category': predicted_category_text,
                'predicted_importance': predicted_importance_text
            }

            return JsonResponse(response_data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
