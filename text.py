import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Загрузка обученной модели
model = load_model('trained_model.h5')  # Замените 'your_trained_model.h5' на путь к вашей обученной модели

# Загрузка тестового обращения (пример)
test_utterance = "I want to change my order"

# Предобработка тестового обращения
test_utterance = test_utterance.lower()  # Приведем текст к нижнему регистру
test_utterance = label_encoder.transform([test_utterance])  # Преобразуем текст в числовой формат

# Предсказание типа запроса и приоритета
prediction = model.predict(test_utterance)

# Обратное преобразование числовых предсказаний в исходные категории
predicted_request = label_encoder.inverse_transform([prediction[0][0]])
predicted_importance = label_encoder.inverse_transform([prediction[0][1]])

# Вывод результатов
print("Predicted Request Type:", predicted_request)
