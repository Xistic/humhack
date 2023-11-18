import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Чтение CSV-файла
df = pd.read_csv('C:/Users/egork/dev/humhack/dataFromImportance.csv')

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
# Преобразование категорий в числовые значения
df['request'] = df['request'].map(category_mapping)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='most_frequent')  # Используйте 'most_frequent' или 'constant'
df['request'] = imputer.fit_transform(df[['request']])

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df[['request']], df['importance'], test_size=0.2, random_state=42)

# Инициализация и обучение модели
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Сохранение модели в формате joblib
joblib.dump(model, 'importance_model.joblib')
