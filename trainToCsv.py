import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных из CSV файла
df = pd.read_csv('train.csv', sep="|")

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

df['request'] = df['request'].map(category_mapping)
df['importance'] = df['importance'].map(importance_mapping)

# Разделение данных на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Сохранение подготовленных данных в новые CSV файлы
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
