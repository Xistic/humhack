import pandas as pd

# Чтение CSV-файла
df = pd.read_csv('C:/Users/egork/dev/humhack/train.csv')

# Маппинги
importance_mapping = {
    'high_priority': 0,
    'standard_priority': 1,
    'medium_priority': 2
}

# Убираем 'utterance' и преобразуем 'importance' в числовые значения
df = df[['request', 'importance']]
df['importance'] = df['importance'].map(importance_mapping)

# Сохраняем в новый CSV-файл
df.to_csv('datasetToModel.csv', index=False)
