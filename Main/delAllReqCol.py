import pandas as pd

# Чтение CSV-файла
df = pd.read_csv('train_data.csv')

# Фильтрация по колонке 'utterance'
df = df[df['utterance'].isna()]

# Удаление колонки 'utterance'
df = df.drop(columns=['utterance'])

# Сохранение в новый CSV-файл
df.to_csv('reqAndImpot.csv', index=False)
