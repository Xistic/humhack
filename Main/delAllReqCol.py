import pandas as pd

# Чтение CSV-файла
df = pd.read_csv('C:/Users/egork/dev/humhack/Main/train_data.csv')

# Оставить только колонки 'request' и 'importance'
df = df[['request', 'importance']]

# Сохранение в новый CSV-файл
df.to_csv('dataFromImportance.csv', index=False)
