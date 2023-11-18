import pandas as pd

# Загрузите CSV-датасет
df = pd.read_csv('train_data.csv', delimiter='|')
# Получите уникальные компетенции
unique_competencies = df['importance'].unique()

# Откройте файл для записи
with open('важность_уникальные.txt', 'w') as file:
    # Запишите каждую компетенцию в новую строку
    for competency in unique_competencies:
        file.write(competency + '\n')
