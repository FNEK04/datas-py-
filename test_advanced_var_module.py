import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from advanced_var_module import AdvancedVARModel

print("=== Генерация синтетических данных ===")
np.random.seed(42)
n_obs = 200
dates = pd.date_range(start='2020-01-01', periods=n_obs, freq='D')

y1 = np.zeros(n_obs)
y2 = np.zeros(n_obs)
eps1 = np.random.normal(0, 1, n_obs)
eps2 = np.random.normal(0, 1, n_obs)

for t in range(1, n_obs):
    y1[t] = 0.5 * y1[t-1] + 0.3 * y2[t-1] + eps1[t]
    y2[t] = 0.2 * y1[t-1] + 0.6 * y2[t-1] + eps2[t]

data = pd.DataFrame({
    'y1': y1,
    'y2': y2
}, index=dates)

print("Сгенерированные данные:")
print(data.head(10))
print(f"Размер данных: {data.shape}")

print("\n=== Создание и запуск модели ===")
model = AdvancedVARModel(data, max_lags=5)

print("\n=== Запуск полного анализа ===")
model.run_entire_analysis(
    significance_level=0.05,
    criteria=['aic', 'bic'],
    forecast_steps=5,
    oos_train_size=0.8
)

print("\n=== Тестирование завершено ===")
