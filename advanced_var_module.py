import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    het_white,
    het_arch,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults
from statsmodels.tools.eval_measures import rmse, meanabs
import warnings

# --- Внутренние вспомогательные функции ---
def _calculate_theil_u(predicted, actual):
    """Рассчитывает статистику Тейла (Theil's U) для прогноза."""
    if len(predicted) != len(actual) or len(predicted) == 0:
        return np.nan
    numerator = np.sqrt(np.mean((predicted - actual) ** 2))
    # Прогноз "random walk": следующее значение = текущее значение
    baseline_forecast = actual[:-1]
    baseline_actual = actual[1:]
    if len(baseline_forecast) == 0:
        return np.nan
    denominator = np.sqrt(np.mean((baseline_forecast - baseline_actual) ** 2))
    if denominator == 0:
        return np.nan
    return numerator / denominator

def _meanabspe(actual, predicted):
    """Рассчитывает среднюю абсолютную процентную ошибку (MAPE)."""
    if len(actual) != len(predicted) or len(actual) == 0:
        return np.nan
    # Избегаем деления на 0
    mask = actual != 0
    if not mask.any():
        return np.nan
    abspe = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    return np.mean(abspe) * 100

class AdvancedVARModel:
    """
    Готовый модуль с расширенными функциями для исследовательского анализа VAR-моделей.

    Позволяет пройти полный цикл анализа: от подготовки данных до оценки прогнозов.
    """

    def __init__(self, data, exog=None, max_lags=12):
        """
        Инициализирует модель.

        Args:
            data (pd.DataFrame): DataFrame с переменными для VAR-модели.
            exog (pd.DataFrame, optional): Экзогенные переменные. Defaults to None.
            max_lags (int): Максимальное количество лагов для рассмотрения. Defaults to 12.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Данные должны быть переданы в виде pandas.DataFrame.")
        self.data = data.copy()
        self.exog = exog
        self.max_lags = max_lags
        self.original_data = self.data.copy()
        self.stationary_data = None
        self.differencing_orders = {}
        self.selected_lag = None
        self.model = None
        self.fitted_model = None
        self.irf = None
        self.fevd = None
        self.forecast_results = None
        self.diagnostics_results = {}
        self.stability_results = {}

    def preprocess_and_stationarize(self, significance_level=0.05):
        """
        Проверяет стационарность рядов и преобразует их, если необходимо.

        Args:
            significance_level (float): Уровень значимости для теста ADF. Defaults to 0.05.
        """
        print("Шаг 1: Проверка стационарности и преобразование данных...")
        self.stationary_data = self.data.copy()
        self.differencing_orders = {col: 0 for col in self.data.columns}

        for col in self.data.columns:
            series = self.stationary_data[col].dropna()
            adf_result = adfuller(series)
            p_value = adf_result[1]
            print(f"  - Переменная '{col}': ADF p-value = {p_value:.4f}")

            order = 0
            while p_value > significance_level:
                if order >= 2:
                    print(f"    Внимание: {col} может быть нестационарной даже после 2-го дифференцирования.")
                    break
                order += 1
                self.stationary_data[col] = self.stationary_data[col].diff().dropna()
                adf_result = adfuller(self.stationary_data[col].dropna())
                p_value = adf_result[1]
                print(f"    Применено {order}-е дифференцирование. Новый p-value = {p_value:.4f}")
            
            self.differencing_orders[col] = order
            if p_value <= significance_level:
                 print(f"    Переменная '{col}' стационарна (порядок дифференцирования: {order}).")

        print("Преобразование данных завершено.")
        return self.stationary_data

    def select_optimal_lag(self, criteria=['aic', 'bic', 'hqic']):
        """
        Выбирает оптимальный порядок лага на основе информационных критериев.

        Args:
            criteria (list): Список критериев ('aic', 'bic', 'hqic', 'fpe'). Defaults to ['aic', 'bic', 'hqic'].
        """
        print("\nШаг 2: Выбор оптимального порядка лага...")
        if self.stationary_data is None:
            raise ValueError("Данные не были преобразованы в стационарные. Вызовите preprocess_and_stationarize().")
        
        self.model = VAR(self.stationary_data, exog=self.exog)
        lag_order_results = self.model.select_order(maxlags=self.max_lags, trend='c')

        print("Значения критериев для разных лагов:")
        print(lag_order_results.summary())
        
        selected_orders = {}
        for crit in criteria:
            if hasattr(lag_order_results, 'summary_frame'):
                # Для более новых версий statsmodels
                summary_df = lag_order_results.summary_frame
                if crit.upper() in summary_df.index:
                    opt_lag = summary_df.loc[crit.upper()].idxmin()
                    selected_orders[crit] = opt_lag
                else:
                    print(f"Критерий {crit} не найден в summary_frame. Попробуем стандартный метод.")
                    # Старый метод
                    if crit == 'aic':
                        selected_orders[crit] = lag_order_results.aic
                    elif crit == 'bic':
                        selected_orders[crit] = lag_order_results.bic
                    elif crit == 'hqic':
                        selected_orders[crit] = lag_order_results.hqic
                    elif crit == 'fpe':
                        selected_orders[crit] = lag_order_results.fpe
                    else:
                        print(f"Критерий {crit} не поддерживается.")
            else:
                # Старый метод
                if crit == 'aic':
                    selected_orders[crit] = lag_order_results.aic
                elif crit == 'bic':
                    selected_orders[crit] = lag_order_results.bic
                elif crit == 'hqic':
                    selected_orders[crit] = lag_order_results.hqic
                elif crit == 'fpe':
                    selected_orders[crit] = lag_order_results.fpe
                else:
                    print(f"Критерий {crit} не поддерживается.")

        print("\nВыбранные лаги по критериям:")
        for crit, lag in selected_orders.items():
            print(f"  - {crit.upper()}: {lag}")
        
        # Выбираем AIC по умолчанию, если он доступен
        self.selected_lag = selected_orders.get('aic', list(selected_orders.values())[0])
        print(f"\nДля дальнейшего анализа выбран лаг p = {self.selected_lag} (по критерию AIC).")
        return self.selected_lag

    def fit_model(self, lag_order=None):
        """
        Оценивает VAR-модель.

        Args:
            lag_order (int, optional): Порядок лага. Если None, используется self.selected_lag.
        """
        print(f"\nШаг 3: Оценка VAR-модели (p={lag_order or self.selected_lag})...")
        if self.model is None:
            raise ValueError("Модель не была инициализирована. Вызовите select_optimal_lag().")
        
        if lag_order is None:
            lag_order = self.selected_lag
        
        if lag_order is None:
            raise ValueError("Порядок лага не определен. Вызовите select_optimal_lag() или передайте его в fit_model().")
        
        self.fitted_model = self.model.fit(maxlags=lag_order, trend='c')
        print("Модель успешно оценена.")
        return self.fitted_model

    def run_diagnostics(self):
        """
        Выполняет расширенную диагностику остатков модели.
        """
        print("\nШаг 4: Расширенная диагностика остатков...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        residuals = self.fitted_model.resid
        nobs = residuals.shape[0]
        
        # 1. Тест на автокорреляцию (Ljung-Box)
        print("  - Тест на автокорреляцию остатков (Ljung-Box)...")
        # Выбираем разумное количество лагов для теста, например, 10
        lags_lb = min(10, nobs // 5) if nobs > 20 else 5
        lb_test = acorr_ljungbox(residuals, lags=lags_lb, return_df=True)
        self.diagnostics_results['autocorrelation'] = lb_test
        print(f"    Проверено до лага {lags_lb}.")
        # Выводим p-values для каждого уравнения
        for col in self.data.columns:
            p_vals = lb_test[f'lb_pvalue'][col]
            significant_lags = [i+1 for i, p in enumerate(p_vals) if p < 0.05]
            if significant_lags:
                print(f"      Уравнение для '{col}': p < 0.05 при лагах {significant_lags} (потенциальная автокорреляция).")
            else:
                print(f"      Уравнение для '{col}': p >= 0.05 для всех лагов (автокорреляция не обнаружена).")

        # 2. Тесты на гетероскедастичность
        print("  - Тесты на гетероскедастичность...")
        # Для Breusch-Pagan и White нужна X (регрессоры), которые можно получить из модели
        # statsmodels VAR не предоставляет X напрямую, но мы можем использовать endog_lagged
        # Однако, для простоты, используем только тесты, которые можно применить к остаткам
        # statsmodels не имеет встроенного теста BP для VAR residuals напрямую через VARResults
        # Используем тест ARCH для условной гетероскедастичности
        print("    - Тест ARCH для условной гетероскедастичности...")
        arch_results = []
        for i, col in enumerate(self.data.columns):
            arch_res = het_arch(residuals.iloc[:, i])
            arch_results.append(arch_res)
            if arch_res[1] < 0.05: # p-value
                print(f"      '{col}': ARCH тест значим (p={arch_res[1]:.4f}), есть условная гетероскедастичность.")
            else:
                print(f"      '{col}': ARCH тест не значим (p={arch_res[1]:.4f}), условной гетероскедастичности нет.")
        self.diagnostics_results['arch'] = arch_results

        # 3. Тест на нормальность (Jarque-Bera)
        print("  - Тест на нормальность остатков (Jarque-Bera)...")
        jb_results = []
        for i, col in enumerate(self.data.columns):
            jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals.iloc[:, i])
            jb_results.append((jb_stat, jb_pvalue, skew, kurtosis))
            if jb_pvalue < 0.05:
                print(f"      '{col}': JB тест значим (p={jb_pvalue:.4f}), отклоняется гипотеза о нормальности.")
            else:
                print(f"      '{col}': JB тест не значим (p={jb_pvalue:.4f}), гипотеза о нормальности не отвергается.")
        self.diagnostics_results['normality'] = jb_results

        print("Диагностика завершена.")
        return self.diagnostics_results

    def check_stability(self):
        """
        Проверяет стабильность VAR-модели.
        """
        print("\nШаг 5: Проверка стабильности модели...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        is_stable = self.fitted_model.is_stable()
        roots = self.fitted_model.irf()._ev
        self.stability_results['is_stable'] = is_stable
        self.stability_results['roots'] = roots

        if is_stable:
            print("  Модель стационарна (все корни внутри единичного круга).")
        else:
            print("  ВНИМАНИЕ: Модель НЕстационарна (некоторые корни находятся НА или СНАРУЖИ единичного круга).")
        
        # Визуализация корней
        plt.figure(figsize=(6,6))
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Единичный круг')
        roots_real = np.real(roots)
        roots_imag = np.imag(roots)
        plt.scatter(roots_real, roots_imag, label='Корни', color='red')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.title('График корней характеристического уравнения')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        return is_stable

    def analyze_granger_causality(self, max_lag=None):
        """
        Проводит тесты Грейнджера на причинность между всеми парами переменных.

        Args:
            max_lag (int, optional): Максимальный лаг для теста. Defaults to self.selected_lag.
        """
        print("\nШаг 6: Анализ причинности по Грейнджеру...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        if max_lag is None:
            max_lag = self.selected_lag
        if max_lag is None:
            max_lag = 1 # Fallback

        causality_results = {}
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 != col2:
                    try:
                        # Используем индексы столбцов
                        idx1 = list(self.data.columns).index(col1)
                        idx2 = list(self.data.columns).index(col2)
                        granger_res = self.fitted_model.test_causality(caused=idx2, causing=idx1, kind='f', maxlag=max_lag)
                        p_value = granger_res.pvalue
                        causality_results[(col1, col2)] = p_value
                        if p_value < 0.05:
                            print(f"  {col1} Грейнджер-причинит {col2} (p={p_value:.4f})")
                        else:
                            print(f"  {col1} НЕ Грейнджер-причинит {col2} (p={p_value:.4f})")
                    except Exception as e:
                        print(f"Ошибка при тесте Грейнджера для {col1} -> {col2}: {e}")
                        causality_results[(col1, col2)] = np.nan
        
        self.causality_results = causality_results
        return causality_results

    def plot_irf(self, periods=10, impulse_var=None, response_var=None):
        """
        Строит функции импульсной реакции (IRF).

        Args:
            periods (int): Количество периодов для отображения. Defaults to 10.
            impulse_var (str, optional): Переменная, для которой моделируется шок. Defaults to all.
            response_var (str, optional): Переменная, на которую смотрим реакцию. Defaults to all.
        """
        print(f"\nШаг 7: Построение функций импульсной реакции (IRF)...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        self.irf = IRAnalysis(self.fitted_model, periods=periods)
        fig, axes = self.irf.plot(orth=True, impulse=impulse_var, response=response_var, figsize=(12, 8))
        plt.suptitle('Функции импульсной реакции (ортогонализованные)')
        plt.show()

    def plot_fevd(self, periods=10):
        """
        Строит декомпозицию ошибок прогнозирования (FEVD).

        Args:
            periods (int): Горизонт для декомпозиции. Defaults to 10.
        """
        print(f"\nШаг 8: Построение декомпозиции ошибок прогнозирования (FEVD)...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        # FEVD теперь доступна как метод объекта VARResults
        self.fevd = self.fitted_model.fevd(periods=periods)
        fig, axes = self.fevd.plot(figsize=(12, 8))
        plt.suptitle('Декомпозиция ошибок прогнозирования')
        plt.show()

    def make_forecast(self, steps=5, confidence_level=0.95):
        """
        Делает прогноз на заданное количество шагов вперед.

        Args:
            steps (int): Количество шагов вперед. Defaults to 5.
            confidence_level (float): Уровень доверия для интервалов. Defaults to 0.95.
        """
        print(f"\nШаг 9: Генерация прогноза на {steps} шагов вперед...")
        if self.fitted_model is None:
            raise ValueError("Модель не была оценена. Вызовите fit_model().")
        
        # Используем последние k_ar наблюдений для прогноза
        forecast_input = self.stationary_data.values[-self.fitted_model.k_ar:]
        forecast_mean = self.fitted_model.forecast(forecast_input, steps=steps)
        forecast_ci = self.fitted_model.forecast_interval(forecast_input, steps=steps, alpha=1-confidence_level)

        # Создаем DataFrame для удобства
        forecast_index = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=steps, freq=self.data.index.freq)
        if forecast_index.freq is None: # Если freq не определен, используем индекс
            forecast_index = range(len(self.data), len(self.data) + steps)
        
        self.forecast_results = {
            'mean': pd.DataFrame(forecast_mean, index=forecast_index, columns=self.data.columns),
            'lower': pd.DataFrame(forecast_ci[0], index=forecast_index, columns=self.data.columns),
            'upper': pd.DataFrame(forecast_ci[1], index=forecast_index, columns=self.data.columns)
        }
        
        print("Прогноз:")
        print(self.forecast_results['mean'])
        print(f"Доверительные интервалы ({confidence_level*100}%):")
        print(f"Нижняя граница:\n{self.forecast_results['lower']}")
        print(f"Верхняя граница:\n{self.forecast_results['upper']}")
        
        # Визуализация прогноза
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(self.data.columns):
            plt.subplot(len(self.data.columns), 1, i+1)
            plt.plot(self.data.index, self.data[col], label='История', color='blue')
            plt.plot(self.forecast_results['mean'].index, self.forecast_results['mean'][col], label='Прогноз', color='red', linestyle='--')
            plt.fill_between(self.forecast_results['lower'].index, self.forecast_results['lower'][col], self.forecast_results['upper'][col], color='red', alpha=0.2, label='Доверительный интервал')
            plt.title(f'Прогноз для {col}')
            plt.legend()
        plt.tight_layout()
        plt.show()

        return self.forecast_results

    def evaluate_out_of_sample(self, train_size=0.7, forecast_steps=1, method='expanding'):
        """
        Проводит out-of-sample оценку производительности.

        Args:
            train_size (float): Доля данных для обучающей выборки. Defaults to 0.7.
            forecast_steps (int): Количество шагов вперед для прогноза. Defaults to 1.
            method (str): 'expanding' или 'rolling'. Defaults to 'expanding'.
        """
        print(f"\nШаг 10: Out-of-sample оценка производительности ({method} window)...")
        if self.stationary_data is None:
            raise ValueError("Данные не были преобразованы в стационарные. Вызовите preprocess_and_stationarize().")
        if self.selected_lag is None:
            raise ValueError("Порядок лага не выбран. Вызовите select_optimal_lag().")
        if self.exog is not None:
            # Учет экзогенных переменных требует дополнительной логики
            raise NotImplementedError("Out-of-sample оценка с экзогенными переменными не реализована в этом примере.")

        n_total = len(self.stationary_data)
        n_train = int(n_total * train_size)
        n_test = n_total - n_train - self.selected_lag # Учитываем лаги для начального прогноза

        if n_test < forecast_steps:
            raise ValueError("Недостаточно данных для теста.")

        actual_values = []
        forecast_values = []

        for i in range(n_test):
            if method == 'expanding':
                train_end_idx = n_train + i
            elif method == 'rolling':
                train_end_idx = n_train + i
                train_start_idx = max(0, train_end_idx - n_train)
            else:
                raise ValueError("Method must be 'expanding' or 'rolling'.")

            if method == 'rolling':
                train_data = self.stationary_data.iloc[train_start_idx:train_end_idx]
            else: # expanding
                train_data = self.stationary_data.iloc[:train_end_idx]

            test_data = self.stationary_data.iloc[train_end_idx:train_end_idx + forecast_steps]
            if len(test_data) < forecast_steps:
                break # Не хватает данных для прогноза

            # Оценка модели на обучающей выборке
            temp_model = VAR(train_data)
            temp_fitted_model = temp_model.fit(maxlags=self.selected_lag, trend='c')

            # Прогноз
            forecast_input = train_data.values[-self.selected_lag:]
            forecast_mean = temp_fitted_model.forecast(forecast_input, steps=forecast_steps)

            # Сохранение результатов
            actual_values.append(test_data.values[0]) # Только первый шаг
            forecast_values.append(forecast_mean[0]) # Только первый шаг

        if not actual_values or not forecast_values:
            print("Не удалось сгенерировать out-of-sample прогнозы.")
            return {}

        actual_df = pd.DataFrame(actual_values, columns=self.data.columns)
        forecast_df = pd.DataFrame(forecast_values, columns=self.data.columns)

        # Расчет метрик
        metrics = {}
        for col in self.data.columns:
            actual_col = actual_df[col].dropna()
            forecast_col = forecast_df[col].dropna()
            # Убедимся, что длины совпадают
            min_len = min(len(actual_col), len(forecast_col))
            actual_slice = actual_col.iloc[:min_len]
            forecast_slice = forecast_col.iloc[:min_len]
            
            if len(actual_slice) == 0:
                continue

            mae = meanabs(actual_slice, forecast_slice)
            rmse_val = rmse(actual_slice, forecast_slice)
            mape = _meanabspe(actual_slice, forecast_slice) # Используем нашу функцию
            theil_u = _calculate_theil_u(forecast_slice.values, actual_slice.values)

            metrics[col] = {
                'MAE': mae,
                'RMSE': rmse_val,
                'MAPE': mape,
                'Theil_U': theil_u
            }
        
        self.oos_metrics = metrics
        print("\nOut-of-sample метрики:")
        for col, m in metrics.items():
            print(f"  {col}:")
            for metric_name, value in m.items():
                print(f"    {metric_name}: {value:.4f}")
        
        return metrics

    def run_entire_analysis(self, significance_level=0.05, criteria=['aic', 'bic', 'hqic'], forecast_steps=5, oos_train_size=0.7):
        """
        Выполняет полный цикл анализа VAR-модели.

        Args:
            significance_level (float): Уровень значимости для ADF теста.
            criteria (list): Критерии для выбора лага.
            forecast_steps (int): Количество шагов для прогноза.
            oos_train_size (float): Доля данных для обучающей выборки в OOS.
        """
        print("=== ЗАПУСК ПОЛНОГО АНАЛИЗА VAR-МОДЕЛИ ===")
        self.preprocess_and_stationarize(significance_level=significance_level)
        self.select_optimal_lag(criteria=criteria)
        self.fit_model()
        self.run_diagnostics()
        self.check_stability()
        self.analyze_granger_causality()
        self.plot_irf(periods=forecast_steps)
        self.plot_fevd(periods=forecast_steps)
        self.make_forecast(steps=forecast_steps)
        self.evaluate_out_of_sample(train_size=oos_train_size)
        print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")

if __name__ == "__main__":
    print("Модуль AdvancedVARModel загружен. Примеры использования:")
    print("1. Создайте DataFrame с вашими временными рядами.")
    print("2. Инициализируйте модель: model = AdvancedVARModel(data)")
    print("3. Запустите полный анализ: model.run_entire_analysis()")
    print("4. Или выполните шаги по отдельности, как описано в документации.")
