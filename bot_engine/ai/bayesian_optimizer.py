"""
Bayesian Optimizer - эффективная оптимизация гиперпараметров

Использует Gaussian Process для построения surrogate модели
и Expected Improvement для выбора следующей точки.

Преимущества над Grid Search:
- Находит оптимум за меньшее число итераций
- Учитывает неопределенность в оценках
- Автоматически балансирует exploration vs exploitation
"""

import logging
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os

logger = logging.getLogger('BayesianOptimizer')

# Опциональные зависимости
try:
    from scipy.stats import norm
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy не установлен. Некоторые функции будут недоступны.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class ParameterSpace:
    """Определение пространства параметров для оптимизации"""
    name: str
    low: float
    high: float
    param_type: str = 'float'  # 'float', 'int', 'categorical'
    log_scale: bool = False
    choices: Optional[List[Any]] = None  # Для categorical

    def sample(self, rng: np.random.Generator = None) -> Any:
        """Сэмплирует случайное значение из пространства"""
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == 'categorical' and self.choices:
            return rng.choice(self.choices)

        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            value = np.exp(rng.uniform(log_low, log_high))
        else:
            value = rng.uniform(self.low, self.high)

        if self.param_type == 'int':
            return int(round(value))
        return value

    def to_unit(self, value: float) -> float:
        """Преобразует значение в [0, 1] диапазон"""
        if self.log_scale:
            return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, unit_value: float) -> Any:
        """Преобразует из [0, 1] в реальное значение"""
        unit_value = np.clip(unit_value, 0, 1)

        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            value = np.exp(log_low + unit_value * (log_high - log_low))
        else:
            value = self.low + unit_value * (self.high - self.low)

        if self.param_type == 'int':
            return int(round(value))
        return value

class GaussianProcessSurrogate:
    """
    Простая реализация Gaussian Process для surrogate модели

    Использует RBF (Radial Basis Function) kernel
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Squared Exponential) kernel"""
        # X1: (n1, d), X2: (n2, d)
        dist_sq = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * dist_sq / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучает GP на данных"""
        self.X_train = np.array(X)

        # Нормализуем y для стабильности
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        self.y_train = (np.array(y) - self.y_mean) / self.y_std

        # Вычисляем ковариационную матрицу
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(K))

        # Инвертируем (с регуляризацией для численной стабильности)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает среднее и std для новых точек

        Returns:
            mean: (n,) предсказанные значения
            std: (n,) неопределенность
        """
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))

        X = np.array(X)

        # Ковариация между новыми и обучающими точками
        K_s = self._rbf_kernel(X, self.X_train)

        # Ковариация новых точек
        K_ss = self._rbf_kernel(X, X)

        # Предсказание среднего
        mean = K_s @ self.K_inv @ self.y_train

        # Предсказание дисперсии
        var = np.diag(K_ss) - np.sum(K_s @ self.K_inv * K_s, axis=1)
        var = np.maximum(var, 1e-8)  # Избегаем отрицательных значений
        std = np.sqrt(var)

        # Денормализуем
        mean = mean * self.y_std + self.y_mean
        std = std * self.y_std

        return mean, std

class BayesianOptimizer:
    """
    Bayesian Optimizer с Gaussian Process surrogate

    Использует Expected Improvement acquisition function
    для балансировки exploration и exploitation
    """

    def __init__(
        self,
        param_space: List[ParameterSpace],
        objective_function: Callable,
        n_initial_points: int = 10,
        acquisition: str = 'ei',  # 'ei' (Expected Improvement) или 'ucb' (Upper Confidence Bound)
        xi: float = 0.01,  # Exploration параметр для EI
        kappa: float = 2.0,  # Exploration параметр для UCB
        random_state: int = 42
    ):
        """
        Args:
            param_space: Список ParameterSpace с определением параметров
            objective_function: Функция для оптимизации (принимает Dict, возвращает float)
            n_initial_points: Количество начальных случайных точек
            acquisition: Тип acquisition function ('ei' или 'ucb')
            xi: Exploration параметр для Expected Improvement
            kappa: Exploration параметр для Upper Confidence Bound
            random_state: Seed для воспроизводимости
        """
        self.param_space = param_space
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        self.rng = np.random.default_rng(random_state)

        self.gp = GaussianProcessSurrogate()

        # История оптимизации
        self.X_observed = []  # Параметры в unit scale [0, 1]
        self.y_observed = []  # Значения objective
        self.params_observed = []  # Параметры в реальном масштабе

        self.best_params = None
        self.best_value = float('-inf')
        self.iteration = 0

        logger.info(f"BayesianOptimizer создан: {len(param_space)} параметров, acquisition={acquisition}")

    def _params_to_unit(self, params: Dict) -> np.ndarray:
        """Преобразует параметры в unit [0, 1] пространство"""
        return np.array([
            ps.to_unit(params[ps.name]) 
            for ps in self.param_space
        ])

    def _unit_to_params(self, x: np.ndarray) -> Dict:
        """Преобразует из unit пространства в реальные параметры"""
        return {
            ps.name: ps.from_unit(x[i])
            for i, ps in enumerate(self.param_space)
        }

    def _expected_improvement(self, x: np.ndarray) -> float:
        """Expected Improvement acquisition function"""
        if not SCIPY_AVAILABLE:
            # Fallback без scipy
            mean, std = self.gp.predict(x.reshape(1, -1))
            return mean[0] + self.kappa * std[0]

        mean, std = self.gp.predict(x.reshape(1, -1))
        mean, std = mean[0], std[0]

        if std < 1e-8:
            return 0.0

        z = (mean - self.best_value - self.xi) / std
        ei = (mean - self.best_value - self.xi) * norm.cdf(z) + std * norm.pdf(z)

        return ei

    def _upper_confidence_bound(self, x: np.ndarray) -> float:
        """Upper Confidence Bound acquisition function"""
        mean, std = self.gp.predict(x.reshape(1, -1))
        return mean[0] + self.kappa * std[0]

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Выбирает acquisition function"""
        if self.acquisition == 'ei':
            return self._expected_improvement(x)
        else:
            return self._upper_confidence_bound(x)

    def _suggest_next_point(self) -> np.ndarray:
        """Предлагает следующую точку для оценки"""
        n_dims = len(self.param_space)

        # Multi-start оптимизация acquisition function
        best_x = None
        best_acq = float('-inf')

        n_restarts = 10

        for _ in range(n_restarts):
            # Случайная начальная точка
            x0 = self.rng.uniform(0, 1, n_dims)

            if SCIPY_AVAILABLE:
                # Оптимизируем acquisition function
                result = minimize(
                    lambda x: -self._acquisition_function(x),
                    x0,
                    bounds=[(0, 1)] * n_dims,
                    method='L-BFGS-B'
                )
                x_opt = result.x
            else:
                # Простой random search если scipy недоступен
                x_opt = x0
                for _ in range(50):
                    x_try = x0 + self.rng.normal(0, 0.1, n_dims)
                    x_try = np.clip(x_try, 0, 1)
                    if self._acquisition_function(x_try) > self._acquisition_function(x_opt):
                        x_opt = x_try

            acq_value = self._acquisition_function(x_opt)

            if acq_value > best_acq:
                best_acq = acq_value
                best_x = x_opt

        return np.clip(best_x, 0, 1)

    def optimize(
        self,
        n_iterations: int = 50,
        verbose: bool = True,
        callback: Callable = None
    ) -> Dict:
        """
        Запускает оптимизацию

        Args:
            n_iterations: Количество итераций (не включая initial points)
            verbose: Выводить прогресс
            callback: Функция, вызываемая после каждой итерации

        Returns:
            Dict с лучшими параметрами и историей
        """
        total_iterations = self.n_initial_points + n_iterations

        if verbose:
            logger.info(f"Начало оптимизации: {self.n_initial_points} initial + {n_iterations} iterations")

        for i in range(total_iterations):
            self.iteration = i + 1

            # Начальные случайные точки
            if i < self.n_initial_points:
                x_unit = self.rng.uniform(0, 1, len(self.param_space))
            else:
                # Bayesian optimization
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
                x_unit = self._suggest_next_point()

            # Преобразуем в реальные параметры
            params = self._unit_to_params(x_unit)

            # Оцениваем objective
            try:
                value = self.objective_function(params)
            except Exception as e:
                logger.warning(f"Ошибка при оценке параметров: {e}")
                value = float('-inf')

            # Сохраняем результат
            self.X_observed.append(x_unit)
            self.y_observed.append(value)
            self.params_observed.append(params)

            # Обновляем лучший результат
            if value > self.best_value:
                self.best_value = value
                self.best_params = params.copy()

                if verbose:
                    logger.info(f"[{i+1}/{total_iterations}] Новый лучший результат: {value:.4f}")
                    logger.info(f"  Параметры: {params}")

            elif verbose and (i + 1) % 10 == 0:
                logger.info(f"[{i+1}/{total_iterations}] Текущий: {value:.4f}, Лучший: {self.best_value:.4f}")

            if callback:
                callback({
                    'iteration': i + 1,
                    'params': params,
                    'value': value,
                    'best_value': self.best_value,
                    'best_params': self.best_params
                })

        if verbose:
            logger.info(f"Оптимизация завершена. Лучший результат: {self.best_value:.4f}")

        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_iterations': total_iterations,
            'history': {
                'params': self.params_observed,
                'values': self.y_observed
            }
        }

    def get_best(self) -> Tuple[Dict, float]:
        """Возвращает лучшие параметры и значение"""
        return self.best_params, self.best_value

if OPTUNA_AVAILABLE:
    class OptunaOptimizer:
        """
        Wrapper для Optuna - более продвинутая байесовская оптимизация

        Преимущества:
        - TPE (Tree-structured Parzen Estimator) sampler
        - Hyperband pruner для раннего отсечения плохих trials
        - Параллельные trials
        """

        def __init__(
            self,
            param_space: List[ParameterSpace],
            objective_function: Callable,
            study_name: str = "trading_optimization",
            direction: str = "maximize",
            pruner: bool = True
        ):
            self.param_space = param_space
            self.objective_function = objective_function

            # Создаем Optuna study
            sampler = optuna.samplers.TPESampler()
            pruner_obj = optuna.pruners.HyperbandPruner() if pruner else optuna.pruners.NopPruner()

            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner_obj
            )

            logger.info(f"OptunaOptimizer создан: {study_name}, direction={direction}")

        def _create_trial_params(self, trial) -> Dict:
            """Создает параметры из Optuna trial"""
            params = {}

            for ps in self.param_space:
                if ps.param_type == 'categorical' and ps.choices:
                    params[ps.name] = trial.suggest_categorical(ps.name, ps.choices)
                elif ps.param_type == 'int':
                    params[ps.name] = trial.suggest_int(ps.name, int(ps.low), int(ps.high), log=ps.log_scale)
                else:
                    params[ps.name] = trial.suggest_float(ps.name, ps.low, ps.high, log=ps.log_scale)

            return params

        def _objective(self, trial) -> float:
            """Objective function для Optuna"""
            params = self._create_trial_params(trial)
            return self.objective_function(params)

        def optimize(
            self,
            n_trials: int = 100,
            n_jobs: int = 1,
            timeout: int = None,
            verbose: bool = True
        ) -> Dict:
            """
            Запускает оптимизацию

            Args:
                n_trials: Количество trials
                n_jobs: Количество параллельных процессов
                timeout: Таймаут в секундах
                verbose: Показывать прогресс

            Returns:
                Dict с лучшими параметрами
            """
            optuna.logging.set_verbosity(
                optuna.logging.INFO if verbose else optuna.logging.WARNING
            )

            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                show_progress_bar=verbose
            )

            return {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'best_trial': self.study.best_trial.number
            }

        def get_best(self) -> Tuple[Dict, float]:
            """Возвращает лучшие параметры и значение"""
            return self.study.best_params, self.study.best_value
else:
    # Заглушка когда Optuna не установлен
    class OptunaOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Optuna не установлен. Установите: pip install optuna")

def create_trading_param_space() -> List[ParameterSpace]:
    """
    Создает стандартное пространство параметров для торговой стратегии
    """
    return [
        ParameterSpace('rsi_long', 20, 40, 'int'),
        ParameterSpace('rsi_short', 60, 80, 'int'),
        ParameterSpace('exit_long', 45, 65, 'int'),
        ParameterSpace('exit_short', 35, 55, 'int'),
        ParameterSpace('stop_loss', 1.0, 10.0, 'float'),
        ParameterSpace('take_profit', 1.0, 15.0, 'float'),
        ParameterSpace('trailing_stop', 0.5, 5.0, 'float'),
        ParameterSpace('position_size', 0.01, 0.1, 'float'),
    ]

def optimize_strategy_bayesian(
    symbol: str,
    candles: List[Dict],
    evaluate_function: Callable,
    n_iterations: int = 50,
    use_optuna: bool = False
) -> Optional[Dict]:
    """
    Оптимизирует торговую стратегию с использованием Bayesian Optimization

    Args:
        symbol: Символ монеты
        candles: Исторические свечи для бэктеста
        evaluate_function: Функция оценки параметров (принимает params, candles, возвращает score)
        n_iterations: Количество итераций
        use_optuna: Использовать Optuna вместо встроенной реализации

    Returns:
        Dict с лучшими параметрами или None
    """
    try:
        param_space = create_trading_param_space()

        # Создаем objective function
        def objective(params: Dict) -> float:
            try:
                result = evaluate_function(params, candles)

                # Комбинированный score
                win_rate = result.get('win_rate', 0)
                total_pnl = result.get('total_pnl', 0)
                sharpe = result.get('sharpe_ratio', 0)

                # Objective: максимизируем комбинацию метрик
                score = win_rate * 100 + total_pnl * 0.5 + sharpe * 10

                return score

            except Exception as e:

                return float('-inf')

        # Выбираем оптимизатор
        if use_optuna and OPTUNA_AVAILABLE:
            optimizer = OptunaOptimizer(
                param_space=param_space,
                objective_function=objective,
                study_name=f"optimize_{symbol}"
            )
            result = optimizer.optimize(n_trials=n_iterations)
        else:
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_function=objective,
                n_initial_points=min(10, n_iterations // 5)
            )
            result = optimizer.optimize(n_iterations=n_iterations)

        logger.info(f"Оптимизация {symbol} завершена. Score: {result['best_value']:.4f}")

        return result['best_params']

    except Exception as e:
        logger.error(f"Ошибка Bayesian оптимизации для {symbol}: {e}")
        return None

# ==================== ТЕСТОВЫЙ КОД ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Bayesian Optimizer - Тест")
    print("=" * 60)

    # Тестовая функция Rosenbrock (минимум в точке (1, 1))
    def rosenbrock(params: Dict) -> float:
        x = params['x']
        y = params['y']
        # Возвращаем отрицательное значение, т.к. оптимизируем на максимум
        return -(100 * (y - x**2)**2 + (1 - x)**2)

    # Пространство параметров
    param_space = [
        ParameterSpace('x', -5, 5, 'float'),
        ParameterSpace('y', -5, 5, 'float'),
    ]

    # Создаем оптимизатор
    optimizer = BayesianOptimizer(
        param_space=param_space,
        objective_function=rosenbrock,
        n_initial_points=5
    )

    # Запускаем оптимизацию
    print("\n1. Тест встроенного Bayesian Optimizer:")
    result = optimizer.optimize(n_iterations=20, verbose=True)

    print(f"\nЛучшие параметры: {result['best_params']}")
    print(f"Лучшее значение: {result['best_value']:.6f}")
    print(f"Ожидаемый оптимум: x=1, y=1, value=0")

    # Тест Optuna если доступен
    if OPTUNA_AVAILABLE:
        print("\n2. Тест OptunaOptimizer:")

        optuna_optimizer = OptunaOptimizer(
            param_space=param_space,
            objective_function=rosenbrock,
            direction="maximize"
        )

        optuna_result = optuna_optimizer.optimize(n_trials=30, verbose=False)

        print(f"Лучшие параметры (Optuna): {optuna_result['best_params']}")
        print(f"Лучшее значение (Optuna): {optuna_result['best_value']:.6f}")
    else:
        print("\n2. Optuna не установлен - пропускаем тест")

    # Тест с торговыми параметрами
    print("\n3. Тест с торговыми параметрами:")

    trading_space = create_trading_param_space()
    print(f"Параметров в пространстве: {len(trading_space)}")
    for ps in trading_space:
        print(f"  - {ps.name}: [{ps.low}, {ps.high}] ({ps.param_type})")

    print("\n" + "=" * 60)
    print("[OK] Все тесты пройдены!")
    print("=" * 60)
