"""
Os dois modelos finais escolhidos no M3.

- Trilha A (prever a potencia em kW): HistGradientBoosting
- Trilha B (avisar se havera queda): Random Forest

Os valores dos hiperparametros do primeiro sao os que a busca do M3 encontrou;
foram fixados aqui para nao precisar rodar a busca de novo.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error

from . import config


def rmse(y_verdadeiro, y_previsto):
    """Raiz do erro quadratico medio (em kW). Quanto menor, melhor."""
    return float(np.sqrt(mean_squared_error(y_verdadeiro, y_previsto)))


def regressor_final():
    """Modelo da Trilha A: preve a potencia AC de 1 hora a frente.
    Os numeros abaixo sao os melhores hiperparametros achados no M3."""
    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.RANDOM_STATE,
    )


def classificador_final():
    """Modelo da Trilha B: avisa se a geracao vai cair abaixo do normal."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def prever_potencia(modelo, X):
    """Faz a previsao da Trilha A e zera valores negativos.
    Potencia nao pode ser negativa, entao qualquer previsao abaixo de zero
    vira zero (mesmo np.clip usado no M3)."""
    previsao = modelo.predict(X)
    return np.clip(previsao, 0, None)
