"""
Constantes e rotulos usados no projeto.

Este arquivo guarda os valores fixos (datas, limiares, lista de atributos)
e os nomes "bonitos" das colunas para exibir no dashboard.
Os notebooks do M2 e M3 usam esses mesmos valores; centralizar aqui evita
que o notebook e o app fiquem diferentes um do outro.
"""
import os
import pandas as pd

# Pastas do projeto (calculadas a partir da posicao deste arquivo).
# __file__ e este config.py, dentro de src/. Subimos um nivel para a raiz.
PASTA_SRC = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_SRC)
PASTA_DATASET = os.path.join(PASTA_RAIZ, 'dataset')
PASTA_ARTEFATOS = os.path.join(PASTA_RAIZ, 'artifacts')

# Nomes dos 4 arquivos CSV dentro da pasta dataset/
CSV_GERACAO_1 = 'Plant_1_Generation_Data.csv'
CSV_CLIMA_1 = 'Plant_1_Weather_Sensor_Data.csv'
CSV_GERACAO_2 = 'Plant_2_Generation_Data.csv'
CSV_CLIMA_2 = 'Plant_2_Weather_Sensor_Data.csv'

# Parametros do problema (iguais aos do M3)
RANDOM_STATE = 42
TARGET = 'AC_POWER'                    # variavel de origem (kW)
ALVO = 'AC_POWER_H1'                   # alvo: AC_POWER 1 hora a frente
PASSO_MIN = 15                         # a serie tem uma leitura a cada 15 min
DATA_CORTE = pd.Timestamp('2020-06-10')  # divide treino (antes) e teste (depois)
LIMIAR_BAIXA = 0.50                    # evento de baixa geracao: abaixo de 50% da mediana
N_SPLITS = 5                           # numero de divisoes na validacao cruzada

# Colunas do sensor meteorologico
COLS_CLIMA = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

# Chave que identifica cada inversor (planta + codigo do inversor)
CHAVE = ['PLANTA', 'SOURCE_KEY']

# Lista final de atributos que entram nos modelos
ATRIBUTOS = [
    'AC_POWER', 'AC_POWER_lag1', 'AC_POWER_lag2', 'AC_POWER_lag4',
    'DC_POWER_NORM', 'DC_POWER_lag1_NORM', 'DC_POWER_lag2_NORM', 'DC_POWER_lag4_NORM',
    'IRRADIATION', 'IRRADIATION_lag4',
    'AMBIENT_TEMPERATURE', 'AMBIENT_TEMPERATURE_lag4',
    'MODULE_TEMPERATURE', 'MODULE_TEMPERATURE_lag4',
    'D_IRRADIATION_1H', 'D_AC_POWER_1H', 'D_MODULE_TEMP_1H', 'DELTA_TEMP',
    'AC_MEDIA_1H', 'AC_DESVIO_1H',
    'SIN_DIA', 'COS_DIA', 'MINUTO_DIA', 'PLANTA', 'MEDIANA_SLOT',
]

# Colunas que nao podem ter valor vazio antes de treinar.
# Se qualquer uma estiver vazia numa linha, essa linha e descartada.
COLS_MODELAGEM = [
    'AC_POWER', 'AC_POWER_lag1', 'AC_POWER_lag2', 'AC_POWER_lag4',
    'DC_POWER', 'DC_POWER_lag1', 'DC_POWER_lag2', 'DC_POWER_lag4',
    'IRRADIATION', 'IRRADIATION_lag4',
    'AMBIENT_TEMPERATURE', 'AMBIENT_TEMPERATURE_lag4',
    'MODULE_TEMPERATURE', 'MODULE_TEMPERATURE_lag4',
    'D_IRRADIATION_1H', 'D_AC_POWER_1H', 'D_MODULE_TEMP_1H', 'DELTA_TEMP',
    'AC_MEDIA_1H', 'AC_DESVIO_1H', 'SIN_DIA', 'COS_DIA', 'MINUTO_DIA',
    ALVO,
]

# Nomes tecnicos -> nomes em linguagem do usuario (para o dashboard).
# O app nunca mostra o nome da coluna; sempre passa por este dicionario.
ROTULOS = {
    'AC_POWER': 'Potencia atual (kW)',
    'AC_POWER_lag1': 'Potencia ha 15 min (kW)',
    'AC_POWER_lag2': 'Potencia ha 30 min (kW)',
    'AC_POWER_lag4': 'Potencia ha 1 hora (kW)',
    'DC_POWER_NORM': 'Potencia CC atual (% do maximo da usina)',
    'DC_POWER_lag1_NORM': 'Potencia CC ha 15 min (% do maximo)',
    'DC_POWER_lag2_NORM': 'Potencia CC ha 30 min (% do maximo)',
    'DC_POWER_lag4_NORM': 'Potencia CC ha 1 hora (% do maximo)',
    'IRRADIATION': 'Irradiancia solar atual',
    'IRRADIATION_lag4': 'Irradiancia ha 1 hora',
    'AMBIENT_TEMPERATURE': 'Temperatura do ar (C)',
    'AMBIENT_TEMPERATURE_lag4': 'Temperatura do ar ha 1 hora (C)',
    'MODULE_TEMPERATURE': 'Temperatura do painel (C)',
    'MODULE_TEMPERATURE_lag4': 'Temperatura do painel ha 1 hora (C)',
    'D_IRRADIATION_1H': 'Variacao da irradiancia na ultima hora',
    'D_AC_POWER_1H': 'Variacao da potencia na ultima hora (kW)',
    'D_MODULE_TEMP_1H': 'Variacao da temperatura do painel (C)',
    'DELTA_TEMP': 'Aquecimento do painel sobre o ar (C)',
    'AC_MEDIA_1H': 'Potencia media da ultima hora (kW)',
    'AC_DESVIO_1H': 'Oscilacao da potencia na ultima hora (kW)',
    'SIN_DIA': 'Posicao no ciclo diario (seno)',
    'COS_DIA': 'Posicao no ciclo diario (cosseno)',
    'MINUTO_DIA': 'Horario (minutos desde a meia-noite)',
    'PLANTA': 'Usina (1 ou 2)',
    'MEDIANA_SLOT': 'Geracao tipica deste horario (kW)',
}


def rotular(coluna):
    """Devolve o nome em linguagem do usuario para uma coluna.
    Se a coluna nao estiver no dicionario, devolve o proprio nome."""
    return ROTULOS.get(coluna, coluna)
