"""
Carrega os CSVs, junta geracao com clima e cria os atributos.

E o mesmo passo a passo dos notebooks do M2 e M3, so que organizado em
funcoes. Os CSVs sao lidos da pasta dataset/ (ver config.py).
"""
import os
import numpy as np
import pandas as pd

from . import config


def combinar_planta(geracao, clima, numero_planta):
    """Junta os dados de geracao dos inversores com o sensor de clima de uma
    planta e preenche os valores climaticos vazios por interpolacao no tempo.
    E exatamente o que o M2 faz para cada planta."""
    d = pd.merge(
        geracao,
        clima.drop(columns=['SOURCE_KEY']),
        on=['DATE_TIME', 'PLANT_ID'],
        how='left',
    )
    d = d.sort_values('DATE_TIME')
    d[config.COLS_CLIMA] = d[config.COLS_CLIMA].interpolate(
        method='linear', limit_direction='both')
    d['PLANTA'] = numero_planta
    return d


def carregar_base():
    """Le os 4 CSVs da pasta dataset/ e devolve a base das duas plantas juntas.
    E a celula de carregamento do M3."""
    pasta = config.PASTA_DATASET

    gen1 = pd.read_csv(os.path.join(pasta, config.CSV_GERACAO_1))
    wth1 = pd.read_csv(os.path.join(pasta, config.CSV_CLIMA_1))
    gen2 = pd.read_csv(os.path.join(pasta, config.CSV_GERACAO_2))
    wth2 = pd.read_csv(os.path.join(pasta, config.CSV_CLIMA_2))

    # As datas vem como texto; convertemos para data/hora de verdade.
    # A planta 1 usa um formato de data diferente das outras.
    gen1['DATE_TIME'] = pd.to_datetime(gen1['DATE_TIME'], format='%d-%m-%Y %H:%M')
    wth1['DATE_TIME'] = pd.to_datetime(wth1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    gen2['DATE_TIME'] = pd.to_datetime(gen2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    wth2['DATE_TIME'] = pd.to_datetime(wth2['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    df = pd.concat(
        [combinar_planta(gen1, wth1, 1), combinar_planta(gen2, wth2, 2)],
        ignore_index=True,
    )
    df = df.sort_values(['PLANTA', 'SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)
    return df


def deslocar(base, colunas, passos, sufixo):
    """Pega os valores de `colunas` de `passos` * 15 min atras (ou a frente,
    se passos for negativo) e junta de volta na base.

    Usamos merge por data/hora em vez de shift() porque a serie tem buracos:
    um shift(4) andaria 4 LINHAS, e nao 4 passos de 15 minutos. Fazer pela
    data garante que o valor pego e mesmo o de 15/30/60 min antes."""
    delta = pd.Timedelta(minutes=config.PASSO_MIN * passos)
    aux = base[config.CHAVE + ['DATE_TIME'] + colunas].copy()
    aux['DATE_TIME'] = aux['DATE_TIME'] + delta
    novos_nomes = {c: c + sufixo for c in colunas}
    aux = aux.rename(columns=novos_nomes)
    return base.merge(aux, on=config.CHAVE + ['DATE_TIME'], how='left')


def criar_atributos(df):
    """Cria todas as colunas usadas pelos modelos (lags, rampas, medias
    moveis e atributos de horario). E a secao de engenharia de atributos do M3."""

    # Alvo: a potencia daqui a 1 hora (4 passos de 15 min a frente)
    df = deslocar(df, [config.TARGET], -4, '_H1')

    # Potencia de 15, 30 e 60 min atras
    for p in [1, 2, 4]:
        df = deslocar(df, ['AC_POWER', 'DC_POWER'], p, '_lag' + str(p))

    # Clima de 60 min atras (para medir tendencia)
    df = deslocar(df, config.COLS_CLIMA, 4, '_lag4')

    # Variacoes na ultima hora (sinal de nuvem chegando)
    df['D_IRRADIATION_1H'] = df['IRRADIATION'] - df['IRRADIATION_lag4']
    df['D_AC_POWER_1H'] = df['AC_POWER'] - df['AC_POWER_lag4']
    df['D_MODULE_TEMP_1H'] = df['MODULE_TEMPERATURE'] - df['MODULE_TEMPERATURE_lag4']
    df['DELTA_TEMP'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']

    # Media e oscilacao da potencia na ultima hora
    df = df.sort_values(config.CHAVE + ['DATE_TIME'])
    janela = df.set_index('DATE_TIME').groupby(config.CHAVE)['AC_POWER'].rolling('1h')
    df['AC_MEDIA_1H'] = janela.mean().reset_index(level=[0, 1], drop=True).values
    df['AC_DESVIO_1H'] = janela.std().reset_index(level=[0, 1], drop=True).fillna(0).values

    # Horario do dia. Seno e cosseno representam a hora de forma circular
    # (23h fica perto de 0h), o que ajuda o modelo a entender o ciclo do dia.
    df['MINUTO_DIA'] = df['DATE_TIME'].dt.hour * 60 + df['DATE_TIME'].dt.minute
    df['HORA'] = df['DATE_TIME'].dt.hour
    df['SIN_DIA'] = np.sin(2 * np.pi * df['MINUTO_DIA'] / 1440)
    df['COS_DIA'] = np.cos(2 * np.pi * df['MINUTO_DIA'] / 1440)

    # Remove as linhas de borda: quando um lag ou o alvo ficou vazio porque
    # o vizinho de 15/30/60 min nao existe na serie.
    df = df.dropna(subset=config.COLS_MODELAGEM).reset_index(drop=True)
    return df


def preparar_treino_teste(df):
    """Normaliza a potencia CC, calcula a mediana de cada horario, cria o
    rotulo de baixa geracao e separa treino de teste pela data.

    Tudo o que e "aprendido" (o maximo de CC, a mediana por horario) sai so do
    treino, para nao vazar informacao do teste. E a secao de split do M3.
    Devolve tres coisas: a base completa, o treino e o teste."""

    treino = df['DATE_TIME'] < config.DATA_CORTE

    # Normaliza a potencia CC pelo maximo de cada planta (medido so no treino).
    # A planta 1 reporta CC numa escala ~10x maior que a planta 2; sem isso a
    # mesma coluna significaria coisas diferentes em cada planta.
    max_cc = df[treino].groupby('PLANTA')['DC_POWER'].max()
    for col in ['DC_POWER', 'DC_POWER_lag1', 'DC_POWER_lag2', 'DC_POWER_lag4']:
        df[col + '_NORM'] = df[col] / df['PLANTA'].map(max_cc)

    # Mediana da potencia em cada horario e planta (so com dados de treino)
    mediana = (df[treino]
               .groupby(['PLANTA', 'MINUTO_DIA'])[config.TARGET]
               .median()
               .rename('MEDIANA_SLOT')
               .reset_index())
    df = df.merge(mediana, on=['PLANTA', 'MINUTO_DIA'], how='left')
    df['MEDIANA_SLOT'] = df['MEDIANA_SLOT'].fillna(0)

    # DIURNO marca os horarios em que ha geracao (mediana > 0).
    # EVENTO_BAIXA marca quando a potencia futura fica abaixo de 50% do normal.
    df['DIURNO'] = df['MEDIANA_SLOT'] > 0
    abaixo = df[config.ALVO] < config.LIMIAR_BAIXA * df['MEDIANA_SLOT']
    df['EVENTO_BAIXA'] = (abaixo & df['DIURNO']).astype(int)

    # Refaz a mascara de treino (o merge acima mudou a ordem das linhas)
    treino = df['DATE_TIME'] < config.DATA_CORTE
    df_treino = df[treino].sort_values('DATE_TIME').reset_index(drop=True)
    df_teste = df[~treino].sort_values('DATE_TIME').reset_index(drop=True)
    return df, df_treino, df_teste


def pipeline_completo():
    """Atalho que faz tudo de uma vez: le os CSVs, cria os atributos e separa
    treino/teste. Devolve (base_completa, treino, teste)."""
    df = carregar_base()
    df = criar_atributos(df)
    return preparar_treino_teste(df)
