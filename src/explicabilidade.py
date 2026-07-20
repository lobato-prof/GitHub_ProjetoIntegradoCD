"""
Explicabilidade dos modelos com SHAP (parte nova do M4).

SHAP mede quanto cada atributo empurrou uma previsao para cima ou para baixo.
Duas visoes:
- global: quais atributos mais pesam no geral (media do efeito)
- local: por que UMA previsao especifica deu aquele valor (grafico waterfall)

Os valores sao calculados uma vez pelo script treinar_e_exportar.py e salvos.
O dashboard so le o resultado pronto, para responder rapido.
"""
import numpy as np
import pandas as pd
import shap

from . import config


def calcular_shap(modelo, X, background=None, classe=None):
    """Calcula os valores SHAP das linhas em X.

    Devolve dois itens: os valores (uma matriz linhas x atributos) e o valor
    base (a previsao media, ponto de partida do waterfall).

    Para o classificador (Trilha B), passar classe=1 para pegar a classe
    "baixa geracao". O `background` e uma amostra de referencia usada pelo SHAP.

    Obs.: a versao atual do SHAP ja aceita HistGradientBoosting e RandomForest
    direto no TreeExplainer, entao nao precisamos trocar o modelo do M3."""
    if background is not None:
        explicador = shap.TreeExplainer(
            modelo, data=background, feature_perturbation='interventional')
    else:
        explicador = shap.TreeExplainer(modelo)

    resultado = explicador(X, check_additivity=False)
    valores = resultado.values
    base = resultado.base_values

    # Para classificador, os valores vem com uma dimensao a mais (as classes).
    # Selecionamos a classe pedida (1 = evento de baixa geracao).
    if valores.ndim == 3:
        c = 1 if classe is None else classe
        valores = valores[:, :, c]
        if np.ndim(base) > 1:
            base = base[:, c]

    base = float(np.mean(base))
    return valores, base


def importancia_global(valores, atributos=None):
    """Importancia geral de cada atributo = media do tamanho do efeito SHAP.
    Devolve uma tabela ordenada do mais importante para o menos importante."""
    if atributos is None:
        atributos = config.ATRIBUTOS

    media_efeito = np.abs(valores).mean(axis=0)
    tabela = pd.DataFrame({
        'atributo': atributos,
        'rotulo': [config.rotular(c) for c in atributos],
        'importancia': media_efeito,
    })
    return tabela.sort_values('importancia', ascending=False).reset_index(drop=True)


def contribuicoes_locais(valores_da_linha, linha_X, atributos=None, top_n=10):
    """Prepara os dados do waterfall de UMA previsao.

    Mostra os `top_n` atributos que mais mexeram na previsao e junta todos os
    outros numa linha so ("demais atributos"), para o grafico nao ficar enorme
    sem perder a soma total."""
    if atributos is None:
        atributos = config.ATRIBUTOS

    tabela = pd.DataFrame({
        'atributo': atributos,
        'rotulo': [config.rotular(c) for c in atributos],
        'valor': [linha_X[c] for c in atributos],
        'shap': valores_da_linha,
    })

    # Ordena pelo tamanho do efeito (sem olhar o sinal)
    tabela['tamanho'] = tabela['shap'].abs()
    tabela = tabela.sort_values('tamanho', ascending=False).reset_index(drop=True)

    # Se houver mais atributos que top_n, agrupa o resto numa linha so
    if len(tabela) > top_n:
        resto = tabela.iloc[top_n:]
        tabela = tabela.iloc[:top_n].copy()
        soma_resto = resto['shap'].sum()
        tabela.loc[len(tabela)] = {
            'atributo': '__resto__',
            'rotulo': 'Demais ' + str(len(resto)) + ' atributos (soma)',
            'valor': np.nan,
            'shap': soma_resto,
            'tamanho': abs(soma_resto),
        }

    return tabela.drop(columns='tamanho').reset_index(drop=True)


def texto_explicacao(contribuicoes, base, previsao, unidade='kW'):
    """Monta uma frase explicando a previsao em linguagem simples, citando os
    atributos que mais aumentaram e os que mais diminuiram o valor."""
    d = contribuicoes[contribuicoes['atributo'] != '__resto__']
    sobem = d[d['shap'] > 0].head(2)
    descem = d[d['shap'] < 0].head(2)

    partes = ['A previsao de referencia (media do modelo) e **%.1f %s**.'
              % (base, unidade)]
    if len(sobem) > 0:
        itens = ', '.join('%s (+%.1f)' % (r.rotulo, r.shap)
                          for r in sobem.itertuples())
        partes.append('O que **aumenta** a previsao: %s.' % itens)
    if len(descem) > 0:
        itens = ', '.join('%s (%.1f)' % (r.rotulo, r.shap)
                          for r in descem.itertuples())
        partes.append('O que **reduz** a previsao: %s.' % itens)
    partes.append('Resultado final: **%.1f %s**.' % (previsao, unidade))
    return ' '.join(partes)
