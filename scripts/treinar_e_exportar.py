"""
Treina os dois modelos finais e salva tudo que o dashboard precisa.

Rode UMA vez, com os 4 CSVs dentro da pasta dataset/:

    python scripts/treinar_e_exportar.py

Salva na pasta artifacts/:
    modelo_regressao.joblib       modelo da Trilha A (potencia)
    modelo_classificacao.joblib   modelo da Trilha B (alerta de queda)
    shap_regressao.npz            valores SHAP ja calculados (Trilha A)
    shap_classificacao.npz        valores SHAP ja calculados (Trilha B)
    amostra_teste.parquet         parte do teste que o app mostra
    erro_por_inversor.parquet     erro medio de cada inversor
    metricas.json                 numeros do teste + informacoes gerais

O dashboard nao treina nada: ele so le esses arquivos.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import joblib

# Permite importar a pasta src/ (que esta um nivel acima de scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             classification_report, confusion_matrix)

from src import config
from src.dados import pipeline_completo
from src.modelos import regressor_final, classificador_final, prever_potencia, rmse
from src.explicabilidade import calcular_shap

# Quantas linhas do teste vao para o app. O SHAP e calculado so para elas.
N_AMOSTRA_APP = 4000
N_BACKGROUND = 500

inicio = time.time()

# Cria a pasta artifacts/ se ainda nao existir
os.makedirs(config.PASTA_ARTEFATOS, exist_ok=True)


def caminho_artefato(nome):
    """Monta o caminho completo de um arquivo dentro de artifacts/."""
    return os.path.join(config.PASTA_ARTEFATOS, nome)


print('[1/6] Carregando dados e criando atributos...')
df, treino, teste = pipeline_completo()
X_tr, y_tr = treino[config.ATRIBUTOS], treino[config.ALVO]
X_te, y_te = teste[config.ATRIBUTOS], teste[config.ALVO]
print('      treino=%d | teste=%d' % (len(treino), len(teste)))

# ---------------------------------------------------------- Trilha A (potencia)
print('[2/6] Treinando Trilha A (regressao)...')
reg = regressor_final()
reg.fit(X_tr, y_tr)
pred_te = prever_potencia(reg, X_te)

# Piso: erro de dois metodos simples que o modelo precisa superar
piso_persistencia = rmse(y_te, teste['AC_POWER'].values)
piso_climatologia = rmse(y_te, teste['MEDIANA_SLOT'].values)
PISO_RMSE = min(piso_persistencia, piso_climatologia)

met_reg = {
    'RMSE': rmse(y_te, pred_te),
    'MAE': float(mean_absolute_error(y_te, pred_te)),
    'R2': float(r2_score(y_te, pred_te)),
    'piso_RMSE': float(PISO_RMSE),
    'piso_persistencia': float(piso_persistencia),
    'piso_climatologia': float(piso_climatologia),
}
met_reg['reducao_sobre_piso'] = float(1 - met_reg['RMSE'] / PISO_RMSE)
print('      RMSE=%.1f | MAE=%.1f | R2=%.3f | ganho=%.1f%%' % (
    met_reg['RMSE'], met_reg['MAE'], met_reg['R2'],
    met_reg['reducao_sobre_piso'] * 100))

# ----------------------------------------------------- Trilha B (alerta de queda)
print('[3/6] Treinando Trilha B (classificacao)...')
# A Trilha B so usa horarios diurnos (onde ha geracao)
tr_dia = treino[treino['DIURNO']]
te_dia = teste[teste['DIURNO']]
Xc_tr, yc_tr = tr_dia[config.ATRIBUTOS], tr_dia['EVENTO_BAIXA']
Xc_te, yc_te = te_dia[config.ATRIBUTOS], te_dia['EVENTO_BAIXA']

clf = classificador_final()
clf.fit(Xc_tr, yc_tr)
proba_te = clf.predict_proba(Xc_te)[:, 1]
pred_clf = (proba_te >= 0.5).astype(int)

# Piso da Trilha B: prever queda so olhando se a geracao ja esta baixa agora
piso_evento = (te_dia['AC_POWER'] < config.LIMIAR_BAIXA * te_dia['MEDIANA_SLOT'])
piso_evento = piso_evento.astype(int).values
PISO_AUC = max(0.5, roc_auc_score(yc_te, piso_evento.astype(float)))

matriz = confusion_matrix(yc_te, pred_clf)
vn, fp, fn, vp = matriz.ravel()   # verdadeiros/falsos negativos e positivos

met_clf = {
    'ROC_AUC': float(roc_auc_score(yc_te, proba_te)),
    'piso_AUC': float(PISO_AUC),
    'recall': float(vp / (vp + fn)),
    'precisao': float(vp / (vp + fp)),
    'matriz_confusao': [[int(vn), int(fp)], [int(fn), int(vp)]],
    'prevalencia_teste': float(yc_te.mean()),
    'relatorio': classification_report(
        yc_te, pred_clf,
        target_names=['Geracao normal', 'Baixa geracao'],
        digits=3, output_dict=True),
}
print('      ROC-AUC=%.3f | recall=%.1f%% | precisao=%.1f%%' % (
    met_clf['ROC_AUC'], met_clf['recall'] * 100, met_clf['precisao'] * 100))

# ------------------------------------------ amostra do teste que vai para o app
print('[4/6] Separando a amostra do teste...')
sorteador = np.random.default_rng(config.RANDOM_STATE)
n = min(N_AMOSTRA_APP, len(teste))
indices = np.sort(sorteador.choice(len(teste), n, replace=False))
amostra = teste.iloc[indices].copy()
amostra['PREVISTO'] = pred_te[indices]
amostra['RESIDUO'] = amostra[config.ALVO].values - amostra['PREVISTO'].values

# Coloca a probabilidade de queda (Trilha B) nas linhas diurnas da amostra
proba_por_linha = pd.Series(proba_te, index=te_dia.index)
amostra['PROBA_BAIXA'] = amostra.index.map(proba_por_linha)

# Escolhe as colunas que o app usa (sem repetir)
colunas = (['DATE_TIME', 'PLANTA', 'SOURCE_KEY', 'HORA', 'DIURNO', 'EVENTO_BAIXA',
            config.ALVO, 'PREVISTO', 'RESIDUO', 'PROBA_BAIXA', 'MEDIANA_SLOT']
           + [c for c in config.ATRIBUTOS if c not in ('PLANTA', 'MEDIANA_SLOT')])
colunas = list(dict.fromkeys(colunas))
amostra = amostra[colunas].reset_index(drop=True)
amostra.to_parquet(caminho_artefato('amostra_teste.parquet'), index=False)
print('      amostra: %d linhas' % len(amostra))

# ---------------------------------------------------------------- SHAP
print('[5/6] Calculando o SHAP...')
fundo_reg = X_tr.sample(min(N_BACKGROUND, len(X_tr)), random_state=config.RANDOM_STATE)
X_amostra = amostra[config.ATRIBUTOS]

# SHAP da Trilha A (potencia)
sv_reg, base_reg = calcular_shap(reg, X_amostra, background=fundo_reg)
np.savez_compressed(
    caminho_artefato('shap_regressao.npz'),
    valores=sv_reg.astype(np.float32),
    base=np.array([base_reg], dtype=np.float32),
    atributos=np.array(config.ATRIBUTOS))
print('      Trilha A: %s' % str(sv_reg.shape))

# SHAP da Trilha B (so linhas diurnas)
so_dia = amostra['DIURNO'].values
X_amostra_dia = X_amostra[so_dia]
fundo_clf = Xc_tr.sample(min(N_BACKGROUND, len(Xc_tr)), random_state=config.RANDOM_STATE)
sv_clf, base_clf = calcular_shap(clf, X_amostra_dia, background=fundo_clf, classe=1)
np.savez_compressed(
    caminho_artefato('shap_classificacao.npz'),
    valores=sv_clf.astype(np.float32),
    base=np.array([base_clf], dtype=np.float32),
    atributos=np.array(config.ATRIBUTOS),
    indices=np.where(so_dia)[0])
print('      Trilha B: %s' % str(sv_clf.shape))

# --------------------------------------------- erro por inversor (analise do M3)
print('[6/6] Calculando o erro por inversor e salvando tudo...')
erro = teste[['PLANTA', 'SOURCE_KEY', 'DIURNO']].copy()
erro['residuo'] = y_te.values - pred_te
erro['erro_abs'] = erro['residuo'].abs()
por_inversor = (erro[erro['DIURNO']]
                .groupby(['PLANTA', 'SOURCE_KEY'])
                .agg(MAE=('erro_abs', 'mean'),
                     vies=('residuo', 'mean'),
                     n=('erro_abs', 'size'))
                .sort_values('MAE', ascending=False)
                .reset_index())
por_inversor.to_parquet(caminho_artefato('erro_por_inversor.parquet'), index=False)

# Salva os dois modelos treinados
joblib.dump(reg, caminho_artefato('modelo_regressao.joblib'), compress=3)
joblib.dump(clf, caminho_artefato('modelo_classificacao.joblib'), compress=3)

# Salva os numeros e informacoes gerais num arquivo JSON
info = {
    'gerado_em': pd.Timestamp.now().isoformat(timespec='seconds'),
    'data_corte': str(config.DATA_CORTE.date()),
    'limiar_baixa': config.LIMIAR_BAIXA,
    'n_treino': int(len(treino)),
    'n_teste': int(len(teste)),
    'periodo_treino': [str(treino['DATE_TIME'].min()), str(treino['DATE_TIME'].max())],
    'periodo_teste': [str(teste['DATE_TIME'].min()), str(teste['DATE_TIME'].max())],
    'atributos': config.ATRIBUTOS,
    'capacidade_max_teste': float(y_te.max()),
    'regressao': met_reg,
    'classificacao': met_clf,
}
with open(caminho_artefato('metricas.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print('\nPronto em %.0f segundos. Arquivos salvos em %s' % (
    time.time() - inicio, config.PASTA_ARTEFATOS))
for nome in sorted(os.listdir(config.PASTA_ARTEFATOS)):
    tamanho_kb = os.path.getsize(caminho_artefato(nome)) / 1024
    print('  %-32s %8.1f KB' % (nome, tamanho_kb))
