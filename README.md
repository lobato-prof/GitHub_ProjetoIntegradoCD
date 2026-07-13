# Projeto Integrado em Ciência de Dados, Pós-UNIMONTES

Previsão de geração fotovoltaica a partir do dataset Solar Power Generation (Kaggle). Espera-se com o projeto obter previsão de potência gerada por usinas fotovoltaicas no curto prazo, a partir de dados históricos de geração e leituras de sensores meteorológicos conforme dataset presente no Kaggle.

## Equipe
Gustavo Lobato Campos e Rafael Vinicius Tayette da Nobrega

---

## Problema

A geração fotovoltaica é intermitente e estocástica: depende da irradiância solar, da temperatura dos módulos e de condições meteorológicas que variam ao longo do dia. Prever a potência gerada apoia o despacho de energia, a estabilidade da rede e a manutenção preventiva.

**Tipo de problema:** regressão supervisionada.
**Variável-alvo:** `AC_POWER` (potência AC entregue, em kW).
**Horizonte:** curto prazo (conforme modelagem no Módulo 3, previsão 1 hora a frente).

# Fonte de dados

[**Solar Power Generation Data**](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — Kaggle (autor: Ani Kannal).

- Duas usinas solares na Índia, 34 dias, leituras a cada 15 minutos (~140 mil linhas).
- Geração no nível de inversor (22 inversores por planta): `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`.
- Sensores no nível de planta: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`.
- Acesso público (4 arquivos CSV, ~2 MB).

## Hipótese de solução

A irradiância e a temperatura do módulo, combinadas a atributos temporais (hora do dia, posição no ciclo diário) e ao histórico recente de geração, permitem prever a potência AC a 1 hora à frente com modelos de gradient boosting que superam uma linha de base de persistência. Uma segunda linha de análise envolve o resíduo de potência (potência prevista menos observada), agregado por inversor, para sinalizar unidades com desempenho abaixo do esperado. (OBS: Atualizado com o Módulo 3).

---

## Escopo (Módulo 1)

Ficha de escopo do projeto (`M1_Escopo.pdf`): definição do problema, da variável-alvo e da fonte de dados, delimitação do tipo de tarefa (regressão supervisionada) e formulação da hipótese de solução. Estabelece o horizonte de previsão de curto prazo e a segunda linha de análise (resíduo de potência por inversor como sinal de desempenho anômalo), retomada no Módulo 3.

## Análise Exploratória (Módulo 2)

EDA das duas usinas (`M2_versao_04_07_2026.ipynb`, evoluída para `M2_versao_05_07_2026.ipynb` após a entrega formal), estruturada por planta e depois em comparação. Para cada planta, o notebook combina os dados de geração dos inversores com as leituras do sensor meteorológico e cobre:

- **Qualidade dos dados:** identificação de valores ausentes e duplicatas, com imputação por interpolação temporal (linear, ordenada pelo tempo) das variáveis climáticas.
- **Distribuições:** análise das variáveis de geração e clima. Constata-se que `AC_POWER` é fortemente concentrada em zero, refletindo os períodos noturnos sem geração, fato que orienta decisões do Módulo 3 (separação diurno/noturno na avaliação).
- **Relações entre variáveis:** `DC_POWER` e `IRRADIATION` são as mais correlacionadas com `AC_POWER`. A correlação quase perfeita entre `DC_POWER` e `AC_POWER` no mesmo instante é o alerta de vazamento que motiva, no Módulo 3, o deslocamento do alvo para 1h à frente.

A comparação entre as plantas cobre estatísticas descritivas, qualidade, distribuições e correlações, evidenciando a diferença de escala de `DC_POWER` entre elas — tratada por normalização no Módulo 3.

## Modelagem (Módulo 3)

O problema é tratado em duas trilhas acopladas sobre os mesmos atributos e o mesmo split temporal, com horizonte de previsão de 1 hora:

| Trilha | Alvo | Tipo | Métrica primária |
|---|---|---|---|
| **A** | `AC_POWER(t+1h)` — potência, kW | Regressão | RMSE |
| **B** | `EVENTO_BAIXA(t+1h)` — geração abaixo de 50% da mediana histórica do horário | Classificação | ROC-AUC |

A Trilha B é a discretização decisória da Trilha A ("haverá déficit de geração a ponto de exigir despacho de reserva?") e fornece os artefatos de classificação (`classification_report`, matriz de confusão, curva ROC), conforme requisito para o Módulo 3.

### Decisões metodológicas centrais

- **Alvo deslocado 1h à frente**, com todos os atributos medidos em `t` ou antes, para evitar vazamento (`DC_POWER(t)` e `AC_POWER(t)` têm correlação ≈ 1).
- **Deslocamentos por junção temporal** (`merge` sobre `DATE_TIME + Δ`), não por `shift()`, para respeitar as lacunas da série.
- **Split temporal** (treino até 09/06/2020, teste 10–17/06/2020), nunca aleatório, por se tratar de série temporal autocorrelacionada.
- **Validação cruzada temporal** (`TimeSeriesSplit`, 5 folds): cada fold treina no passado e valida no futuro imediato.
- **Rótulo do evento definido apenas com estatísticas de treino** (mediana por planta e horário), sem vazamento do conjunto de teste.

### Resultados

**Trilha A - regressão.** O modelo final é o HistGradientBoosting com hiperparâmetros ajustados por busca aleatória, selecionado por desempenho de CV e desempate por custo/simplicidade. No conjunto de teste: **RMSE 146,8 kW, MAE 73,7 kW, R² 0,797**, uma redução de **22,6%** sobre a linha de base de persistência. A hipótese inicial de R² ≥ 0,90 não se confirmou, que os 34 dias de uma única estação limitam o teto de desempenho, mas o modelo agrega valor mensurável sobre a persistência, concentrado no período diurno.

**Trilha B - classificação.** O modelo final (Random Forest) atinge **ROC-AUC 0,914** no teste (contra piso de 0,547), com recall de 73% e precisão de 82% no evento de baixa geração. O contraste entre as trilhas é o achado central: prever o *valor exato* em kW é difícil (ganho modesto), mas prever *se haverá déficit* é altamente viável e é esta a decisão operacionalmente relevante.

**Análise de resíduo por inversor.** Confirmando a segunda linha de análise prevista, os inversores com maior erro sistemático concentram-se todos na Planta 2, com viés negativo (o modelo os superestima). Como a identidade do inversor foi excluída dos atributos, esse resíduo sistemático sinaliza unidades subgerando, possível degradação de hardware (sujeira no módulo, string desconectada, derating térmico). Este é o gancho para um detector de anomalia (potencialmente a ser trabalhado no Módulo 4).

### Limitações principais

O horizonte útil está travado em ~1h porque o modelo usa clima *observado*, não *previsto* (estender ao despacho intradiário exigiria previsão numérica de tempo). Os resultados valem para a janela de 34 dias de uma única estação, sem validação sazonal. O limiar de 50% que define o evento é escolha do analista e deve, antes de qualquer uso operacional, derivar do custo real de despacho. A lista completa está na Seção 5 do notebook.

---

## Estrutura de entregas
- `M1_Escopo.pdf` - Ficha de escopo do projeto
- `M2_versao_04_07_2026.ipynb` - Análise Exploratória de Dados (EDA)
- `M3_versao_13_07_2026.ipynb` - Modelagem, avaliação e análise de erros

---
