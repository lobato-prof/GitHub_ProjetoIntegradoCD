# Projeto Integrado em Ciência de Dados, Pós-UNIMONTES

Previsão de geração fotovoltaica a partir do dataset Solar Power Generation (Kaggle). Espera-se com o projeto obter previsão de potência gerada por usinas fotovoltaicas no curto prazo, a partir de dados históricos de geração e leituras de sensores meteorológicos conforme dataset presente no Kaggle.

## Equipe
Gustavo Lobato Campos e Rafael Vinicius Tayette da Nobrega

---

## Problema

A geração fotovoltaica é intermitente e estocástica: depende da irradiância solar, da temperatura dos módulos e de condições meteorológicas que variam ao longo do dia. Prever a potência gerada apoia o despacho de energia, a estabilidade da rede e a manutenção preventiva.

**Tipo de problema:** regressão supervisionada.
**Variável-alvo:** `AC_POWER` (potência AC entregue, em kW).
**Horizonte:** curto prazo (intra-diário a poucos dias).

# Fonte de dados

[**Solar Power Generation Data**](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — Kaggle (autor: Ani Kannal).

- Duas usinas solares na Índia, 34 dias, leituras a cada 15 minutos (~140 mil linhas).
- Geração no nível de inversor (22 inversores por planta): `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`.
- Sensores no nível de planta: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`.
- Acesso público (4 arquivos CSV, ~2 MB).

## Hipótese de solução

A irradiância e a temperatura do módulo, combinadas a atributos temporais (hora do dia, posição no ciclo diário), são suficientes para prever a potência AC com R² ≥ 0,90, usando modelos de gradient boosting (XGBoost / LightGBM) que superam uma linha de base de regressão linear. Uma segunda linha de análise pode envolver o resíduo de potência (potência prevista menos observada), agregado por inversor, pois assim pode-se sinalizar unidades com desempenho abaixo do esperado.

---

## Estrutura de entregas
- `M1_Escopo.pdf` - Ficha de escopo do projeto
- `M2_versao_04_07_2026.ipynb` — Análise Exploratória de Dados (EDA)
