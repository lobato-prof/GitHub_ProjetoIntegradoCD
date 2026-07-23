"""Dashboard - Previsão de Geração Fotovoltaica (Módulo 4).

Produto de dados sobre os modelos treinados no Módulo 3.
Não treina nada: carrega os artefatos gerados por scripts/treinar_e_exportar.py.

Executar:  streamlit run app/app.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Permite importar a pasta src/ (que esta um nivel acima de app/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ATRIBUTOS, ALVO, PASTA_ARTEFATOS, rotular
from src.explicabilidade import (importancia_global, contribuicoes_locais,
                                 texto_explicacao)

st.set_page_config(page_title='Previsão de Geração Solar',
                   page_icon='sun', layout='wide',
                   initial_sidebar_state='expanded')

# Paleta: derivada do domínio (espectro solar do amanhecer ao meio-dia),
# não de um tema padrão. Índigo noturno -> âmbar -> branco de meio-dia.
COR_NOITE = '#1b2a4a'
COR_AMBAR = '#e8a33d'
COR_TERRA = '#b5502a'
COR_VERDE = '#2b7a78'
COR_CINZA = '#6b7280'

st.markdown("""
<style>
  .bloco-metrica {
      background: linear-gradient(135deg, #1b2a4a 0%, #2d4373 100%);
      border-radius: 10px; padding: 14px 18px; color: #fff;
      border-left: 4px solid #e8a33d;
  }
  .bloco-metrica .rotulo { font-size: 0.75rem; opacity: 0.75;
      text-transform: uppercase; letter-spacing: 0.06em; }
  .bloco-metrica .valor { font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
  .bloco-metrica .nota { font-size: 0.72rem; opacity: 0.7; }
  .ato { border-left: 3px solid #e8a33d; padding-left: 14px; margin: 10px 0 18px 0; }
  .ato h4 { margin: 0 0 4px 0; color: #1b2a4a; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ carga
def caminho(nome):
    """Caminho completo de um arquivo dentro da pasta artifacts/."""
    return os.path.join(PASTA_ARTEFATOS, nome)


@st.cache_data(show_spinner='Carregando os dados do modelo...')
def carregar():
    # Verifica se todos os arquivos necessarios existem
    necessarios = ['metricas.json', 'amostra_teste.parquet',
                   'shap_regressao.npz', 'shap_classificacao.npz',
                   'erro_por_inversor.parquet']
    faltando = [n for n in necessarios if not os.path.exists(caminho(n))]
    if faltando:
        return None, None, None, None, None, faltando

    with open(caminho('metricas.json'), encoding='utf-8') as f:
        meta = json.load(f)
    amostra = pd.read_parquet(caminho('amostra_teste.parquet'))
    zr = np.load(caminho('shap_regressao.npz'), allow_pickle=True)
    zc = np.load(caminho('shap_classificacao.npz'), allow_pickle=True)
    shap_reg = {'valores': zr['valores'], 'base': float(zr['base'][0])}
    shap_clf = {'valores': zc['valores'], 'base': float(zc['base'][0]),
                'indices': zc['indices']}
    inv = pd.read_parquet(caminho('erro_por_inversor.parquet'))
    return meta, amostra, shap_reg, shap_clf, inv, []


meta, amostra, shap_reg, shap_clf, err_inv, faltando = carregar()

if faltando:
    st.error('**Artefatos do modelo não encontrados.**')
    st.markdown(
        'O dashboard não treina modelos: ele carrega o que o Módulo 3 produziu. '
        f'Arquivos ausentes em `artifacts/`: `{"`, `".join(faltando)}`.\n\n'
        'Para gerar, com os dados em `dataset/`:\n'
        '```bash\npython scripts/treinar_e_exportar.py\n```')
    st.stop()

MET_R = meta['regressao']
MET_C = meta['classificacao']
CAP_MAX = meta['capacidade_max_teste']


# ------------------------------------------------------------ componentes
def cartao(rotulo, valor, nota=''):
    st.markdown(f"""<div class="bloco-metrica">
        <div class="rotulo">{rotulo}</div>
        <div class="valor">{valor}</div>
        <div class="nota">{nota}</div></div>""", unsafe_allow_html=True)


def limitacoes(chave: str):
    """Limitações visíveis com st.warning(), sem rolar a tela.

    Cada visão exibe apenas as limitações que se aplicam a ela, no topo,
    dentro da primeira dobra. Origem: Seção 5 do notebook M3.
    """
    textos = {
        'panorama': (
            'Estes números valem para **7 dias de junho de 2020** (34 dias de coleta, '
            'uma única estação). Não há validação sazonal: não use como estimativa de '
            'desempenho anual.'),
        'predicao': (
            'A previsão vale para **1 hora à frente** e usa clima **medido agora**, não '
            'clima previsto. Não serve para planejar o dia seguinte.'),
        'global': (
            'A importância mostra **o que o modelo usa**, não o que **causa** a geração. '
            'Atributos correlacionados dividem crédito entre si.'),
        'inversores': (
            'O resíduo por inversor é **sugestivo, não conclusivo**: há um único sensor '
            'de clima por usina, então "nuvem sobre o inversor" e "inversor degradado" '
            'são indistinguíveis nos dados.'),
    }
    st.warning(textos[chave], icon=':material/warning:')


def waterfall(contrib: pd.DataFrame, base: float, final: float, unidade: str):
    d = contrib.iloc[::-1]
    fig = go.Figure(go.Waterfall(
        orientation='h',
        measure=['relative'] * len(d),
        y=[r if len(r) < 42 else r[:39] + '...' for r in d['rotulo']],
        x=d['shap'],
        base=base,
        connector={'line': {'color': '#cbd5e1', 'width': 1}},
        increasing={'marker': {'color': COR_AMBAR}},
        decreasing={'marker': {'color': COR_NOITE}},
        text=[f'{v:+.1f}' for v in d['shap']],
        textposition='outside',
        hovertemplate='%{y}<br>contribuição: %{x:+.2f} ' + unidade + '<extra></extra>',
    ))
    fig.add_vline(x=base, line_dash='dot', line_color=COR_CINZA,
                  annotation_text=f'referência {base:.1f}',
                  annotation_position='top')
    fig.update_layout(
        height=430, margin=dict(l=0, r=30, t=34, b=0),
        xaxis_title=f'Contribuição para a previsão ({unidade})',
        showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'Da referência ({base:.1f}) até a previsão ({final:.1f})',
                   font=dict(size=13, color=COR_CINZA)))
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('### Previsão de geração solar')
    st.caption('Projeto Integrado em Ciência de Dados · Pós-UNIMONTES')

    visao = st.radio(
        'Visão',
        ['Panorama geral', 'Previsão individual', 'O que o modelo aprendeu',
         'Desempenho por inversor', 'A história do projeto'],
        label_visibility='collapsed')

    st.divider()
    st.caption(f"**Período analisado**  \n{meta['periodo_teste'][0][:10]} a "
               f"{meta['periodo_teste'][1][:10]}")
    st.caption(f"**Usinas**  \n2 usinas · 44 inversores · leitura a cada 15 min")
    st.caption(f"**Modelo treinado em**  \n{meta['gerado_em'][:10]}")
    st.divider()
    st.caption('Dados: Kaggle — Solar Power Generation Data (Ani Kannal). '
               'Uso educacional.')


# ======================================================== VISÃO 1: PANORAMA
if visao == 'Panorama geral':
    st.title('Panorama geral')
    st.markdown('Como o modelo se comporta nos 7 dias que ele **nunca viu** durante o treino.')
    limitacoes('panorama')

    c = st.columns(4)
    with c[0]:
        cartao('Erro típico da previsão', f"{MET_R['MAE']:.0f} kW",
               f"{MET_R['MAE'] / CAP_MAX:.1%} da capacidade máxima")
    with c[1]:
        cartao('Ganho sobre o método simples', f"{MET_R['reducao_sobre_piso']:.0%}",
               f"contra repetir a potência atual")
    with c[2]:
        cartao('Variação explicada', f"{MET_R['R2']:.1%}",
               'R² no conjunto de teste')
    with c[3]:
        cartao('Acerto do alerta de queda', f"{MET_C['ROC_AUC']:.3f}",
               f"ROC-AUC · piso {MET_C['piso_AUC']:.3f}")

    st.markdown('')
    esq, dir_ = st.columns([2, 1])

    with esq:
        st.markdown('#### Previsto e realizado ao longo dos dias')
        usina = st.selectbox('Usina', [1, 2], format_func=lambda x: f'Usina {x}',
                             key='u_pan')
        d = amostra[amostra['PLANTA'] == usina].copy()
        d['dia'] = pd.to_datetime(d['DATE_TIME']).dt.date
        serie = (d.groupby([pd.to_datetime(d['DATE_TIME']).dt.floor('h')])
                 .agg(real=(ALVO, 'mean'), previsto=('PREVISTO', 'mean')).reset_index())
        serie.columns = ['instante', 'real', 'previsto']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie['instante'], y=serie['real'],
                                 name='Geração realizada', mode='lines',
                                 line=dict(color=COR_NOITE, width=2)))
        fig.add_trace(go.Scatter(x=serie['instante'], y=serie['previsto'],
                                 name='Previsão do modelo', mode='lines',
                                 line=dict(color=COR_AMBAR, width=2, dash='dot')))
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=6, b=0),
                          yaxis_title='Potência média por inversor (kW)',
                          xaxis_title='', plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation='h', y=1.12, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Média horária entre os inversores da usina. As linhas se separam '
                   'nos dias de nuvem, aqui que o modelo erra.')

    with dir_:
        st.markdown('#### Onde o erro se concentra')
        d = amostra.copy()
        d['periodo'] = np.where(d['DIURNO'], 'Dia', 'Noite')
        agg = (d.assign(erro=d['RESIDUO'].abs())
               .groupby('periodo')['erro'].mean().reset_index())
        fig = px.bar(agg, x='periodo', y='erro', color='periodo',
                     color_discrete_map={'Dia': COR_AMBAR, 'Noite': COR_NOITE})
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=6, b=0),
                          showlegend=False, yaxis_title='Erro médio (kW)',
                          xaxis_title='', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('À noite não há geração e acertar é trivial. O problema real, '
                   'e todo o erro, está no período diurno.')

    st.divider()
    st.markdown('#### O alerta de queda de geração')
    a, b = st.columns([1, 1])
    with a:
        cm = MET_C['matriz_confusao']
        z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
        fig = px.imshow(z, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x='O modelo previu', y='O que aconteceu'),
                        x=['Geração normal', 'Queda de geração'],
                        y=['Geração normal', 'Queda de geração'])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=6, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown(f"""
De cada 100 quedas de geração que realmente ocorreram, o modelo antecipou
**{MET_C['recall'] * 100:.0f}**. De cada 100 alertas que ele disparou,
**{MET_C['precisao'] * 100:.0f}** eram quedas de verdade.

Os dois erros custam coisas diferentes: deixar de avisar significa déficit
sem reserva despachada; avisar sem necessidade significa reserva parada.
Como o primeiro custa mais, na operação o gatilho do alerta deveria ser mais
sensível que o padrão de 50% usado aqui.
        """)
        st.info(f"O alerta separa quedas de não-quedas com ROC-AUC de "
                f"**{MET_C['ROC_AUC']:.3f}**, contra **{MET_C['piso_AUC']:.3f}** "
                f"do método ingênuo.", icon=':material/insights:')


# =============================================== VISÃO 2: PREVISÃO INDIVIDUAL
elif visao == 'Previsão individual':
    st.title('Previsão individual')
    st.markdown('Escolha um instante e veja **por que** o modelo previu aquele valor.')
    limitacoes('predicao')

    f = st.columns([1, 1, 2])
    with f[0]:
        usina = st.selectbox('Usina', [1, 2], format_func=lambda x: f'Usina {x}')
    with f[1]:
        so_dia = st.checkbox('Apenas horário diurno', value=True)
    sub = amostra[amostra['PLANTA'] == usina]
    if so_dia:
        sub = sub[sub['DIURNO']]
    with f[2]:
        inversor = st.selectbox('Inversor', sorted(sub['SOURCE_KEY'].unique()))
    sub = sub[sub['SOURCE_KEY'] == inversor].sort_values('DATE_TIME')

    if len(sub) == 0:
        st.warning('Nenhum registro com esses filtros. Ajuste a seleção acima.',
                   icon=':material/filter_alt_off:')
        st.stop()

    rotulos_ts = [pd.to_datetime(t).strftime('%d/%m %H:%M') for t in sub['DATE_TIME']]
    escolha = st.select_slider('Instante', options=range(len(sub)),
                               format_func=lambda i: rotulos_ts[i],
                               value=len(sub) // 2)
    linha = sub.iloc[escolha]
    pos = sub.index[escolha]

    m = st.columns(4)
    m[0].metric('Previsão para daqui a 1 hora', f"{linha['PREVISTO']:.0f} kW")
    m[1].metric('O que de fato ocorreu', f"{linha[ALVO]:.0f} kW",
                f"{linha[ALVO] - linha['PREVISTO']:+.0f} kW de diferença")
    m[2].metric('Geração típica deste horário', f"{linha['MEDIANA_SLOT']:.0f} kW")
    if pd.notna(linha['PROBA_BAIXA']):
        m[3].metric('Chance de queda de geração', f"{linha['PROBA_BAIXA']:.0%}")
    else:
        m[3].metric('Chance de queda de geração', '—', 'período noturno')

    st.divider()
    aba_a, aba_b = st.tabs(['Por que este valor de potência?',
                            'Por que este nível de alerta?'])

    with aba_a:
        vals = shap_reg['valores'][pos]
        contrib = contribuicoes_locais(vals, linha[ATRIBUTOS], top_n=10)
        e, d = st.columns([3, 2])
        with e:
            st.plotly_chart(waterfall(contrib, shap_reg['base'],
                                      linha['PREVISTO'], 'kW'),
                            use_container_width=True)
        with d:
            st.markdown('##### Em palavras')
            st.markdown(texto_explicacao(contrib, shap_reg['base'],
                                         linha['PREVISTO'], 'kW'))
            st.caption('Cada barra é quanto aquele fator empurrou a previsão para '
                       'cima (âmbar) ou para baixo (azul), partindo da média do modelo.')
            with st.expander('Valores lidos neste instante'):
                mostrar = contrib[contrib['atributo'] != '__resto__'][['rotulo', 'valor']]
                st.dataframe(mostrar.rename(columns={'rotulo': 'Fator',
                                                     'valor': 'Valor medido'}),
                             hide_index=True, use_container_width=True)

    with aba_b:
        if pd.isna(linha['PROBA_BAIXA']):
            st.info('O alerta de queda só opera em horário diurno: prever que não '
                    'haverá geração às 2h da manhã não é uma tarefa útil. '
                    'Escolha um instante do período diurno.',
                    icon=':material/dark_mode:')
        else:
            onde = np.where(shap_clf['indices'] == pos)[0]
            if len(onde) == 0:
                st.info('Explicação não pré-computada para este instante.',
                        icon=':material/info:')
            else:
                vals = shap_clf['valores'][onde[0]]
                contrib = contribuicoes_locais(vals, linha[ATRIBUTOS], top_n=10)
                e, d = st.columns([3, 2])
                with e:
                    st.plotly_chart(
                        waterfall(contrib, shap_clf['base'],
                                  shap_clf['base'] + vals.sum(), 'log-odds'),
                        use_container_width=True)
                with d:
                    st.markdown('##### Em palavras')
                    st.markdown(
                        f"O modelo estima **{linha['PROBA_BAIXA']:.0%}** de chance de "
                        f"a geração cair abaixo de metade do normal para este horário.")
                    sobe = contrib[(contrib['shap'] > 0) &
                                   (contrib['atributo'] != '__resto__')].head(2)
                    if len(sobe):
                        st.markdown('**Aumenta o risco:** ' + ', '.join(
                            f'{r.rotulo}' for r in sobe.itertuples()))
                    desce = contrib[(contrib['shap'] < 0) &
                                    (contrib['atributo'] != '__resto__')].head(2)
                    if len(desce):
                        st.markdown('**Reduz o risco:** ' + ', '.join(
                            f'{r.rotulo}' for r in desce.itertuples()))
                    st.caption('A escala é log-odds: valores positivos empurram para '
                               '"vai haver queda", negativos para "geração normal".')


# ============================================= VISÃO 3: INTERPRETABILIDADE
elif visao == 'O que o modelo aprendeu':
    st.title('O que o modelo aprendeu')
    st.markdown('Quais fatores mais pesam nas previsões, considerando **todos** os '
                'instantes analisados.')
    limitacoes('global')

    trilha = st.radio('Modelo', ['Previsão de potência (kW)',
                                 'Alerta de queda de geração'],
                      horizontal=True, label_visibility='collapsed')

    if trilha.startswith('Previsão'):
        vals, unidade = shap_reg['valores'], 'kW'
    else:
        vals, unidade = shap_clf['valores'], 'log-odds'

    imp = importancia_global(vals)
    e, d = st.columns([3, 2])

    with e:
        st.markdown('#### Peso de cada fator')
        top = imp.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top['importancia'], y=top['rotulo'], orientation='h',
            marker=dict(color=top['importancia'], colorscale=[[0, COR_NOITE],
                                                              [1, COR_AMBAR]]),
            hovertemplate='%{y}<br>peso médio: %{x:.2f}<extra></extra>'))
        fig.update_layout(height=440, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title=f'Influência média na previsão ({unidade})',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with d:
        st.markdown('#### Leitura')
        t3 = imp.head(3)['rotulo'].tolist()
        st.markdown(f"""
Os três fatores de maior peso são **{t3[0]}**, **{t3[1]}** e **{t3[2]}**.

O padrão é coerente com o domínio: a geração solar é governada pela **posição
no ciclo do dia** e pela **inércia de curto prazo** — em uma hora, a usina
raramente muda de regime, a menos que entre nuvem.

Isso também explica o limite do modelo. Ele aprendeu bem o comportamento
*típico* de cada horário, e é justamente por isso que erra nas **transições
de nebulosidade**, quando o presente deixa de ser um bom guia do futuro.
        """)
        with st.expander('Tabela completa'):
            st.dataframe(
                imp[['rotulo', 'importancia']].rename(
                    columns={'rotulo': 'Fator', 'importancia': f'Peso ({unidade})'}),
                hide_index=True, use_container_width=True, height=260)

    st.divider()
    st.markdown('#### Como um fator específico move a previsão')
    escolha = st.selectbox('Fator', imp['atributo'].tolist(),
                           format_func=rotular)
    j = ATRIBUTOS.index(escolha)
    if trilha.startswith('Previsão'):
        base_x = amostra[escolha].values
    else:
        base_x = amostra.iloc[shap_clf['indices']][escolha].values

    dd = pd.DataFrame({'valor': base_x, 'efeito': vals[:, j]})
    fig = px.scatter(dd, x='valor', y='efeito', opacity=0.35,
                     color_discrete_sequence=[COR_VERDE])
    fig.add_hline(y=0, line_dash='dot', line_color=COR_CINZA)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=6, b=0),
                      xaxis_title=rotular(escolha),
                      yaxis_title=f'Efeito na previsão ({unidade})',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Cada ponto é um instante analisado. Acima da linha, o fator empurrou '
               'a previsão para cima naquele instante; abaixo, para baixo.')


# ============================================ VISÃO 4: ERRO POR INVERSOR
elif visao == 'Desempenho por inversor':
    st.title('Desempenho por inversor')
    st.markdown('O erro do modelo, quando é **sistemático em uma unidade**, deixa de '
                'ser ruído e vira sinal de manutenção.')
    limitacoes('inversores')

    d = err_inv.copy()
    d['usina'] = 'Usina ' + d['PLANTA'].astype(str)
    d['inversor'] = d['SOURCE_KEY'].str.slice(0, 10)

    e, dd = st.columns([3, 2])
    with e:
        top = d.head(14).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top['MAE'], y=top['inversor'], orientation='h',
            marker=dict(color=np.where(top['PLANTA'] == 2, COR_TERRA, COR_VERDE)),
            hovertemplate='%{y}<br>erro médio: %{x:.1f} kW<extra></extra>'))
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title='Erro médio do modelo no período diurno (kW)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'Vermelho: Usina 2 · Verde: Usina 1')

    with dd:
        st.markdown('#### Leitura')
        piores = d.head(8)
        n2 = int((piores['PLANTA'] == 2).sum())
        st.markdown(f"""
Dos 8 inversores com maior erro, **{n2} estão na Usina 2**.

O modelo **não sabe** qual inversor está prevendo: a identidade da unidade foi
deliberadamente retirada dos dados de entrada. Então, se ele erra sempre na
mesma direção para as mesmas unidades, a explicação não está no modelo — está
no equipamento.

Viés negativo significa que o modelo **espera mais do que a unidade entrega**:
o comportamento esperado de um painel sujo, uma string desconectada ou perda
por temperatura.
        """)
        st.info('Este é o gancho para um detector de anomalia — a segunda linha de '
                'análise prevista desde o Módulo 1.', icon=':material/build:')

    st.markdown('#### Todos os inversores')
    tab = d[['usina', 'SOURCE_KEY', 'MAE', 'vies', 'n']].rename(columns={
        'usina': 'Usina', 'SOURCE_KEY': 'Inversor', 'MAE': 'Erro médio (kW)',
        'vies': 'Viés (kW)', 'n': 'Leituras'})
    st.dataframe(tab.round(1), hide_index=True, use_container_width=True, height=260)


# ================================================ VISÃO 5: NARRATIVA 4 ATOS
elif visao == 'A história do projeto':
    st.title('A história do projeto')
    st.caption('Do problema de operação ao que o modelo entrega e ao que ele não entrega.')

    a1, a2, a3, a4 = st.tabs(['1 · O problema', '2 · Os dados',
                              '3 · O modelo', '4 · Os limites'])

    with a1:
        st.markdown("""
<div class="ato"><h4>A operação enxerga a queda depois que ela aconteceu</h4></div>
        """, unsafe_allow_html=True)
        e, d = st.columns([3, 2])
        with e:
            st.markdown("""
A geração solar é intermitente por natureza: depende da irradiância, da
temperatura dos painéis e do tempo, que muda ao longo do dia.

Quem sente isso é a equipe de operação e manutenção. Sem previsão de curto
prazo, ela trabalha de forma **reativa**, a queda de produção só aparece
depois de ter ocorrido, e não há referência objetiva para dizer qual inversor
está gerando abaixo do que deveria.

O custo é duplo. Para a rede, o desvio entre a energia programada e a
entregue pode gerar penalidade. Para a usina, um inversor com defeito passa
semanas subgerando antes de alguém notar.

**A pergunta do projeto:** dá para antecipar a geração da próxima hora e, ainda, 
apontar quais unidades estão fora do padrão?
            """)
        with d:
            st.info("""
**Quem usa este painel**

**Operação:** antecipar déficit e programar reserva.

**Manutenção:** priorizar a visita aos inversores fora do padrão.

**Gestão:** entender o alcance e o limite do que foi construído.
            """, icon=':material/groups:')

    with a2:
        st.markdown("""
<div class="ato"><h4>34 dias, duas usinas, 44 inversores</h4></div>
        """, unsafe_allow_html=True)
        e, d = st.columns([3, 2])
        with e:
            st.markdown(f"""
Os dados são públicos (Kaggle) e cobrem duas usinas na Índia entre maio e
junho de 2020, com leitura a cada 15 minutos: cerca de **140 mil registros**.

Cada inversor reporta potência CC, potência CA e energia acumulada. Cada usina
tem **um** sensor de clima, medindo irradiância e temperatura do ar e do painel.

A análise exploratória (Módulo 2) revelou o achado que definiu todo o resto:
**potência CC e CA no mesmo instante são quase a mesma variável** (correlação
≈ 1). Um modelo que usasse uma para prever a outra teria R² de 0,999 e não
preveria nada, apenas reaprenderia a eficiência do inversor.

A saída foi ajustar a pergunta: em vez de "qual a potência agora", **"qual será a
potência daqui a uma hora"**. Aí a potência CC de agora vira informação
legítima sobre o futuro.
            """)
        with d:
            st.info(f"""
**A base em números**

Treino: **{meta['n_treino']:,}** leituras
({meta['periodo_treino'][0][:10]} a {meta['periodo_treino'][1][:10]})

Teste: **{meta['n_teste']:,}** leituras
({meta['periodo_teste'][0][:10]} a {meta['periodo_teste'][1][:10]})

O corte é **por data**, nunca aleatório: o modelo treina no passado e é
avaliado no futuro, como aconteceria na operação.
            """.replace(',', '.'), icon=':material/database:')
            st.caption('Fonte: Kaggle — Solar Power Generation Data (Ani Kannal).')

    with a3:
        st.markdown("""
<div class="ato"><h4>Prever quantos kW é difícil. Prever se vai faltar, não.</h4></div>
        """, unsafe_allow_html=True)
        e, d = st.columns([3, 2])
        with e:
            st.markdown(f"""
Foram testadas cinco abordagens para a previsão de potência e três para o
alerta de queda, todas validadas com cortes temporais sucessivos. Venceram
um modelo de **árvores com gradient boosting** e uma **floresta aleatória**.

O resultado tem duas metades, e a segunda é a interessante.

**Prever o valor exato** rende erro médio de **{MET_R['MAE']:.0f} kW**
({MET_R['MAE'] / CAP_MAX:.0%} da capacidade) e ganho de
**{MET_R['reducao_sobre_piso']:.0%}** sobre simplesmente repetir a potência
atual. Ganho real, mas modesto e abaixo da meta de R² ≥ 0,90 que o projeto
tinha fixado no Módulo 1.

**Prever se haverá queda** salta de **{MET_C['piso_AUC']:.3f}** para
**{MET_C['ROC_AUC']:.3f}** de ROC-AUC.

E é essa a decisão que a operação precisa tomar. Ninguém despacha reserva
porque a previsão é de 812 kW; despacha porque vai faltar energia. A pergunta
mais fácil era também a mais útil.
            """)
        with d:
            st.info(f"""
**A hipótese que não se confirmou**

O Módulo 1 previa R² ≥ 0,90. O resultado foi **{MET_R['R2']:.2f}**.

Com 34 dias de uma única estação, esse era o teto. Reportar a hipótese
refutada é parte do trabalho, não uma falha dele.
            """, icon=':material/science:')

    with a4:
        st.markdown("""
<div class="ato"><h4>O que este modelo não faz</h4></div>
        """, unsafe_allow_html=True)
        st.markdown('Cada limite abaixo é uma condição do treino, seguida do que ela impede.')

        c1, c2 = st.columns(2)
        with c1:
            st.warning("""
**O horizonte trava em 1 hora.**
O modelo usa clima **medido**, não **previsto**. Às 10h ele entrega a geração
das 11h. Não entrega a das 16h, isso exigiria previsão meteorológica como
entrada, que não existe nesta base. É o limite mais sério para uso real.
            """, icon=':material/schedule:')
            st.warning("""
**34 dias, uma estação.**
Maio–junho de 2020, pré-monção. Nada aqui sustenta afirmação sobre desempenho
anual. Não há validação em outro ano nem estimativa de deriva sazonal.
            """, icon=':material/calendar_month:')
            st.warning("""
**Um sensor de clima para 22 inversores.**
Sombra parcial, nuvem sobre um setor e sujeira localizada são invisíveis.
O modelo não distingue "nuvem sobre o inversor X" de "inversor X degradado".
            """, icon=':material/sensors:')
        with c2:
            st.warning("""
**O limiar de 50% é escolha do analista.**
"Queda de geração" foi definida como gerar menos da metade do normal para o
horário. Com 30% ou 70%, todas as métricas do alerta mudam. Antes de operar,
esse número deve vir do custo real de despacho de reserva, não do analista.
            """, icon=':material/tune:')
            st.warning("""
**Sete dias de teste, uma única partição.**
O intervalo de confiança das métricas é desconhecido. Sete dias podem ter sido
atipicamente limpos ou nublados.
            """, icon=':material/query_stats:')
            st.warning("""
**Faltam dados de manutenção.**
Limpeza de painel, indisponibilidade programada e curtailment não estão na
base. Parte do que foi chamado de "erro do modelo" pode ser evento operacional
que o operador já conhecia.
            """, icon=':material/engineering:')

        st.divider()
        st.markdown("""
**Para onde isso vai.** O erro sistemático concentrado nos inversores da Usina 2
é o achado mais promissor: com dados de manutenção para confirmar, vira um
detector de anomalia de hardware. Estender o horizonte para o despacho
intradiário (4–6h) exige acoplar previsão meteorológica e revalidar tudo.
        """)
        st.caption('Lista completa e detalhada: Seção 5 do notebook do Módulo 3.')
