"""Exercita todas as visões do dashboard e reporta exceções."""
import sys
import os
from streamlit.testing.v1 import AppTest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISOES = ['Panorama geral', 'Previsão individual', 'O que o modelo aprendeu',
          'Desempenho por inversor', 'A história do projeto']

falhas = 0
for v in VISOES:
    at = AppTest.from_file(os.path.join(RAIZ, 'app', 'app.py'), default_timeout=180)
    at.run()
    if at.exception:
        print(f'FALHA [carga inicial] {v}: {at.exception[0].message}')
        falhas += 1
        continue
    at.sidebar.radio[0].set_value(v).run()
    if at.exception:
        print(f'FALHA [{v}]: {at.exception[0].value}')
        falhas += 1
    else:
        n_warn = len(at.warning)
        n_tabs = len(at.tabs)
        print(f'OK   [{v}] warnings={n_warn} tabs={n_tabs} '
              f'selectbox={len(at.selectbox)} plotly=?')

    # exercitar tabs/filtros da visão de predição
    if v == 'Previsão individual' and not at.exception:
        for u in [1, 2]:
            at.selectbox[0].set_value(u).run()
            if at.exception:
                print(f'  FALHA usina={u}: {at.exception[0].value}')
                falhas += 1
        at.checkbox[0].set_value(False).run()
        if at.exception:
            print(f'  FALHA sem filtro diurno: {at.exception[0].value}')
            falhas += 1
        else:
            print('  OK  filtro diurno desligado (instantes noturnos acessíveis)')

    if v == 'O que o modelo aprendeu' and not at.exception:
        alvo = next((r for r in at.radio
                     if any('Alerta' in str(o) for o in r.options)), None)
        if alvo is None:
            print('  FALHA: radio de trilha não encontrado')
            falhas += 1
        else:
            alvo.set_value('Alerta de queda de geração').run()
            if at.exception:
                print(f'  FALHA trilha B: {at.exception[0].value}')
                falhas += 1
            else:
                print('  OK  trilha B (classificação)')
                for fator in ['IRRADIATION', 'AC_POWER', 'MEDIANA_SLOT']:
                    sb = at.selectbox[-1]
                    sb.set_value(fator).run()
                    if at.exception:
                        print(f'  FALHA fator={fator}: {at.exception[0].value}')
                        falhas += 1
                        break
                else:
                    print('  OK  gráfico de dependência (3 fatores)')

print(f'\n{"TODOS OS TESTES PASSARAM" if falhas == 0 else f"{falhas} FALHA(S)"}')
sys.exit(1 if falhas else 0)
