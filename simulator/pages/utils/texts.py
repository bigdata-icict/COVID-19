INTRODUCTION = '''
# Simulador COVID-19

Ciência de Dados aplicada à pandemia do novo coronavírus

---
'''

ABOUT = '''
Este projeto é uma força tarefa das comunidades científica e tecnológica a fim de criar modelos de previsão de infectados pelo COVID-19 - e outras métricas relacionadas -, para o Brasil. O projeto é público e pode ser usado por todos.

Acesse [este link](https://github.com/bigdata-icict/COVID-19) para informações detalhadas e instruções sobre como contribuir.
'''

PARAMETER_SELECTION='''
# Seleção de parâmetros
Para simular outros cenários, altere um parâmetro e tecle **Enter**. O novo resultado será calculado e apresentado automaticamente.
#### Parâmetros de Localidade
'''

MODEL_INTRO='''
---
## Simuladores
### Previsão de expostos e infectados
O gráfico abaixo mostra o resultado da simulação da evolução de pacientes expostos e infectados para os parâmetros selecionados. Mais informações sobre este modelo estão disponíveis [aqui](https://github.com/bigdata-icict/COVID-19#seir-bayes).

**(!) Importante**: Os resultados apresentados são *preliminares* e estão em fase de validação.
'''
PARAMS_LEITOS='''
#### Parâmetros de Leitos

'''
DESC_PARAMS_LEITOS='''
### Parâmetros da previsão de leitos
* A Taxa de internação  é utilizada para a simulação do comportamento das curvas de demanda de leitos.
'''

DESC_PARAMS_DEATHS = '''
### Parâmetros da previsão de óbitos
* O valor da taxa pode ser alterado com o parâmetro Taxa de letalidade (em %, proporção dos casos que resultam em óbito).
'''

INTRO_MODELO='''
## Introdução ao modelo
### Nota Técnica
[Clique aqui para acessar a nóta técnica do modelo SEIR-Bayes e a estimação do número básico de reprodução](https://github.com/3778/COVID-19/raw/master/nota-tecnica.pdf)
'''

DEATHS_INTRO='''
### Previsão de óbitos
O gráfico abaixo mostra a estimativa de óbitos diários para os parâmetros selecionados. O cálculo é realizado a partir da aplicação de uma taxa de letalidade.

**(!) Importante**: Os resultados apresentados são *preliminares* e estão em fase de validação.
'''

DEATH_DETAIL='''
<details>
    <summary style="color: rgb(38, 39, 48);"><strong>Mostrar metodologia de cálculo</strong></summary>
    <div style="color: rgb(38, 39, 48);">
        <p>A estimativa de óbitos é calculada multiplicando-se o número de casos estimados por uma taxa de letalidade. Como padrão, essa taxa é calculada a partir dos casos e óbitos oficialmente registrados no nível geográfico selecionado.</p>
    </div>
</details>
'''

LEITOS_INTRO='''
### Previsão da demanda de leitos

O gráfico abaixo mostra a estimativa da demanda de leitos acumulada para os parâmetros selecionados. O cálculo é realizado a partir da aplicação de umataxa de internação.
(!) Importante: Os resultados apresentados são preliminares e estão em fase de validação.
'''

LEITOS_DETAIL='''
<details>
    <summary style="color: rgb(38, 39, 48);"><strong>Mostrar metodologia de cálculo</strong></summary>
    <div style="color: rgb(38, 39, 48);">
        <p>A demanda de leitos é calculada a partir da incidência de infectados e a taxa de internação, ou seja, a proporção de indivíduos que necessitam ser internados.</p>
    </div>
</details>
'''
def make_SIMULATION_PARAMS(SEIR0, intervals, should_estimate_r0):
    alpha_inv_inf, alpha_inv_sup, _, _ = intervals[0]
    gamma_inv_inf, gamma_inv_sup, _, _ = intervals[1]

    if not should_estimate_r0:
        r0_inf, r0_sup, _, _ = intervals[2]
        r0_txt = f'- $${r0_inf:.03} < R_{{0}} < {r0_sup:.03}$$'
    else:
        r0_txt = '- $$R_{{0}}$$ está sendo estimado com dados históricos'

    intro_txt = '''

    ### Outros Parâmetros 

    Valores iniciais dos compartimentos:
    '''

    seir0_labels = [
        "Suscetíveis",
        "Expostos",
        "Infectados",
        "Removidos",
    ]
    seir0_values = list(map(int, SEIR0))
    seir0_dict = {
        "Compartimento": seir0_labels,
        "Valor inicial": seir0_values,
    }

    other_params_txt = f'''
    Demais parâmetros:
    - $${alpha_inv_inf:.03} < T_{{incub}} = 1/\\alpha < {alpha_inv_sup:.03}$$
    - $${gamma_inv_inf:.03} < T_{{infec}} = 1/\gamma < {gamma_inv_sup:.03}$$
    {r0_txt}

    Os intervalos de $$T_{{incub}}$$ e $$T_{{infec}}$$ definem 95% do intervalo de confiança de uma distribuição LogNormal.
    '''
    return intro_txt, seir0_dict, other_params_txt


SIMULATION_CONFIG = '''
---

### Configurações da  simulação 

#### Seleção Brasil ou UF
É possível selecionar o Brasil ou uma unidade da federação para utilizar seus parâmetros nas condições inicias de *População total* (N), *Indivíduos infecciosos inicialmente* (I0), *Indivíduos removidos com imunidade inicialmente* (R0) e *Indivíduos expostos inicialmente (E0)*.

#### Limites inferiores e superiores dos parâmetros
Também podem ser ajustados limites superior e inferior dos parâmetros *Período infeccioso*, *Tempo de incubação* e *Número básico de reprodução*. Estes limites definem um intervalo de confiança de 95% de uma distribuição log-normal para cada parâmetro.\n\n\n
'''

DATA_SOURCES = '''
---

### Fontes dos dados

* Casos confirmados por estado: [Painel de casos de doença pelo coronavírus 2019 (COVID-19) no Brasil pelo Monitora Covid](https://bigdata-covid19.icict.fiocruz.br)
* População: Estimativas da população enviadas ao TCU pelo IBGE em 01/07/2019(disponível em: [IBGE - Estimativas da população](https://www.ibge.gov.br/estatisticas/sociais/populacao/9103-estimativas-de-populacao.html))
### Referências

* [Report of the WHO-China Joint Mission on Coronavirus Disease 2019 (COVID-19)](https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf)
* [Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia](https://www.nejm.org/doi/full/10.1056/NEJMoa2001316)
* [Estimation of the reproductive number of novel coronavirus (COVID-19) and the probable outbreak size on the Diamond Princess cruise ship: A data-driven analysis](https://www.ijidonline.com/article/S1201-9712(20)30091-6/fulltext)
* [MIDAS Online Portal for COVID-19 Modeling Research](https://midasnetwork.us/covid-19/#resources)

'''

r0_ESTIMATION_TITLE = '''

### Número de reprodução básico $R_{{0}}$

'''

def r0_ESTIMATION(place, date): return  f'''
O número de reprodução básico $R_{0}$ está sendo estimado com os dados históricos de {place}. O valor utilizado no modelo SEIR-Bayes é o do dia {date}, que é o mais recente.

Caso você queria especificar o valor manualmente, desabilite a opção acima e insira os valores desejados.

**(!) Importante**: A estimação é sensível à qualidade das notificações dos casos positivos.
'''

r0_ESTIMATION_DONT = '''
Utilize o menu à esquerda para configurar o parâmetro.
'''

r0_CITATION = '''
A metodologia utilizada para estimação foi baseada no artigo [*Thompson, R. N., et al. "Improved inference of time-varying reproduction numbers during infectious disease outbreaks." Epidemics 29 (2019): 100356*](https://www.sciencedirect.com/science/article/pii/S1755436519300350). O código da implementação pode ser encontrado [aqui](https://github.com/3778/COVID-19/blob/master/covid19/estimation.py).
'''

def r0_NOT_ENOUGH_DATA(w_place, w_date): return f'''
**{w_place} não possui dados suficientes na data
{w_date} para fazer a estimação. Logo, foram
utilizados os dados agregados Brasil**
'''



def insert_logos():
    logo_html = {}
    logo_html["3778"] = (
        '<a href="https://3778.care"> '
        '<img src="https://imgur.com/XVMCKGT.png" alt="Logomarca 3778" style="border:0px;margin-left:40px;margin-top:7px;float:right;width:70px;"></img>'
        "</a>"
    )
    logo_html["fiocruz"] = (
        '<a href="https://bigdata.icict.fiocruz.br/"> '
        '<img src="https://i.imgur.com/tS4CNnB.png" alt="Logomarca PCDAS" style="border:0px;margin-left:40px;margin-top:7px;float:right;width:70px;"></img>'
        "</a>"
    )
    return '<div style="text-align: right;">' + logo_html["3778"] + logo_html["fiocruz"] + "</div>"

SRAG_DETAIL = '''
<details>
    <summary style="color: rgb(38, 39, 48);"><strong>Mostrar metodologia de cálculo</strong></summary>
    <div style="color: rgb(38, 39, 48);">
        <p>O cálculo da subnotificação corresponde ao excesso de registros de óbitos por SRAG em relação ao histórico de ocorrências.</p>
    </div>
</details>
'''

UTI_INTERNACAO_DETAIL = '''
<details>
    <summary style="color: rgb(38, 39, 48);"><strong>Mostrar metodologia de cálculo</strong></summary>
    <div style="color: rgb(38, 39, 48);">
        <ul>
            <li><b> População idosa</b>: Utiliza a taxa de internação considerando apenas as pessoas com 60 anos ou mais por local (estado, município ou país).</li>
            <li><b> População adulta e crônica</b>: Utiliza a taxa de internação considerando apenas pessoas entre 20 e 59 anos com pelo menos uma doença crônica não transmissível.</li>
        </ul>
    </div>
</details>
'''

LETHALITY_TYPE_DETAIL = '''
<details>
    <summary style="color: rgb(38, 39, 48);"><strong>Mostrar metodologia de cálculo</strong></summary>
    <div style="color: rgb(38, 39, 48);">
        <ul>
            <li><b>Estimada</b>: Utiliza a porcentagem de óbitos do último dado histórico.</li>
            <li><b>Ponderada por faixa etária</b>: Utiliza a porcentagem de óbitos do estado (se estiver disponível).</li>
            <li><b>Média Móvel</b>: Calcula a médias móvel com halflife de 7. Este método atribui um peso maior à óbitos que ocorreram mais recentemente.</li>
        </ul>
    </div>
</details>
'''

def DEATHS_TOTAL_COUNT(lower, mean, upper): return f'''
Total de óbitos acumulados:

* Limite superior: {upper}
* Média: {mean}
* Limite inferior: {lower}
'''
