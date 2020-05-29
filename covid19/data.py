import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import requests
from covid19.utils import state2initial

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL=('https://raw.githubusercontent.com/wcota/covid19br/'
                      'master/cases-brazil-cities-time.csv')
IBGE_POPULATION_PATH=DATA_DIR / 'ibge_population.csv'
WORLD_POPULATION_PATH=DATA_DIR / 'country_population.csv'
COVID_SAUDE_URL = ('https://raw.githubusercontent.com/3778/COVID-19/'
                   'master/data/latest_cases_ms.csv')

FIOCRUZ_URL = 'https://bigdata-covid19.icict.fiocruz.br/sd/dados_casos.csv'

LETHALITY_PATH=DATA_DIR / 'lethality_rates.csv'


def _prepare_fiocruz_data(df, by):
    if by == 'country':
        return (df.assign(country=np.where((df['name'].str.contains('^[\wA-z\wÀ-ú]')),
                                           df['name'],
                                           None)))

    if by == 'state':
        return (df.assign(state=np.where(df['name'].str.startswith('#BR'),
                                         df['name'].str[5:],
                                         None))
                  .replace({'state': state2initial}))
    if by == 'city':
        return (df.assign(city=np.where(df['name'].str.startswith('#Mun BR'),
                                        df['name'].str[9:],
                                        None))
                  .assign(city=lambda df: df['city'].str.rsplit(' ', 1)
                                                    .str.join('/')))


def _make_total_deaths(df, by):
    df = df.assign(deaths=lambda df: df.groupby([by])['new_deaths'].cumsum())
    return df


def load_cases(by, source='fiocruz'):
    '''Load cases from wcota/covid19br or covid.saude.gov.br or fiocruz

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:

        >>> cases_city = load_cases('city')
        >>> cases_city['São Paulo/SP']['newCases']['2020-03-20']
        47

        >>> cases_state = load_cases('state')
        >>> cases_state['SP']['newCases']['2020-03-20']
        110

        >>> cases_ms = load_cases('state', source='ms')
        >>> cases_ms['SP']['newCases']['2020-03-20']
        110

    '''
    assert source in ['ms', 'wcota', 'fiocruz']
    assert by in ['country', 'state', 'city']

    if source == 'monitora':
        assert by == 'state'
        df = (pd.read_csv(COVID_MONITORA_URL,
                          sep=';',
                          parse_dates=['date'],
                          dayfirst=True)
                .rename(columns={'casosNovos': 'newCases',
                                 'casosAcumulados': 'totalCases',
                                 'estado': 'state'}))

    if source == 'ms':
        assert by == 'state'
        df = (pd.read_csv(COVID_SAUDE_URL,
                          sep=';',
                          parse_dates=['date'],
                          dayfirst=True)
                .rename(columns={'casosNovos': 'newCases',
                                 'casosAcumulados': 'totalCases',
                                 'estado': 'state'}))

    elif source == 'wcota':
        df = (pd.read_csv(COVID_19_BY_CITY_URL, parse_dates=['date'])
                .query("state != 'TOTAL'"))

    elif source == 'fiocruz':
        df = (pd.read_csv(FIOCRUZ_URL, parse_dates=['date'])
                .rename(columns={'new_cases': 'newCases'})
                .pipe(_prepare_fiocruz_data, by=by)
                .assign(totalCases=lambda df: df.groupby([by])['newCases'].cumsum()))
        df = _make_total_deaths(df, by)


    return (df.groupby(['date', by])
              [['newCases', 'totalCases', 'deaths']]
              .sum()
              .unstack(by)
              .sort_index()
              .swaplevel(axis=1)
              .fillna(0)
              .astype(int))


def load_population(by):
    ''''Load population from IBGE.

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:

        >>> load_population('state').head()
        state
        AC      881935
        AL     3337357
        AM     4144597
        AP      845731
        BA    14873064
        Name: estimated_population, dtype: int64

        >>> load_population('city').head()
        city
        Abadia de Goiás/GO          8773
        Abadia dos Dourados/MG      6989
        Abadiânia/GO               20042
        Abaetetuba/PA             157698
        Abaeté/MG                  23237
        Name: estimated_population, dtype: int64

    '''
    assert by in ['country', 'state', 'city']

    if by == 'country':
        return (pd.read_csv(WORLD_POPULATION_PATH)
                    .groupby('country')
                    ['population']
                    .first())
    else:

        return (pd.read_csv(IBGE_POPULATION_PATH)
                   .rename(columns={'uf': 'state'})
                   .assign(city=lambda df: df.city + '/' + df.state)
                   .groupby(by)
                   ['estimated_population']
                   .sum()
                   .sort_index())

def prepare_age_data(level, old_col, new_col):
    BASE_URL = "http://api.sidra.ibge.gov.br/values/t/5918/p/201904/v/606/C58/all/f/n"
    url = f"{BASE_URL}{level}"
    r = requests.get(url)
    df = pd.read_json(r.text)
    df = (
        df.rename(columns=df.iloc[0])
        .drop(df.index[0])
        .drop(columns=["Nível Territorial", "Trimestre", "Variável", "Unidade de Medida"])
        .rename(columns={"Grupo de idade": "g_idade", old_col: new_col})
    )
    df = df.pivot(new_col, columns="g_idade")["Valor"].reset_index()
    df.columns.name = None
    return df

def load_age_group_rate(granularity):
    assert granularity in ["state", "country"]
    if granularity == "state":
        df = (
            prepare_age_data("/N3/all", "Unidade da Federação", "state")
            .replace({"state": state2initial})
            .set_index("state")
            .astype(int)
        )
    else:
        df = (
            prepare_age_data("/N1/all", "Brasil", "country")
            .assign(country=lambda df: df["country"].str.replace(" - ", "/"))
            .set_index(granularity)
            .astype(int)

        )
    return (df.assign(Jovem= lambda df: (df['0 a 13 anos'] + df['14 a 17 anos'])/df['Total'])
              .assign(Adulto= lambda df: (df['18 a 24 anos'] + df['25 a 39 anos'] + df['40 a 59 anos'])/df['Total'])
              .assign(Idoso= lambda df: df['60 anos ou mais']/df['Total'])
              .drop(df.columns[0:7], axis=1))

def load_lethality_rate():
    return (pd.read_csv(LETHALITY_PATH)
              .set_index('state')
              .rename(columns={'adult_lethality': 'Adulto',
                               'elder_lethality': 'Idoso',
                               'young_lethality': 'Jovem'}))
