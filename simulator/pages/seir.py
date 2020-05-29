from json import dumps
import altair as alt
import base64
import numpy as np
import pandas as pd
import streamlit as st

from pages.utils.formats import global_format_func
from pages.utils.viz import prep_tidy_data_to_plot, make_combined_chart, plot_r0, prep_death_data_to_plot, make_death_chart, plot_derivatives
from pages.utils import texts

from covid19 import data
from covid19.estimation import ReproductionNumber
from covid19.models import SEIRBayes


SAMPLE_SIZE=500
MIN_CASES_TH = 10
MIN_DAYS_r0_ESTIMATE = 14
MIN_DATA_BRAZIL = '2020-03-26'
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_COUNTRY = 'Brasil'
DEFAULT_PARAMS = {
    'fator_subr': 1.0,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.0, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),
}
DERIVATIVES = {
    'functions': {
        'Leitos': lambda df: df['Infected'] * DERIVATIVES['values']['Leitos'],
        'Ventiladores': lambda df: df['Leitos'] * DERIVATIVES['values']['Ventiladores'],
    },
    'values': {
        'Leitos': 0.005,
        'Ventiladores': 0.25,
    },
    'descriptions': {
        'Leitos': 'Número de leitos necessários por infectado',
        'Ventiladores': 'Número de ventiladores necessários por leito ocupado',
    },
}

def prepare_for_r0_estimation(df):
    return (
        df
        ['newCases']
        .asfreq('D')
        .fillna(0)
        .rename('incidence')
        .reset_index()
        .rename(columns={'date': 'dates'})
        .set_index('dates')
    )


@st.cache
def make_brazil_cases(cases_df):
    return (cases_df
            .stack(level=1)
            .sum(axis=1)
            .unstack(level=1))



@st.cache
def make_place_options(cases_df, population_df,w_granularity):
    
    return (cases_df
            .swaplevel(0,1, axis=1) 
            ['totalCases']
            .pipe(lambda df: df >= MIN_CASES_TH)
            .any()
            .pipe(lambda s: s[s & s.index.isin(population_df.index)])
            .index if w_granularity == 'state' else
            cases_df
            .swaplevel(0,1, axis=1) 
            ['totalCases']
            .pipe(lambda df: df >= MIN_CASES_TH)
            .any()
            .pipe(lambda s: s[s & s.index.isin(['Brasil'])])
            .index)


@st.cache
def make_date_options(cases_df, place):
    return (cases_df
            [place]
            ['totalCases']
            .pipe(lambda s: s[s >= MIN_CASES_TH])
            [MIN_DATA_BRAZIL:]
            .index
            .strftime('%Y-%m-%d'))


def make_param_widgets(NEIR0, widget_values, lethality_mean_est,
                       r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    fator_subr = st.sidebar.number_input(
            ('Fator de subnotificação. Este número irá multiplicar o número de infectados e expostos.'),
            min_value=1.0, max_value=200.0, step=0.1,
            value=widget_values.get('fator_subr', defaults['fator_subr']))

    lethality_mean = st.sidebar.number_input(
            ('Taxa de letalidade (em %).'),
            min_value=0.0, max_value=100.0, step=0.1,
            value=lethality_mean_est)


    st.sidebar.markdown('#### Condições iniciais')
    N = st.sidebar.number_input('População total (N)',
                                min_value=0, max_value=1_000_000_000, step=1,
                                value=widget_values.get('N', _N0))

    E0 = st.sidebar.number_input('Indivíduos expostos inicialmente (E0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=widget_values.get('E0', _E0))

    I0 = st.sidebar.number_input('Indivíduos infecciosos inicialmente (I0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=widget_values.get('I0', _I0))

    R0 = st.sidebar.number_input('Indivíduos removidos com imunidade inicialmente (R0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=widget_values.get('R0', _R0))

    st.sidebar.markdown('#### Período de infecção (1/γ) e tempo incubação (1/α)')

    gamma_inf = st.sidebar.number_input(
            'Limite inferior do período infeccioso médio em dias (1/γ)',
            min_value=1.0, max_value=60.0, step=0.1,
            value=widget_values.get('gamma_inf', defaults['gamma_inv_dist'][0]))

    gamma_sup = st.sidebar.number_input(
            'Limite superior do período infeccioso médio em dias (1/γ)',
            min_value=1.0, max_value=60.0, step=0.1,
            value=widget_values.get('gamma_sup', defaults['gamma_inv_dist'][1]))

    alpha_inf = st.sidebar.number_input(
            'Limite inferior do tempo de incubação médio em dias (1/α)',
            min_value=0.1, max_value=60.0, step=0.1,
            value=widget_values.get('alpha_inf', defaults['alpha_inv_dist'][0]))

    alpha_sup = st.sidebar.number_input(
            'Limite superior do tempo de incubação médio em dias (1/α)',
            min_value=0.1, max_value=60.0, step=0.1,
            value=widget_values.get('alpha_sup', defaults['alpha_inv_dist'][1]))

    st.sidebar.markdown('#### Parâmetros gerais')

    t_max = st.sidebar.number_input('Período de simulação em dias (t_max)',
                                    min_value=7, max_value=90, step=1,
                                    value=widget_values.get('t_max', 90))


    return ({'fator_subr': fator_subr,
            'alpha_inv_dist': (alpha_inf, alpha_sup, interval_density, family),
            'gamma_inv_dist': (gamma_inf, gamma_sup, interval_density, family),
            't_max': t_max,
            'NEIR0': (N, E0, I0, R0)},
            lethality_mean)


def make_derivatives_widgets(defaults):
    for derivative in DERIVATIVES['descriptions']:
        DERIVATIVES['values'][derivative] = st.sidebar.number_input(
            DERIVATIVES['descriptions'][derivative],
            min_value=0.0, max_value=10.0, step=0.0001,
            value=defaults[derivative], format="%.4f"
        )

def make_death_subr_widget(defaults, place):
    return st.sidebar.number_input(
        ('Fator de subnotificação de óbitos'),
        min_value=0.0, max_value=100.0, step=0.1,
        value=defaults.get(place, 1.0)
    )

@st.cache
def make_NEIR0(cases_df, population_df, place, date):
    N0 = population_df[place]
    I0 = cases_df[place]['totalCases'][date]
    E0 = 2*I0
    R0 = 0
    return (N0, E0, I0, R0)


def make_download_href(df, params, r0_dist, should_estimate_r0):
    _params = {
        'subnotification_factor': params['fator_subr'],
        'incubation_period': {
            'lower_bound': params['alpha_inv_dist'][0],
            'upper_bound': params['alpha_inv_dist'][1],
            'density_between_bounds': params['alpha_inv_dist'][2]
         },
        'infectious_period': {
            'lower_bound': params['gamma_inv_dist'][0],
            'upper_bound': params['gamma_inv_dist'][1],
            'density_between_bounds': params['gamma_inv_dist'][2]
         },
    }
    if should_estimate_r0:
        _params['reproduction_number'] = {
            'samples': list(r0_dist)
        }
    else:
        _params['reproduction_number'] = {
            'lower_bound': r0_dist[0],
            'upper_bound': r0_dist[1],
            'density_between_bounds': r0_dist[2]
        }
    csv = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    b64_params = base64.b64encode(dumps(_params).encode()).decode()
    size = (3*len(b64_csv)/4)/(1_024**2)
    return f"""
    <a download='covid-simulator.3778.care.csv'
       href="data:file/csv;base64,{b64_csv}">
       Clique para baixar os resultados da simulação em format CSV ({size:.02} MB)
    </a><br>
    <a download='covid-simulator.3778.care.json'
       href="data:file/json;base64,{b64_params}">
       Clique para baixar os parâmetros utilizados em formato JSON.
    </a>
    """


def make_EI_df(model, model_output, sample_size):
    _, E, I, _, t = model_output
    size = sample_size*model.params['t_max']
    return (pd.DataFrame({'Exposed': E.reshape(size),
                          'Infected': I.reshape(size),
                          'run': np.arange(size) % sample_size})
              .assign(day=lambda df: (df['run'] == 0).cumsum() - 1))


def plot_EI(model_output, scale, start_date):
    _, E, I, _, t = model_output
    source = prep_tidy_data_to_plot(E, I, t, start_date)
    return make_combined_chart(source,
                               scale=scale,
                               show_uncertainty=True)


def plot_deaths(model_output, scale, start_date, lethality_mean, subnotification_factor):
    _, _, _, R, t = model_output
    R /= subnotification_factor
    R *= (lethality_mean/100)
    source = prep_death_data_to_plot(R, t, start_date)
    return make_death_chart(source,
                            scale=scale,
                            show_uncertainty=True)


def estimate_r0(cases_df, place, sample_size, min_days, w_date):
    used_brazil = False

    incidence = (
        cases_df
        [place]
        .query("totalCases > @MIN_CASES_TH")
        .pipe(prepare_for_r0_estimation)
        [:w_date]
    )

    if len(incidence) < MIN_DAYS_r0_ESTIMATE:
        used_brazil = True
        incidence = (
            make_brazil_cases(cases_df)
            .pipe(prepare_for_r0_estimation)
            [:w_date]
        )

    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=5.12, prior_scale=0.64,
                            si_pars={'mean': 4.89, 'sd': 1.48},
                            window_width=MIN_DAYS_r0_ESTIMATE - 2)
    Rt.compute_posterior_parameters()
    samples = Rt.sample_from_posterior(sample_size=sample_size)
    return samples, used_brazil



def estimate_lethality_mean(cases_death, cases_covid):
    lethality_mean = float((cases_death / cases_covid).mean()) * 100
    return lethality_mean



def make_r0_widgets(widget_values, defaults=DEFAULT_PARAMS):
    r0_inf = st.number_input(
             'Limite inferior do número básico de reprodução médio (R0)',
             min_value=0.01, max_value=10.0, step=0.25,
             value=widget_values.get('r0_inf', defaults['r0_dist'][0]))

    r0_sup = st.number_input(
            'Limite superior do número básico de reprodução médio (R0)',
            min_value=0.01, max_value=10.0, step=0.25,
            value=widget_values.get('r0_sup', defaults['r0_dist'][1]))
    return (r0_inf, r0_sup, .95, 'lognorm')


def make_EI_derivatives(ei_df, defaults=DERIVATIVES['functions']):
    ei_cols = ['Infected', 'Exposed']

    return (
        ei_df
        .groupby('day')
        [ei_cols]
        .mean()
        .assign(**defaults)
        .reset_index()
        .drop(ei_cols, axis=1)
    )


def write():

    st.markdown("## Modelo Epidemiológico (SEIR-Bayes)")
    st.sidebar.markdown(texts.PARAMETER_SELECTION)
    w_granularity = st.sidebar.selectbox('Unidade',
                                         options=['country', 'state'],
                                         index=1,
                                         format_func=global_format_func)

    cases_df = data.load_cases(w_granularity, 'fiocruz')
    population_df = data.load_population(w_granularity)
    srag_death_subnotification = data.load_srag_death_subnotification()

    DEFAULT_PLACE = (DEFAULT_STATE if w_granularity == 'state' else
                     DEFAULT_COUNTRY)
    
    options_place = make_place_options(cases_df, population_df,w_granularity) 
    
    w_place = st.sidebar.selectbox(global_format_func(w_granularity),
                                   options=options_place,
                                   index=options_place.get_loc(DEFAULT_PLACE),
                                   format_func=global_format_func)
    try:
        widget_values = (pd.read_csv('data/foo.csv')
                          .set_index('place')
                          .T
                          .to_dict()
                          [w_place])
    except:
        widget_values = {}
    options_date = make_date_options(cases_df, w_place)
    w_date = st.sidebar.selectbox('Data inicial',
                                  options=options_date,
                                  index=len(options_date)-1)
    death_subnotification = make_death_subr_widget(srag_death_subnotification, w_place)
    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)

    # Estimativa R0
    st.markdown(texts.r0_ESTIMATION_TITLE)
    should_estimate_r0 = st.checkbox(
            'Estimar R0 a partir de dados históricos',
            value=True)
    if should_estimate_r0:
        r0_samples, used_brazil = estimate_r0(cases_df,
                                              w_place,
                                              SAMPLE_SIZE,
                                              MIN_DAYS_r0_ESTIMATE,
                                              w_date)
        if used_brazil:
            st.write(texts.r0_NOT_ENOUGH_DATA(w_place, w_date))

        _place = 'Brasil' if used_brazil else w_place
        st.markdown(texts.r0_ESTIMATION(_place, w_date))

        st.altair_chart(plot_r0(r0_samples, w_date,
                                _place, MIN_DAYS_r0_ESTIMATE))
        r0_dist = r0_samples[:, -1]
        st.markdown(f'*O $R_{{0}}$ estimado está entre '
                    f'${np.quantile(r0_dist, 0.01):.03}$ e ${np.quantile(r0_dist, 0.99):.03}$*')
        st.markdown(texts.r0_CITATION)
    else:
        r0_dist = make_r0_widgets(widget_values)
        st.markdown(texts.r0_ESTIMATION_DONT)

    # Estimativa de Letalidade
    lethality_mean_est =  estimate_lethality_mean(cases_df[w_place]['deaths'],
                                                  cases_df[w_place]['totalCases'])
    # Previsão de infectados

    w_params, lethality_mean = make_param_widgets(NEIR0, widget_values, lethality_mean_est=lethality_mean_est)
    make_derivatives_widgets(DERIVATIVES['values'])

    model = SEIRBayes(**w_params, r0_dist=r0_dist)
    model_output = model.sample(SAMPLE_SIZE)
    ei_df = make_EI_df(model, model_output, SAMPLE_SIZE)
    st.markdown(texts.MODEL_INTRO)
    w_scale = st.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    fig = plot_EI(model_output, w_scale, w_date)
    st.altair_chart(fig)
    download_placeholder = st.empty()
    if download_placeholder.button('Preparar dados para download em CSV'):
        href = make_download_href(ei_df, w_params, r0_dist, should_estimate_r0)
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

    # Plot Deaths
    st.markdown(texts.DEATHS_INTRO)
    fig_deaths = plot_deaths(model_output, 'linear', w_date,
                             lethality_mean * death_subnotification,
                             w_params['fator_subr'])
    st.altair_chart(fig_deaths)
    st.markdown(texts.DEATH_DETAIL,unsafe_allow_html=True)

    derivatives = make_EI_derivatives(ei_df)
    derivatives_chart = plot_derivatives(derivatives, w_date)
    st.altair_chart(derivatives_chart)

    # Parâmetros de simulação
    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             r0_dist]
    SEIR0 = model._params['init_conditions']
    params_intro_txt, seir0_dict, other_params_txt = texts.make_SIMULATION_PARAMS(SEIR0, dists,
                                             should_estimate_r0)
    st.markdown(params_intro_txt)
    st.write(pd.DataFrame(seir0_dict).set_index("Compartimento"))
    st.markdown(other_params_txt)

    # Configurações da simulação
    st.markdown(texts.SIMULATION_CONFIG)
    # Fontes dos dados
    st.markdown(texts.DATA_SOURCES)

if __name__ == '__main__':
    write()
