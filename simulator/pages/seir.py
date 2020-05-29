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
    'asymptomatic_rate': 50.0,
    'Leitos': 0.005,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.0, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),
}
DERIVATIVES = {
    'functions': {
        'Leitos': lambda df: df['Infected'] * DERIVATIVES['values']['Leitos'],
        #'Ventiladores': lambda df: df['Leitos'] * DERIVATIVES['values']['Ventiladores'],
    },
    'values': {
        'Leitos': 0.005,
        #'Ventiladores': 0.25,
    },
    'descriptions': {
        'Leitos': 'Número de leitos necessários por infectado',
        #'Ventiladores': 'Número de ventiladores necessários por leito ocupado',
    },
}

def hideable(func, show, hidden_value=None):
    def curry(*args, **kwargs):
        return func(*args, **kwargs) if show else hidden_value
    return curry

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


def make_param_widgets(NEIR0, widget_values, r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    st.sidebar.markdown('#### Parâmetros da simulação')
    fator_subr = hideable(st.sidebar.number_input,
                          show=not('fator_subr' in widget_values),
                          hidden_value=widget_values.get('fator_subr'))(
            ('Fator de subnotificação. Este número irá multiplicar o número de infectados e expostos.'),
            min_value=1.0, max_value=200.0, step=0.1,
            value=defaults['fator_subr'])

    lethality_mean_place = st.sidebar.empty()
    
    asymptomatic_rate = hideable(st.sidebar.number_input,
                          show=not('asymptomatic_rate' in widget_values),
                          hidden_value=widget_values.get('asymptomatic_rate'))(
            ('Taxa de assintomáticos em %'),
            min_value=0.0, max_value=99.0, step=0.1,
            value=defaults['asymptomatic_rate'])/ 100
    
    for derivative in DERIVATIVES['descriptions']:
        DERIVATIVES['values'][derivative] = hideable(
            st.sidebar.number_input,
            show=not(derivative in widget_values),
            hidden_value=widget_values.get(derivative))(
            DERIVATIVES['descriptions'][derivative],
            min_value=0.0, max_value=10.0, step=0.0001,
            value=defaults[derivative], format="%.4f"
        )


    st.sidebar.markdown('#### Condições iniciais')
    N = hideable(st.sidebar.number_input,
                show=not('N' in widget_values),
                hidden_value=widget_values.get('N'))(
                'População total (N)',
                min_value=_N0, max_value=_N0, step=1,
                value=_N0)

    E0 = hideable(st.sidebar.number_input,
                 show=not('E0' in widget_values),
                 hidden_value=widget_values.get('E0'))(
                 'Indivíduos expostos inicialmente (E0)',
                 min_value=_E0, max_value=_E0,
                 value=_E0)

    I0 = hideable(st.sidebar.number_input,
                  show=not('I0' in widget_values),
                  hidden_value=widget_values.get('I0'))(
                  'Indivíduos infecciosos inicialmente (I0)',
                  min_value=_I0, max_value=_I0,
                  value= _I0)

    R0 = hideable(st.sidebar.number_input,
                  show=not('R0' in widget_values),
                  hidden_value=widget_values.get('R0'))(
                  'Indivíduos removidos com imunidade inicialmente (R0)',
                  min_value=_R0, max_value=_R0,
                  value=_R0)

    st.sidebar.markdown('#### Período de infecção (1/γ) e tempo incubação (1/α)')

    gamma_inf = hideable(st.sidebar.number_input,
                        show=not('gamma_inf' in widget_values),
                        hidden_value=widget_values.get('gamma_inf'))(
                        'Limite inferior do período infeccioso médio em dias (1/γ)',
                        min_value=1.0, max_value=60.0, step=0.1,
                        value=defaults['gamma_inv_dist'][0])

    gamma_sup = hideable(st.sidebar.number_input,
                        show=not('gamma_sup' in widget_values),
                        hidden_value=widget_values.get('gamma_sup'))(
                        'Limite superior do período infeccioso médio em dias (1/γ)',
                        min_value=1.0, max_value=60.0, step=0.1,
                        value=defaults['gamma_inv_dist'][1])

    alpha_inf = hideable(st.sidebar.number_input,
                        show=not('alpha_inf' in widget_values),
                        hidden_value=widget_values.get('alpha_inf'))(
                        'Limite inferior do tempo de incubação médio em dias (1/α)',
                        min_value=0.1, max_value=60.0, step=0.1,
                        value=defaults['alpha_inv_dist'][0])

    alpha_sup = hideable(st.sidebar.number_input,
                        show=not('alpha_sup' in widget_values),
                        hidden_value=widget_values.get('alpha_sup'))(
                        'Limite superior do tempo de incubação médio em dias (1/α)',
                        min_value=0.1, max_value=60.0, step=0.1,
                        value=defaults['alpha_inv_dist'][1])

    st.sidebar.markdown('#### Parâmetros gerais')

    t_max = hideable(st.sidebar.number_input,
                    show=not('t_max' in widget_values),
                    hidden_value=int(widget_values.get('t_max', 90)))(
                    'Período de simulação em dias (t_max)',
                    min_value=7, max_value=90*3, step=1,
                    value=90)
    return ({'fator_subr': fator_subr/ (1 - asymptomatic_rate),
            'alpha_inv_dist': (alpha_inf, alpha_sup, interval_density, family),
            'gamma_inv_dist': (gamma_inf, gamma_sup, interval_density, family),
            't_max': t_max,
            'NEIR0': (N, E0, I0, R0)},
            lethality_mean_place)


def make_derivatives_widgets(defaults, widget_values):
    for derivative in DERIVATIVES['descriptions']:
        DERIVATIVES['values'][derivative] = hideable(
            st.sidebar.number_input,
            show=not(derivative in widget_values),
            hidden_value=widget_values.get(derivative))(
            DERIVATIVES['descriptions'][derivative],
            min_value=0.0, max_value=10.0, step=0.0001,
            value=defaults[derivative], format="%.4f"
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


def make_death_widget(lethality_mean_est, lethality_mean_place, widget_values):
    lethality_mean = hideable(lethality_mean_place.number_input,
                              show=not('lethality_mean' in widget_values),
                              hidden_value=widget_values.get('lethality_mean'))(
            ('Taxa de letalidade (em %).'),
            min_value=0.0, max_value=100.0, step=0.1,
            value=lethality_mean_est)
    return lethality_mean

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


def estimate_lethality(cases_death, cases_covid, is_ewm):
    if is_ewm:
        lethality_mean = (
            cases_death
            .div(cases_covid)
            .ewm(halflife=7)
            .mean()
            .iloc[-1]
        ) * 100
    else:
        lethality_mean = (
            cases_death.iloc[-1]
            / cases_covid.iloc[-1]
        ) * 100

    return float(lethality_mean)


def make_r0_widgets(widget_values, defaults=DEFAULT_PARAMS):
    r0_inf = hideable(st.number_input,
                     show=not('r0_inf' in widget_values),
                     hidden_value=widget_values.get('r0_inf'))(
                     'Limite inferior do número básico de reprodução médio (R0)',
                     min_value=0.01, max_value=10.0, step=0.25,
                     value=defaults['r0_dist'][0])

    r0_sup = hideable(st.number_input,
                      show=not('r0_sup' in widget_values),
                      hidden_value=widget_values.get('r0_sup'))(
            'Limite superior do número básico de reprodução médio (R0)',
            min_value=0.01, max_value=10.0, step=0.25,
            value=defaults['r0_dist'][1])
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
    st.markdown(texts.INTRO_MODELO)
    st.sidebar.markdown(texts.PARAMETER_SELECTION)
    w_granularity = st.sidebar.selectbox('Unidade',
                                         options=['country', 'state'],
                                         index=1,
                                         format_func=global_format_func)

    cases_df = data.load_cases(w_granularity, 'fiocruz')
    population_df = data.load_population(w_granularity)

    DEFAULT_PLACE = (DEFAULT_STATE if w_granularity == 'state' else
                     DEFAULT_COUNTRY)

    options_place = make_place_options(cases_df, population_df,w_granularity)

    w_place = st.sidebar.selectbox(global_format_func(w_granularity),
                                   options=options_place,
                                   index=options_place.get_loc(DEFAULT_PLACE),
                                   format_func=global_format_func)
    try:
        raw_widget_values = (pd.read_csv('data/param.csv')
                          .set_index('place')
                          .T
                          .to_dict()
                          [w_place])
        widget_values = {k: v for k, v in raw_widget_values.items() if not np.isnan(v)}
    except:
        widget_values = {}

    if widget_values:
        st.markdown("### Parâmetros carregados")
        st.write(f"Foram carregados parâmetros pré-selecionados para a unidade escolhida:  **{w_place}**.")
        st.write('  \n'.join(f"{param}: {value}" for param, value in widget_values.items()))
        st.markdown("---")

    options_date = make_date_options(cases_df, w_place)
    w_date = st.sidebar.selectbox('Data inicial',
                                  options=options_date,
                                  index=len(options_date)-1)
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
   
    
    # Previsão de infectados
    w_params, lethality_mean_place = make_param_widgets(NEIR0, widget_values)
    #make_derivatives_widgets(DERIVATIVES['values'])

    #Definições do modelo
    model = SEIRBayes(**w_params, r0_dist=r0_dist)
    model_output = model.sample(SAMPLE_SIZE)

    # Parâmetros de simulação
    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             r0_dist]
    SEIR0 = model._params['init_conditions']
    params_intro_txt, seir0_dict, other_params_txt = texts.make_SIMULATION_PARAMS(SEIR0, dists,
                                             should_estimate_r0)
    # Configurações da simulação
    st.markdown(texts.SIMULATION_CONFIG)
    # Explicação dos parâmetros de óbito e leitos
    st.markdown(texts.DESC_PARAMS_DEATHS)
    st.markdown(texts.DESC_PARAMS_LEITOS)
    #Outros parâmetros
    st.markdown(params_intro_txt)
    st.write(pd.DataFrame(seir0_dict).set_index("Compartimento"))
    
    st.markdown(other_params_txt)
    
    
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

    # Deaths
    st.markdown(texts.DEATHS_INTRO)

    # Estimativa de Letalidade
    is_ewm = st.checkbox(
            'Calcular taxa de letalidade por média móvel',
            value=True)
    lethality_mean_est = estimate_lethality(cases_df[w_place]["deaths"],
                                            cases_df[w_place]["totalCases"],
                                            is_ewm)
    # Make param
    lethality_mean = make_death_widget(lethality_mean_est,
            lethality_mean_place, widget_values)

    # Plot Deaths
    fig_deahts = plot_deaths(model_output, 'linear', w_date, lethality_mean,
                             w_params['fator_subr'])
    st.altair_chart(fig_deahts)
    st.markdown(texts.DEATH_DETAIL,unsafe_allow_html=True)
    st.markdown(texts.LEITOS_INTRO)
    derivatives = make_EI_derivatives(ei_df)
    derivatives_chart = plot_derivatives(derivatives, w_date)
    st.altair_chart(derivatives_chart)
    st.markdown(texts.LEITOS_DETAIL,unsafe_allow_html=True)
    

    
    # Fontes dos dados
    st.markdown(texts.DATA_SOURCES)

if __name__ == '__main__':
    write()
