import pandas as pd
from covid19 import data
from covid19.estimation import ReproductionNumber
from covid19.models import SEIRBayes

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
    return Rt.sample_from_posterior(sample_size=sample_size)

def generateR0DistributionByDate(r0_samples, date, place, min_days):
    r0_samples_cut = r0_samples[-min_days:]
    columns = pd.date_range(end=date, periods=r0_samples_cut.shape[1])
    data = (pd.DataFrame(r0_samples_cut, columns=columns)
              .stack(level=0)
              .reset_index()
              .rename(columns={"level_1": "Dias",
                               0: "r0"})
              [["Dias", "r0"]])
    data = data.groupby('Dias').mean()
    return data

cases_df = data.load_cases('state', 'fiocruz')
cases_df.head()
cases_df.info()
SAMPLE_SIZE=500
MIN_CASES_TH = 10
MIN_DAYS_r0_ESTIMATE = 14
EndDate = '2021-02-05'
Estado = 'RJ'

r0_samples = estimate_r0(cases_df,
                        Estado,# estado da estimativa
                        SAMPLE_SIZE,# dias da estimativa
                        MIN_DAYS_r0_ESTIMATE,# 
                        EndDate)
data = generateR0DistributionByDate(r0_samples,EndDate,Estado,MIN_DAYS_r0_ESTIMATE)


data.to_csv(f"R0_{Estado}_diario.csv")