def global_format_func(s):
    fmt = {
        'country': "País",
        'state': 'Estado',
        'city': 'Município',
        'leth_est': 'Estimada',
        'leth_age': 'Ponderada por faixa etária',
        'leth_ewm': 'Média móvel',
        'elderly_risk': 'População idosa',
        'chronic_disease_risk': 'População adulta e crônica'
    }
    if s in fmt.keys():
        return fmt[s]
    else:
        return s
