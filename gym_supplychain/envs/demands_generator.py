import numpy as np

def generate_demand(rand_generator, dem_shape, horizon, minv, maxv, std=None,
                    sen_peaks=None, minavg=None, maxavg=None, perturb_norm=True):
    """ Gerador de demandas aleatórios (distr. uniforme, normal ou senoidal)
        A função gera a demanda para todos os períodos do horizonte para vários clientes.
        
        :param rand_generator: (numpy random) Gerador de números aleatórios utilizado.
        :param dem_shape: (tuple) Quantidade de demandas a serem geradas (número de períodos,número de clientes).
        :param horizon: (int) Horizonte (tamanho do episódio), ou seja, número máximo de períodos.        
        :param minv: (int) Menor valor possível de demanda.
        :param maxv: (int) Maior valor possível de demanda.
        :param std: (float) Desvio padrão da distribuição Normal utilizada (se `None` será distribuição uniforme).
        :param sen_peaks: (int) Número de picos da função senoidal (se `None` não será uma senoidal).
        :param minv: (int) Menor valor de média (apenas no caso de senoidal)
        :param maxv: (int) Maior valor de média (apenas no caso de senoidal)
        :param perturb_norm: (bool) No caso de função senoidal indica se perturbação é dada pela distribuição normal ou uniforme.
        :return: (tuple) As demandas geradas.
    """
    # Se não passou número de picos de senoidal não é uma função senoidal
    if sen_peaks is None:
        # Se não passou desvio padrão é porque é distribuição uniforme
        if std is None:
            return uniform_data(rand_generator, horizon, dem_shape, minv, maxv)
        # Se passou desvio padrão é distribuição normal
        else:
            return normal_data(rand_generator, horizon, dem_shape, minv, maxv, std)
    # Se passou número de picos, é uma função senoidal
    else: 
        std = 0 if std is None else std
        return senoidal_data(rand_generator, horizon, dem_shape, minv, maxv, std, sen_peaks, minavg, maxavg, perturb_norm)

def uniform_data(rand_generator, horizon, dem_shape, minv, maxv):
    """ Gera a quantidade de demandas solicitadas a partir de uma distribuição uniforme
        dentro da faixa passada [minv,maxv] """
    return rand_generator.randint(low=minv, high=maxv+1, size=dem_shape)

def normal_data(rand_generator, horizon, dem_shape, minv, maxv, std):
    """ Gera a quantidade de demandas solicitadas a partir do valor média da faixa passada
        com um perturbação dada pela distribuição normal com média zero e desvio padrão """
    data = rand_generator.normal((maxv+minv)/2, std, size=dem_shape)
    data = np.round(data)
    for period in range(dem_shape[0]):
        for r in range(dem_shape[1]):
            data[period, r] = cut_limit_data(int(round(data[period, r])), minv, maxv)
    return data

def senoidal_data(rand_generator, horizon=360, dem_shape=(361,2), minv=0, maxv=400, std=5, num_peaks=4,
                  minavg=100, maxavg=300, perturb_norm=True):
    """Gera pontos ponto de uma curva senoidal referentes à ordenada `horizon`.

       A senoidal base varia entre `minavg` e `maxavg`. A partir dessa base uma
       perturbação é gerada por uma distribuição (normal ou uniforme).
       - No caso de perturbação normal, ela tem média zero e desvio padrão `std`.
       - No caso de perturbação uniforme, ela varia na faixa de [-3*std,3*std].
         Essa faixa foi escolhida porque conter 99.7% dos dados de uma distribuição
         normal (e assim poder comparar melhor com ela).
       
       Os valores que ficarem fora dos limites mínimo e máximo são cortados.
    """
    curve_range = maxavg - minavg

    # Para gerar o dado de um único período a fórmula é:
    #   demand = minavg + curve_range/2 * (1 + np.sin(num_peaks*2*np.pi*period/horizon)) + rand_generator.normal(0, std)
    # 
    # O trecho abaixo otimiza os cálculos guardando resultados parciais para serem reutilizados
    #   demand = minavg + half_curve * (1 + np.sin(sin_arg*period)) + perturb[period,d]
    half_curve = curve_range/2
    sin_arg = num_peaks*2*np.pi/horizon
    if perturb_norm:
        perturb = rand_generator.normal(0, std, size=dem_shape)
    else:
        perturb = rand_generator.randint(low=-3*std, high=3*std+1, size=dem_shape)

    data = np.zeros(dem_shape)
    for period in range(dem_shape[0]):
        for d in range(dem_shape[1]):
            dem = minavg + half_curve * (1 + np.sin(sin_arg*period)) + perturb[period,d]
            data[period,d] =  cut_limit_data(int(np.round(dem)), minv, maxv)
    
    return data

def cut_limit_data(value, minv, maxv):
    if value > maxv:
        return maxv
    elif value < minv:
        return minv
    return value

if __name__ == '__main__':
    from matplotlib import pyplot as ppt

    H = 360
    dem_shape = (361,2)
    num_demands = 2
    minv = 0
    maxv = 400
    minavg = 100
    maxavg = 300
    std = 50
    sen_peaks = 4
    rand_generator = np.random.RandomState(None)

    uni_data = generate_demand(rand_generator, dem_shape, H, minv=minv, maxv=maxv)
    nor_data = generate_demand(rand_generator, dem_shape, H, minv=minv, maxv=maxv, std=std)
    sen_data_norm = generate_demand(rand_generator, dem_shape, H, minv=minv, maxv=maxv, std=std,
                                    sen_peaks=sen_peaks, minavg=minavg, maxavg=maxavg)
    sen_data_unif = generate_demand(rand_generator, dem_shape, H, minv=minv, maxv=maxv, std=std,
                                    sen_peaks=sen_peaks, minavg=minavg, maxavg=maxavg, perturb_norm=False)
    
    def plot(i, data1, data2, label1, label2):
        ppt.plot(data1[:,i], label=label1)
        ppt.plot(data2[:,i], label=label2)
        ppt.legend()
        ppt.show()

    for i in range(2):
        plot(i, uni_data, nor_data, 'uniform', 'normal')
        plot(i, sen_data_norm, sen_data_unif, 'senoidal norm', 'senoidal unif')
