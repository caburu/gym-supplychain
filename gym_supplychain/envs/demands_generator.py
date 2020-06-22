import numpy as np

def generate_demand(rand_generator, dem_shape, horizon, minv, maxv, std=None, sen_peaks=None):
    """ Gerador de demandas aleatórios (distr. uniforme, normal ou senoidal)
        A função gera a demanda para todos os períodos do horizonte para vários clientes.
        
        :param rand_generator: (numpy random) Gerador de números aleatórios utilizado.
        :param dem_shape: (tuple) Quantidade de demandas a serem geradas (número de períodos,número de clientes).
        :param horizon: (int) Horizonte (tamanho do episódio), ou seja, número máximo de períodos.        
        :param minv: (int) Menor valor possível de demanda.
        :param maxv: (int) Maior valor possível de demanda.
        :param std: (float) Desvio padrão da distribuição Normal utilizada (se `None` será distribuição uniforme).
        :param sen_peaks: (int) Número de picos da função senoidal (se `None` não será uma senoidal).
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
        return senoidal_data(rand_generator, horizon, dem_shape, minv, maxv, std, sen_peaks)

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

def senoidal_data(rand_generator, horizon=360, dem_shape=(361,2), minv=0, maxv=100, std=5, num_peaks=4):
    """Gera pontos ponto de uma curva senoidal referentes à ordenada `period`

       A perturbação é gerada por uma distribuição normal de média zero
       e desvio padrão std. 
       Como 99.7% dos dados da distribuição ficam entre [-3std,+3std]
       a faixa de valores considerada para o valor média leva isso em consideração.
       Os valores que ficarem fora dos limites mínimo e máximo são cortados.
    """

    curve_min = 3*std+minv
    curve_max = maxv-3*std
    curve_range = curve_max - curve_min

    # Para gerar o dado de um único período a fórmula é:
    #   demand = curve_min + curve_range/2 * (1 + np.sin(num_peaks*2*np.pi*period/horizon)) + rand_generator.normal(0, std)
    # 
    # O trecho abaixo otimiza os cálculos guardando resultados parciais para serem reutilizados
    #   demand = curve_min + half_curve * (1 + np.sin(sin_arg*period)) + perturb[period,d]
    half_curve = curve_range/2
    sin_arg = num_peaks*2*np.pi/horizon
    perturb = rand_generator.normal(0, std, size=dem_shape)

    data = np.zeros(dem_shape)
    for period in range(dem_shape[0]):
        for d in range(dem_shape[1]):
            dem = curve_min + half_curve * (1 + np.sin(sin_arg*period)) + perturb[period,d]
            data[period,d] =  cut_limit_data(int(round(dem)), minv, maxv)
    
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
    rand_generator = np.random.RandomState(None)

    uni_data = generate_demand(rand_generator, dem_shape, H, 0, 100, std=None, sen_peaks=None)
    nor_data = generate_demand(rand_generator, dem_shape, H, 0, 100, std=15, sen_peaks=None)
    sen_data = generate_demand(rand_generator, dem_shape, H, 0, 100, std=5, sen_peaks=4)
    
    for i in range(2):
        ppt.plot(uni_data[:,i], label='uniform')
        ppt.plot(nor_data[:,i], label='normal')
        ppt.plot(sen_data[:,i], label='senoidal')
        ppt.legend()
        ppt.show()
