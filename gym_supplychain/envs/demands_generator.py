import numpy as np

def generate_demand(rand_generator, num_demands, period, horizon, minv, maxv, std=None, sen_peaks=None):
    """ Gerador de demandas aleatórios (distr. uniforme, normal ou senoidal)
        A função gera a demanda para um único período para vários clientes.
        
        :param rand_generator: (numpy random) Gerador de números aleatórios utilizado.
        :param num_demands: (int) Quantidade de demandas a serem geradas (número de clientes).
        :param period: (int) Período para o qual as demandas serão geradas.
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
            return uniform_data(rand_generator, minv, maxv, num_demands)
        # Se passou desvio padrão é distribuição normal
        else:
            return normal_data(rand_generator, num_demands, minv, maxv, std)
    # Se passou número de picos, é uma função senoidal
    else: 
        std = 0 if std is None else std
        demands = np.zeros(num_demands)
        for i in range(num_demands):
            demands[i], _ = senoidal_data(rand_generator, period, horizon, minv, maxv, std, sen_peaks)
        return demands

def uniform_data(rand_generator, minv, maxv, num_demands):
    """ Gera a quantidade de demandas solicitadas a partir de uma distribuição uniforme
        dentro da faixa passada [minv,maxv] """
    return rand_generator.randint(low=minv, high=maxv+1, size=num_demands)

def normal_data(rand_generator, num_demands, minv, maxv, std):
    """ Gera a quantidade de demandas solicitadas a partir do valor média da faixa passada
        com um perturbação dada pela distribuição normal com média zero e desvio padrão """
    data = rand_generator.normal((maxv+minv)/2, std, size=num_demands)
    for i in range(len(data)):
        data[i] = cut_limit_data(data[i], minv, maxv)
    return data

def senoidal_data(rand_generator, period, horizon=360, minv=0, maxv=100, std=5, num_peaks=4):
    """Gera um ponto de uma curva senoidal referente à ordenada `period`

       A perturbação é gerada por uma distribuição normal de média zero
       e desvio padrão std. 
       Como 99.7% dos dados da distribuição ficam entre [-3std,+3std]
       a faixa de valores considerada para o valor média leva isso em consideração.
       Os valores que ficarem fora dos limites mínimo e máximo são cortados.
    """

    curve_min = 3*std+minv
    curve_max = maxv-3*std
    curve_range = curve_max - curve_min

    avg_value = curve_min + curve_range/2 + (curve_range/2)*np.sin(num_peaks*2*np.pi*period/horizon)
    sto_value = avg_value + rand_generator.normal(0, std)

    sto_value = cut_limit_data(sto_value, minv, maxv)

    return sto_value, avg_value

def cut_limit_data(value, minv, maxv):
    if value > maxv:
        return maxv
    elif value < minv:
        return minv
    return value

if __name__ == '__main__':
    from matplotlib import pyplot as ppt

    H = 360
    num_demands = 2
    rand_generator = np.random.RandomState(None)

    uni_data = []
    nor_data = []
    sen_data = []
    for i in range(H):
        uni_data.append(generate_demand(rand_generator, num_demands, i, H, 0, 100, std=None, sen_peaks=None))
        nor_data.append(generate_demand(rand_generator, num_demands, i, H, 0, 100, std=15, sen_peaks=None))
        sen_data.append(generate_demand(rand_generator, num_demands, i, H, 0, 100, std=5, sen_peaks=4))
    # print(data)
    # print(avg_data)
    for i in range(2):
        ppt.plot([d[i] for d in uni_data], label='uniform')
        ppt.plot([d[i] for d in nor_data], label='normal')
        ppt.plot([d[i] for d in sen_data], label='senoidal')
        ppt.legend()
        ppt.show()
