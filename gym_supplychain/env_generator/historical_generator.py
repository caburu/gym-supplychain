import numpy as np

class ProdPlanScenario:
    def __init__(self):
        # Tamanho do período (número de horas)
        self.H = 0

        # Listas das entidades do problema (produtos, frentes, estoques, etc.)
        self.produtos   = []
        self.mat_primas = []
        self.frentes    = []
        self.est_matp   = []
        self.est_prod   = []
        self.centrais   = []
        self.pts_dem    = []
        self.periodos   = []

        # Transformação matéria-prima em produto
        self.transf_matp_prod = []

        # Estoques iniciais
        self.esto_p0_estmp = []
        self.esto_p0_estpr = []

        # Material inicial nas centrais
        self.cen_p1_matp_cen = []

        # Demandas
        self.demanda_per_prod_dem = []

        # Parâmetros de tempo de produção/processamento (em unidades h de tempo)
        self.tempo_prod_matp_fren = []
        self.tempo_prod_prod_fren = []
        self.tempo_proc_matp_cen  = []
        self.tempo_proc_prod_cen  = []

        # Custos de produção
        self.custo_prod_matp_fren = []
        self.custo_prod_prod_fren = []

        # Custos de estoques
        self.custo_esto_estmp  = []
        self.custo_esto_estpr  = []

        # Custos de processamento
        self.custo_proc_matp_cen  = []
        self.custo_proc_prod_cen  = []

        # Custos de transporte
        self.custo_tran_matp_fren_est = []
        self.custo_tran_prod_fren_est = []
        self.custo_tran_matp_est_est  = []
        self.custo_tran_prod_est_est  = []
        self.custo_tran_matp_fren_cen = []
        self.custo_tran_matp_est_cen  = []
        self.custo_tran_prod_cen_est  = []
        self.custo_tran_prod_cen_dem  = []
        self.custo_tran_prod_est_dem  = []

        # Limites de produção
        self.UB_prod_matp_fren = []
        self.UB_prod_prod_fren = []

        # Limites de estoque
        self.UB_esto_estmp = []
        self.UB_esto_estpr = []

        # Limites de processamento
        self.UB_proc_matp_cen = []
        self.UB_proc_prod_cen = []

        # Limites de transporte
        self.UB_tran_matp_fren_est = []
        self.UB_tran_matp_est_est  = []
        self.UB_tran_prod_est_est  = []
        self.UB_tran_matp_fren_cen = []
        self.UB_tran_matp_est_cen  = []
        self.UB_tran_prod_cen_est  = []
        self.UB_tran_prod_cen_dem  = []
        self.UB_tran_prod_est_dem  = []


def chain_settings():
    """
    Define as configurações gerais da cadeia de suprimentos.

    São os dados necessários para gerar o histórico da cadeia.
    """
    chain = {}

    # Número de períodos a se gerar o histórico
    chain['periods'] = 10

    # Definição do tipo de cadeia
    chain['type'] = {}
    # Cadeia sequencial significa que os materiais seguem ordem fixa:
    #   fornecedores -> estoques -> fábricas -> estoques -> pontos de demanda
    chain['type']['network'] = 'sequencial'
    # Estoques locais significa que os fornecedores e fábricas possuem um
    #   estoque de cada produto ligados diretamente a eles e que não existem
    #   outros estoques na cadeia.
    chain['type']['stocks']  = 'local'
    # Definição das capacidades da cadeia em geral (produção, transporte, fabricação, etc.)
    #   Infinito indica que a rede não tem restrição de capacidade
    chain['type']['upper_limits']  = 'infinity'

    # Definição das quantidades de fornecedores, fábricas e revendedores
    #   Obs: como estoques são locais não é necessário explicitar aqui.
    chain['levels'] = {}
    chain['levels'] ['suppliers'] = 2
    chain['levels'] ['factories'] = 2
    chain['levels'] ['retailers'] = 2

    # Quantidades de matérias-primas e produtos e fator de conversão de matéria-
    #   prima para produto.
    chain['materials'] = {}
    chain['materials']['raw'] = 1
    chain['materials']['products'] = 1
    chain['materials']['raw_to_product'] = [[1]]
    chain['materials']['initial_stocks'] = {}
    #  quantidade inicial de matéria-prima por estoque
    chain['materials']['initial_stocks']['supp_stocks'] = [[10, 10]]
    #  quantidade inicial de produto por estoque
    chain['materials']['initial_stocks']['fact_stocks'] = [[10, 10]]

    # Definição da demanda de produtos
    chain['demand'] = {}
    # Demanda inicial será definida por uma distribuição uniforme
    chain['demand']['function'] = 'uniform'
    # ... com valores definidos na faixa abaixo
    chain['demand']['range'] = [10,30]
    chain['demand']['variation'] = {}
    # Durante a execução, a demanda sofrerá uma variação de acordo com a faixa abaixo
    chain['demand']['variation']['range'] = [0.05, 0.20]
    # ... em X% dos perídos (com X definido abaixo)
    chain['demand']['variation']['frequency'] = 0.25

    # Definição dos custos por unidade de matéria-prima ou produto
    chain['costs'] = {}

    # Custos referentes aos fornecedores
    chain['costs']['suppliers'] = {}
    #   custos de produção de cada fornecedor (para cada matéria-prima, por frente)
    chain['costs']['suppliers']['production'] = [[2, 2]]
    #   custos de estoque local (para cada matéria-prima, por estoque)
    chain['costs']['suppliers']['stock'] = [[1, 1]]

    # Custos referentes às fábricas
    chain['costs']['factories'] = {}
    #   custos de transformação de matéria-prima em produto (para cada produto, por fábrica)
    chain['costs']['factories']['production'] = [[3,3]]
    #   custos de estoque local (para cada produto, por estoque)
    chain['costs']['factories']['stock'] = [[1, 1]]

    # Custos de transporte
    chain['costs']['transport'] = {}
    #   de fornecedores para estoques (por matéria-prima, fornecedores e estoques)
    chain['costs']['transport']['supp_stock'] = [[[1,2],[2,1]]]
    #   de fornecedores para fábricas (por matéria-prima, fornecedores e fábricas)
    chain['costs']['transport']['supp_factories'] = [[[2,3],[3,2]]]
    #   de estoques para fábricas (por matéria-prima, estoques e fábricas)
    chain['costs']['transport']['stock_factories'] = [[[1,2],[2,1]]]
    #   de fábricas para estoques (por produto, fábricas e estoques)
    chain['costs']['transport']['fact_stock'] = [[[1,2],[2,1]]]
    #   de fábricas para pontos de demanda (por produto, fábrica e pontos de demanda)
    chain['costs']['transport']['fact_demand'] = [[[2,3],[3,2]]]
    #   de estoques para fábricas (por produto, estoques e pontos de demanda)
    chain['costs']['transport']['stock_demand'] = [[[1,2],[2,1]]]

    # Dados externos à cadeia (no caso cotação do produto)
    chain['external_data'] = {}
    chain['external_data']['price'] = {}
    chain['external_data']['price']['function'] = 'uniform'
    chain['external_data']['price']['range'] = [60,70]

    return chain

def production_plan_generator(chain):
    """
    Gera um cenário para que possa ser criado um plano de produção otimizado
    """

    # O gerador por enquanto só trabalha com esse tipo de cadeia
    assert chain['type']['network'] == 'sequencial'
    assert chain['type']['stocks']  == 'local'
    assert chain['type']['upper_limits']  == 'infinity'

    pps = ProdPlanScenario()

    # Tamanho do período (número de horas)
    pps.H = 0

    # Listas das entidades do problema (produtos, frentes, estoques, etc.)
    pps.produtos   = range(chain['materials']['products'])
    pps.mat_primas = range(chain['materials']['raw'])
    pps.frentes    = range(chain['levels'] ['suppliers'])
    pps.est_matp   = range(chain['materials']['raw']*chain['levels'] ['suppliers'])
    pps.est_prod   = range(chain['materials']['products']*chain['levels'] ['factories'])
    pps.centrais   = range(chain['levels'] ['factories'])
    pps.pts_dem    = range(chain['levels'] ['retailers'])
    pps.periodos   = range(chain['periods'])

    # Transformação matéria-prima em produto
    pps.transf_matp_prod = chain['materials']['raw_to_product']

    # Estoques iniciais
    pps.esto_p0_estmp = chain['materials']['initial_stocks']['supp_stocks']
    pps.esto_p0_estpr = chain['materials']['initial_stocks']['fact_stocks']

    # Material inicial nas centrais (ZERADO)
    pps.cen_p1_matp_cen = np.zeros(size=(chain['materials']['raw'],chain['levels'] ['factories']))

    # Demandas
    pps.demanda_per_prod_dem = []

    # Parâmetros de tempo de produção/processamento (em unidades h de tempo)
    pps.tempo_prod_matp_fren = []
    pps.tempo_prod_prod_fren = []
    pps.tempo_proc_matp_cen  = []
    pps.tempo_proc_prod_cen  = []

    # Custos de produção
    pps.custo_prod_matp_fren = []
    pps.custo_prod_prod_fren = []

    # Custos de estoques
    pps.custo_esto_estmp  = []
    pps.custo_esto_estpr  = []

    # Custos de processamento
    pps.custo_proc_matp_cen  = []
    pps.custo_proc_prod_cen  = []

    # Custos de transporte
    pps.custo_tran_matp_fren_est = []
    pps.custo_tran_prod_fren_est = []
    pps.custo_tran_matp_est_est  = []
    pps.custo_tran_prod_est_est  = []
    pps.custo_tran_matp_fren_cen = []
    pps.custo_tran_matp_est_cen  = []
    pps.custo_tran_prod_cen_est  = []
    pps.custo_tran_prod_cen_dem  = []
    pps.custo_tran_prod_est_dem  = []

    # Limites de produção
    pps.UB_prod_matp_fren = []
    pps.UB_prod_prod_fren = []

    # Limites de estoque
    pps.UB_esto_estmp = []
    pps.UB_esto_estpr = []

    # Limites de processamento
    pps.UB_proc_matp_cen = []
    pps.UB_proc_prod_cen = []

    # Limites de transporte
    pps.UB_tran_matp_fren_est = []
    pps.UB_tran_matp_est_est  = []
    pps.UB_tran_prod_est_est  = []
    pps.UB_tran_matp_fren_cen = []
    pps.UB_tran_matp_est_cen  = []
    pps.UB_tran_prod_cen_est  = []
    pps.UB_tran_prod_cen_dem  = []
    pps.UB_tran_prod_est_dem  = []
