from .supplychain_env import SupplyChainEnv

class SupplyChain2perStageEnv(SupplyChainEnv):
    """ Cria uma Cadeia de Suprimentos com 2 Fornecedores, 2 Fábricas, 2 distribuidores e 2 Varejistas (revendedores)

        - A capacidade de estoque é a mesma para todos os nós da cadeia.
        - Os custos são os mesmos para toda a cadeia.
        - A quantidade inicial de estoque pode ser especificada para cada nó da cadeia.
        
        Obs: eu havia criado outras classes para configurações específicas, mas acabei não usando.
        E para não ter que ficar evoluindo código de coisa que não testei acabei removendo os arquivos.
        Mas até a versão do ICCL2020 ainda tinha (como referência caso seja necessário futuramente).
    """
    def __init__(self, initial_stocks=[0]*8, initial_supply=[60,60]*2, initial_shipments=[60,60]*2 + [20,20]*4,
                 supply_capacities=[120,150], processing_capacities=[300,300], stock_capacities=[200,300]*4,
                 processing_ratio=3, processing_costs=[12,10], 
                 stock_costs=[1]*8, supply_costs=[6,4], dest_cost=2,
                 unmet_demand_cost=216, exceeded_stock_capacity_cost=10, exceeded_process_capacity_cost=10,
                 demand_range=(10,20), demand_std=None, demand_sen_peaks=None, avg_demand_range=None,
                 stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2, 
                 total_time_steps=360, seed=None, build_info=False, demand_perturb_norm=False):

        if not initial_stocks: # A posição zero é do primeiro fornecedor, e assim por diante
            initial_stocks = [0]*(8)

        nodes_info = {}
        for i in range(2):
            nodes_info['Supplier'+str(i+1)] = {
                'initial_stock':initial_stocks[i], 'initial_supply':initial_supply[2*i:2*i+2],
                'stock_capacity':stock_capacities[i], 'stock_cost':stock_costs[i],
                'supply_capacity':supply_capacities[i], 'supply_cost':supply_costs[i],
                'destinations':['Factory1','Factory2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Factory'+str(i+1)] = {
                'initial_stock':initial_stocks[2+i], 'initial_shipments':initial_shipments[2*(i):2*(i)+2],
                'stock_capacity':stock_capacities[2+i], 'stock_cost':stock_costs[2+i],
                'processing_capacity':processing_capacities[i], 'processing_cost':processing_costs[i],
                'destinations':['WholeSaler1','WholeSaler2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['WholeSaler'+str(i+1)] = {
                'initial_stock':initial_stocks[4+i], 'initial_shipments':initial_shipments[2*(2+i):2*(2+i)+2], 
                'stock_capacity':stock_capacities[4+i], 'stock_cost':stock_costs[4+i],
                'destinations':['Retailer1','Retailer2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Retailer'+str(i+1)] =  {
                'initial_stock':initial_stocks[6+i], 'initial_shipments':initial_shipments[2*(4+i):2*(4+i)+2],
                'stock_capacity':stock_capacities[6+i], 'stock_cost':stock_costs[6+i],
                'last_level':True}

        super().__init__(nodes_info, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_std, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)


class SupplyChain2perStageSeasonalEnv(SupplyChain2perStageEnv):
    """ Classe atalho para criar uma cadeia 2perStage com demandas sazonais """
    def __init__(self, initial_stocks=[800]*8, initial_supply=[600]*2+[840]*2, 
                         initial_shipments=[600]*2+[840]*2+[240,240]*4,
                         supply_capacities=[600,840], processing_capacities=[840,960], stock_capacities=[1600,1800]*4,
                         processing_ratio=3, processing_costs=[12,10],
                         stock_costs=[1]*8, supply_costs=[6,4], dest_cost=2,
                         unmet_demand_cost=216, exceeded_stock_capacity_cost=10, exceeded_process_capacity_cost=10,
                         demand_range=(0,400), demand_std=10, demand_sen_peaks=4, avg_demand_range=(150,250),
                         stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2,
                         total_time_steps=360, seed=None, build_info=False, check_actions=False,
                         demand_perturb_norm=True):

        super().__init__(initial_stocks=initial_stocks, initial_supply=initial_supply, 
                         initial_shipments=initial_shipments,
                         supply_capacities=supply_capacities, processing_capacities=processing_capacities, 
                         stock_capacities=stock_capacities,
                         processing_ratio=processing_ratio, processing_costs=processing_costs, 
                         stock_costs=stock_costs, supply_costs=supply_costs, dest_cost=dest_cost,
                         unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         demand_range=demand_range, demand_std=demand_std, demand_sen_peaks=demand_sen_peaks,
                         avg_demand_range=avg_demand_range,
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         total_time_steps=total_time_steps, seed=seed,
                         build_info=build_info, demand_perturb_norm=demand_perturb_norm)

if __name__ == '__main__':

    
    def simulate(env):
        episodes = 2        
    
        env.seed(0)
        for ep in range(episodes):
                print('\n\nEpisódio:', ep, '\n\n')
                env.reset()
                env.render()
                done = False
                while not done:
                    action = env.action_space.sample()
                    _, _, done, _ = env.step(action)
                    env.render()

    total_time_steps = 5
    print('=========== Demanda uniforme ==========\n\n')
    simulate(SupplyChain2perStageEnv(total_time_steps=total_time_steps))
    
    print('\n\n=========== Demanda senoidal ==========\n\n')
    simulate(SupplyChain2perStageSeasonalEnv(total_time_steps=total_time_steps))
             
    
