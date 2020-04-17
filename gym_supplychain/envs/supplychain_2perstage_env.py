from gym_supplychain.envs.supplychain_env import SupplyChainEnv

class SupplyChain2perStageEnv(SupplyChainEnv):
    """ Cria uma Cadeia de Suprimentos com 2 Fornecedores, 2 Fábricas, 2 distribuidores e 2 Varejistas (revendedores)

        - A capacidade de estoque é a mesma para todos os nós da cadeia.
        - Os custos são os mesmos para toda a cadeia.
        - A quantidade inicial de estoque pode ser especificada para cada nó da cadeia.
    """
    def __init__(self, initial_stocks=[], supply_capacity=20, stock_capacity=1000,
                 stock_cost=0.001, supply_cost=0.005, dest_cost=0.002,
                 unmet_demand_cost=1.0, exceeded_capacity_cost=1.0,
                 demand_range=(0,10), leadtime=1, total_time_steps=1000, seed=None):

        if not initial_stocks: # A posição zero é do primeiro fornecedor, e assim por diante
            initial_stocks = [0]*(8)

        nodes_info = {}
        for i in range(2):
            nodes_info['Supplier'+str(i+1)] = {
                'initial_stock':initial_stocks[i], 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                'destinations':['Factory1','Factory2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Factory'+str(i+1)] = {
                'initial_stock':initial_stocks[2+i], 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                'destinations':['WholeSaler1','WholeSaler2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['WholeSaler'+str(i+1)] = {
                'initial_stock':initial_stocks[4+i], 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                'destinations':['Retailer1','Retailer2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Retailer'+str(i+1)] =  {
                'initial_stock':initial_stocks[6+i], 'stock_capacity':stock_capacity,
                'stock_cost':stock_cost, 'last_level':True}

        super().__init__(nodes_info, unmet_demand_cost=unmet_demand_cost, exceeded_capacity_cost=exceeded_capacity_cost,
                         total_time_steps=total_time_steps, leadtime=leadtime)

if __name__ == '__main__':
    initial_stocks  = [0, 0, 10, 10, 15, 15, 20, 20]
    stock_capacity  = 1000
    supply_capacity = 20
    stock_cost  = 0.001
    supply_cost = 0.005
    dest_cost   = 0.002
    unmet_demand_cost = 1.0
    exceeded_capacity_cost = 1.0
    demand_range = (0,10)
    leadtime = 1
    total_time_steps = 5

    env = SupplyChain2perStageEnv(
             initial_stocks=initial_stocks, supply_capacity=supply_capacity, stock_capacity=stock_capacity,
             stock_cost=stock_cost, supply_cost=supply_cost, dest_cost=dest_cost,
             unmet_demand_cost=unmet_demand_cost, exceeded_capacity_cost=exceeded_capacity_cost,
             demand_range=demand_range, leadtime=leadtime, total_time_steps=total_time_steps)
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
