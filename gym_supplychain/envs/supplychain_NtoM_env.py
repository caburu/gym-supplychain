from gym_supplychain.envs.supplychain_env import SupplyChainEnv

class SupplyChainNtoMEnv(SupplyChainEnv):
    """ Cria uma Cadeia de Suprimentos com N Fornecedores e M Varejistas (revendedores)

        - A capacidade de estoque é a mesma para todos os nós da cadeia.
        - Os custos são os mesmos para toda a cadeia.
        - A quantidade inicial de estoque pode ser especificada para cada nó da cadeia.
    """
    def __init__(self, num_suppliers=1, num_retailers=3, initial_stocks=[], supply_capacity=20, stock_capacity=1000,
                 stock_cost=0.001, supply_cost=0.005, dest_cost=0.002,
                 unmet_demand_cost=1.0, exceeded_capacity_cost=1.0,
                 demand_range=(0,10), leadtime=1, total_time_steps=1000, seed=None):

        if not initial_stocks: # A posição zero é do fornecedor, as demais dos varejistas
            initial_stocks = [0]*(num_suppliers+num_retailers)

        retailers_names = []
        for i in range(num_retailers):
            retailers_names.append('Retailer'+str(i))

        nodes_info = {}
        for i in range(num_suppliers):
            nodes_info['Supplier'+str(i)] = {
                'initial_stock':initial_stocks[0], 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                'destinations':retailers_names, 'dest_costs':[dest_cost]*num_retailers}

        for i in range(num_retailers):
            nodes_info[retailers_names[i]] =  {
                'initial_stock':initial_stocks[num_suppliers+i], 'stock_capacity':stock_capacity,
                'stock_cost':stock_cost, 'last_level':True}

        super().__init__(nodes_info, unmet_demand_cost=unmet_demand_cost, exceeded_capacity_cost=exceeded_capacity_cost,
                         total_time_steps=total_time_steps, leadtime=leadtime)

if __name__ == '__main__':
    num_suppliers   = 2
    num_retailers   = 3
    initial_stocks  = [0, 20, 10, 0 , 5]
    stock_capacity  = 1000
    supply_capacity = 20
    stock_cost  = 0.001
    supply_cost = 0.005
    dest_cost   = 0.002
    unmet_demand_cost = 1.0
    exceeded_capacity_cost = 1.0
    demand_range = (0,10)
    leadtime = 2
    total_time_steps = 5

    env = SupplyChainNtoMEnv(num_suppliers=num_suppliers, num_retailers=num_retailers,
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
