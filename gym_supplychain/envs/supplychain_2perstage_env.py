from gym_supplychain.envs.supplychain_env import SupplyChainEnv

class SupplyChain2perStageEnv(SupplyChainEnv):
    """ Cria uma Cadeia de Suprimentos com 2 Fornecedores, 2 Fábricas, 2 distribuidores e 2 Varejistas (revendedores)

        - A capacidade de estoque é a mesma para todos os nós da cadeia.
        - Os custos são os mesmos para toda a cadeia.
        - A quantidade inicial de estoque pode ser especificada para cada nó da cadeia.
    """
    def __init__(self, initial_stocks=[20]*8, supply_capacities=[150,120], stock_capacities=[300,200]*4,
                 processing_ratio=3, processing_costs=[0.012,0.010], 
                 stock_costs=[0.001]*8, supply_costs=[0.006,0.004], dest_cost=0.002,
                 unmet_demand_cost=0.213, exceeded_capacity_cost=0.010,
                 demand_range=(10,21), leadtime=2, total_time_steps=360, seed=None):

        if not initial_stocks: # A posição zero é do primeiro fornecedor, e assim por diante
            initial_stocks = [0]*(8)

        nodes_info = {}
        for i in range(2):
            nodes_info['Supplier'+str(i+1)] = {
                'initial_stock':initial_stocks[i], 'stock_capacity':stock_capacities[i], 'stock_cost':stock_costs[i],
                'supply_capacity':supply_capacities[i], 'supply_cost':supply_costs[i],
                'destinations':['Factory1','Factory2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Factory'+str(i+1)] = {
                'initial_stock':initial_stocks[2+i], 'stock_capacity':stock_capacities[2+i], 'stock_cost':stock_costs[2+i],
                'processing_cost':processing_costs[i],
                'destinations':['WholeSaler1','WholeSaler2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['WholeSaler'+str(i+1)] = {
                'initial_stock':initial_stocks[4+i], 'stock_capacity':stock_capacities[4+i], 'stock_cost':stock_costs[4+i],
                'destinations':['Retailer1','Retailer2'], 'dest_costs':[dest_cost]*2}
        for i in range(2):
            nodes_info['Retailer'+str(i+1)] =  {
                'initial_stock':initial_stocks[6+i], 'stock_capacity':stock_capacities[6+i], 'stock_cost':stock_costs[6+i],
                'last_level':True}

        super().__init__(nodes_info, unmet_demand_cost=unmet_demand_cost, exceeded_capacity_cost=exceeded_capacity_cost,
                         processing_ratio=processing_ratio, 
                         total_time_steps=total_time_steps, leadtime=leadtime)

if __name__ == '__main__':
    initial_stocks  = [0, 0, 10, 10, 15, 15, 20, 20]
    
    demand_range      = (10,21)
    stock_capacities  = [300,200,300,200,300,200,300,200]
    supply_capacities = [50,40]
    processing_ratio  = 3
    leadtime     = 2
    stock_costs  = [0.001]*8
    dest_cost    = 0.002
    supply_costs = [0.006, 0.004]
    processing_costs = [0.012, 0.010]
    # Essa primeira tentativa se mostrou não muito "real"
        # Quanto custa para produzir e entregar uma unidade de produto (sem usar estoque)
        # product_cost = max(supply_costs) + 2*leadtime*dest_cost + max(processing_costs) + 4*max(stock_costs)        
    # Usei então um baseline para encontrar o custo médio do produto
    # e cheguei aos valores abaixo
    product_cost = 0.071
    
    # O custo de demanda não atendida é três vezes o custo de produzir (como se comprasse do concorrente).
    unmet_demand_cost = 3*product_cost
    
    # O custo de excesso de estoque talvez pudesse nem existir, já que o custo já incorrido no material
    # é perdido. Mas podemos considerar também que existiria um custo de desfazer do material.
    exceeded_capacity_cost = 10*max(stock_costs)
    
    episodes = 2
    total_time_steps = 5

    env = SupplyChain2perStageEnv(
             initial_stocks=initial_stocks, supply_capacities=supply_capacities, stock_capacities=stock_capacities,
             processing_ratio=processing_ratio, processing_costs=processing_costs,
             stock_costs=stock_costs, supply_costs=supply_costs, dest_cost=dest_cost,
             unmet_demand_cost=unmet_demand_cost, exceeded_capacity_cost=exceeded_capacity_cost,
             demand_range=demand_range, leadtime=leadtime, total_time_steps=total_time_steps)
             
    for ep in range(episodes):
        print('\n\nEpisódio:', ep, '\n\n')
        env.reset()
        env.render()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            env.render()
