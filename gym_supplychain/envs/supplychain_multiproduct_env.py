from .supplychain_env import SupplyChainEnv

class SupplyChainMultiProduct(SupplyChainEnv):
    def _create_chain(self, initial_stocks, stock_capacities, stock_costs, 
                      initial_supply, supply_capacities, supply_costs,
                      dest_cost, ship_capacity, initial_shipments, 
                      processing_capacities, processing_costs):
        """ Cria uma cadeia com a configuração bem diversa """
        nodes_info = {}

        nodes_info['Supplier1'] = {'initial_stock':initial_stocks[0], 'stock_capacity':stock_capacities[0], 'stock_cost':stock_costs,
                                    'initial_supply':initial_supply[0], 'supply_capacity':supply_capacities[0], 'supply_cost':supply_costs[0],
                                    'destinations':['Factory1','Factory2'], 'dest_costs':dest_cost, 
                                    'ship_capacity':ship_capacity}

        nodes_info['Supplier2'] = {'initial_stock':initial_stocks[1], 'stock_capacity':stock_capacities[1], 'stock_cost':stock_costs,
                                    'initial_supply':initial_supply[1], 'supply_capacity':supply_capacities[1], 'supply_cost':supply_costs[1],
                                    'destinations':['Factory1','Factory2'], 'dest_costs':dest_cost, 
                                    'ship_capacity':ship_capacity}

        nodes_info['Factory1'] = {'initial_stock':initial_stocks[2], 'stock_capacity':stock_capacities[2], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[0],
                                    'processing_capacity':processing_capacities[0], 'processing_cost':processing_costs[0],
                                    'destinations':['Wholesal1','Wholesal2'], 'dest_costs':dest_cost, 
                                    'ship_capacity':ship_capacity}

        nodes_info['Factory2'] = {'initial_stock':initial_stocks[3], 'stock_capacity':stock_capacities[3], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[1],
                                    'processing_capacity':processing_capacities[1], 'processing_cost':processing_costs[1],
                                    'destinations':['Wholesal1','Wholesal2'], 'dest_costs':dest_cost, 
                                    'ship_capacity':ship_capacity}

        nodes_info['Wholesal1'] = {'initial_stock':initial_stocks[4], 'stock_capacity':stock_capacities[4], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[2],
                                    'destinations':['Retailer1','Retailer2'], 'dest_costs':dest_cost,
                                    'ship_capacity':ship_capacity}

        nodes_info['Wholesal2'] = {'initial_stock':initial_stocks[5], 'stock_capacity':stock_capacities[5], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[3],
                                    'destinations':['Retailer1','Retailer2'], 'dest_costs':dest_cost,
                                    'ship_capacity':ship_capacity}

        nodes_info['Retailer1'] = {'initial_stock':initial_stocks[6], 'stock_capacity':stock_capacities[6], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[4],
                                    'last_level':True}

        nodes_info['Retailer2'] = {'initial_stock':initial_stocks[7], 'stock_capacity':stock_capacities[7], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments[5],
                                    'last_level':True}

        return nodes_info

    def __init__(self, 
                 demand_config_by_product=False,
                 num_products=2,                 
                 initial_stocks=None,
                 stock_capacities=None,
                 stock_costs=1,
                 initial_supply=None,
                 supply_capacities=None,
                 supply_costs=None,
                 dest_cost=None,
                 ship_capacity=None,
                 initial_shipments=None,
                 processing_capacities=None,
                 processing_costs=None,
                 processing_ratio=3,                 
                 unmet_demand_cost=216, 
                 exceeded_stock_capacity_cost=10,
                 exceeded_process_capacity_cost=10,
                 exceeded_ship_capacity_cost=10,
                 demand_range=(0,400), 
                 demand_std=None,
                 demand_sen_peaks=None,
                 avg_demand_range=None,
                 demand_perturb_norm=False,
                 stochastic_leadtimes=False,
                 avg_leadtime=2,
                 max_leadtime=2,
                 total_time_steps=360, 
                 seed=None,
                 build_info=False):                
        
        if not stock_capacities:
            stock_capacities=[[1600]*num_products,[1800]*num_products,
                              [6400]*num_products,[7200]*num_products,
                              [1600]*num_products,[1800]*num_products,
                              [1600]*num_products,[1800]*num_products]
        
        if not initial_stocks: initial_stocks=[[800]*num_products]*8
        if not initial_supply: initial_supply=[[[600]*avg_leadtime]*num_products,[[840]*avg_leadtime]*num_products]        
        if not supply_capacities: supply_capacities=[[600]*num_products,[840]*num_products]
        if not supply_costs: supply_costs=[[6]*num_products,[4]*num_products]
        if not dest_cost: dest_cost=[[2]*2]*num_products
        if not ship_capacity: ship_capacity=[500*num_products,500*num_products]
        if not initial_shipments: 
            initial_shipments=[[[600]*avg_leadtime]*num_products,[[840]*avg_leadtime]*num_products]+[[[240]*avg_leadtime]*num_products]*4
        if not processing_capacities: processing_capacities=[840*num_products,960*num_products]
        if not processing_costs: processing_costs=[[12]*num_products,[10]*num_products]

        nodes_info = self._create_chain(initial_stocks, stock_capacities, stock_costs, initial_supply,
                                        supply_capacities, supply_costs, dest_cost, ship_capacity, initial_shipments,
                                        processing_capacities, processing_costs)

        super().__init__(nodes_info, demand_config_by_product=demand_config_by_product,
                         num_products=num_products, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_std, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)


class SupplyChainMultiProduct_IncreasingCosts(SupplyChainMultiProduct):
    def __init__(self, num_products=2,
                 processing_ratio=3,                 
                 unmet_demand_cost=216,
                 exceeded_stock_capacity_cost=10,
                 exceeded_process_capacity_cost=10,
                 exceeded_ship_capacity_cost=10,
                 demand_range=(0,400),
                 demand_std=None,
                 demand_sen_peaks=None,
                 avg_demand_range=None,
                 demand_perturb_norm=False,
                 stochastic_leadtimes=False,
                 avg_leadtime=2,
                 max_leadtime=2,
                 total_time_steps=360, 
                 seed=None,
                 build_info=False):
                
        supply_costs=[[6*(i+1) for i in range(num_products)],
                      [4*(i+1) for i in range(num_products)]]

        dest_cost=[[2*(i+1)]*2 for i in range(num_products)]
        
        processing_costs=[[12*(i+1) for i in range(num_products)],
                          [10*(i+1) for i in range(num_products)]]

        stock_costs = [1*(i+1) for i in range(num_products)]

        super().__init__(supply_costs=supply_costs, dest_cost=dest_cost, processing_costs=processing_costs, stock_costs=stock_costs,
                         num_products=num_products, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_std, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)

class SupplyChainMultiProduct_DemConfigByProd(SupplyChainMultiProduct):
    def __init__(self, num_products=2,
                 processing_ratio=3,                 
                 unmet_demand_cost=216,
                 demand_std=None,
                 demand_perturb_norm=False,
                 exceeded_stock_capacity_cost=10,
                 exceeded_process_capacity_cost=10,
                 exceeded_ship_capacity_cost=10,                 
                 stochastic_leadtimes=False,
                 avg_leadtime=2,
                 max_leadtime=2,
                 total_time_steps=360, 
                 seed=None,
                 build_info=False):
        
        # Classe tratada para no máximo 3 produtos
        assert 1 <= num_products <= 3

        demand_config_by_product = True

        demand_perturb_norm = [demand_perturb_norm]*num_products
        
        # Configuração do primeiro produto (range (0,400), sazonal, 4 picos, média 100 a 300, perturbação passada)
        demand_range = [(0,400)]
        demand_stds = [demand_std]
        demand_sen_peaks = [4]
        avg_demand_range = [(100,300)]
                
        # Configuração do segundo produto (range (0,300), regular, perturbação passada)
        if num_products > 1:
            demand_range.append((0,300))
            demand_stds.append(demand_std)
            demand_sen_peaks.append(None)
            avg_demand_range.append(None)
        
        # Configuração do terceiro produto (range (0,300), sazonal, 3 picos, média 100 a 200, perturbação passada)
        if num_products > 2:
            demand_range.append((0,400))
            demand_stds.append(demand_std)
            demand_sen_peaks.append(2)
            avg_demand_range.append((100,300))

        super().__init__(demand_config_by_product=demand_config_by_product,
                         num_products=num_products, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_stds, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)

class SupplyChainMultiProduct_DemConfigByProd_IncCosts(SupplyChainMultiProduct):
    def __init__(self, num_products=2,
                 processing_ratio=3,                 
                 unmet_demand_cost=216,
                 demand_std=None,
                 demand_perturb_norm=False,
                 exceeded_stock_capacity_cost=10,
                 exceeded_process_capacity_cost=10,
                 exceeded_ship_capacity_cost=10,                 
                 stochastic_leadtimes=False,
                 avg_leadtime=2,
                 max_leadtime=2,
                 total_time_steps=360, 
                 seed=None,
                 build_info=False):
        
        # Classe tratada para no máximo 3 produtos
        assert 1 <= num_products <= 3
        
        demand_config_by_product = True

        demand_perturb_norm = [demand_perturb_norm]*num_products
        
        # Configuração do primeiro produto (range (0,400), sazonal, 4 picos, média 100 a 300, perturbação passada)
        demand_range = [(0,400)]
        demand_stds = [demand_std]
        demand_sen_peaks = [4]
        avg_demand_range = [(100,300)]
                
        # Configuração do segundo produto (range (0,300), regular, perturbação passada)
        if num_products > 1:
            demand_range.append((0,300))
            demand_stds.append([demand_std])
            demand_sen_peaks.append(None)
            avg_demand_range.append(None)
        
        # Configuração do terceiro produto (range (0,300), sazonal, 3 picos, média 100 a 200, perturbação passada)
        if num_products > 2:
            demand_range.append((0,400))
            demand_stds.append([demand_std])
            demand_sen_peaks.append(2)
            avg_demand_range.append((100,300))               

        supply_costs=[[6*(i+1) for i in range(num_products)],
                      [4*(i+1) for i in range(num_products)]]

        dest_cost=[[2*(i+1)]*2 for i in range(num_products)]
        
        processing_costs=[[12*(i+1) for i in range(num_products)],
                          [10*(i+1) for i in range(num_products)]]

        stock_costs = [1*(i+1) for i in range(num_products)]

        super().__init__(demand_config_by_product=demand_config_by_product,
                         supply_costs=supply_costs, dest_cost=dest_cost, processing_costs=processing_costs, stock_costs=stock_costs,
                         num_products=num_products, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_stds, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)