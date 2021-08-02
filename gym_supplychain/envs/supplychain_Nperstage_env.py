from .supplychain_env import SupplyChainEnv

class SupplyChainNPerStage(SupplyChainEnv):
    def _create_chain(self, nodes_per_echelon, initial_stocks, stock_capacities, stock_costs, 
                      initial_supply, supply_capacities, supply_costs,
                      dest_cost, ship_capacity, initial_shipments, 
                      processing_capacities, processing_costs):
        """ Cria uma cadeia com a configuração bem diversa """
        nodes_info = {}

        for i in range(nodes_per_echelon['suppliers']):
            nodes_info[f'Supplier{i}'] = {'initial_stock':initial_stocks['suppliers'][i], 'stock_capacity':stock_capacities['suppliers'][i], 'stock_cost':stock_costs,
                                        'initial_supply':initial_supply[i], 'supply_capacity':supply_capacities[i], 'supply_cost':supply_costs[i],
                                        'destinations':[f'Factory{j}' for j in range(nodes_per_echelon['factories'])], 'dest_costs':dest_cost['suppliers'], 
                                        'ship_capacity':ship_capacity['suppliers']}

        for i in range(nodes_per_echelon['factories']):
            nodes_info[f'Factory{i}'] = {'initial_stock':initial_stocks['factories'][i], 'stock_capacity':stock_capacities['factories'][i], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments['factories'][i],
                                    'processing_capacity':processing_capacities[i], 'processing_cost':processing_costs[i],
                                    'destinations':[f'Wholesal{j}' for j in range(nodes_per_echelon['wholesalers'])], 'dest_costs':dest_cost['factories'], 
                                    'ship_capacity':ship_capacity['factories']}

        for i in range(nodes_per_echelon['wholesalers']):
            nodes_info[f'Wholesal{i}'] = {'initial_stock':initial_stocks['wholesalers'][i], 'stock_capacity':stock_capacities['wholesalers'][i], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments['wholesalers'][i],
                                    'destinations':[f'Retailer{j}' for j in range(nodes_per_echelon['retailers'])], 'dest_costs':dest_cost['wholesalers'],
                                    'ship_capacity':ship_capacity['wholesalers']}
        
        for i in range(nodes_per_echelon['retailers']):
            nodes_info[f'Retailer{i}'] = {'initial_stock':initial_stocks['retailers'][i], 'stock_capacity':stock_capacities['retailers'][i], 'stock_cost':stock_costs,
                                    'initial_shipments':initial_shipments['retailers'][i],
                                    'last_level':True}

        return nodes_info

    def __init__(self, 
                 nodes_per_echelon=3, # pode ser ser uma lista com quantidade por estágio
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
        
        if isinstance(nodes_per_echelon, int):
            nodes_per_echelon = [nodes_per_echelon]*4
        nodes_per_echelon = {'suppliers':nodes_per_echelon[0],
                             'factories':nodes_per_echelon[1],
                             'wholesalers':nodes_per_echelon[2],
                             'retailers':nodes_per_echelon[3]}

        if not stock_capacities:
            stock_capacities={'suppliers':[[1600]*num_products]*nodes_per_echelon['suppliers'],
                              'factories':[[6400]*num_products]*nodes_per_echelon['factories'],
                              'wholesalers':[[1600]*num_products]*nodes_per_echelon['wholesalers'],
                              'retailers':[[1600]*num_products]*nodes_per_echelon['retailers']}
        
        if not initial_stocks: 
            initial_stocks={'suppliers':[[800]*num_products]*nodes_per_echelon['suppliers'],
                            'factories':[[800]*num_products]*nodes_per_echelon['factories'],
                            'wholesalers':[[800]*num_products]*nodes_per_echelon['wholesalers'],
                            'retailers':[[800]*num_products]*nodes_per_echelon['retailers']}

        if not initial_supply:
            initial_supply=[[[600]*avg_leadtime]*num_products]*nodes_per_echelon['suppliers']

        if not supply_capacities: 
            supply_capacities=[[600]*num_products]*nodes_per_echelon['suppliers']

        if not supply_costs: 
            supply_costs=[[6]*num_products]*nodes_per_echelon['suppliers']

        if not dest_cost: 
            dest_cost={'suppliers':[[2]*nodes_per_echelon['factories']]*num_products,
                       'factories':[[2]*nodes_per_echelon['wholesalers']]*num_products,
                       'wholesalers':[[2]*nodes_per_echelon['retailers']]*num_products,}
            
            [[2]*2]*num_products
            
        if not ship_capacity:             
            ship_capacity={'suppliers':[500*num_products]*nodes_per_echelon['factories'],
                           'factories':[500*num_products]*nodes_per_echelon['wholesalers'],
                           'wholesalers':[500*num_products]*nodes_per_echelon['retailers']}

        if not initial_shipments:
            initial_shipments={'factories':[[[600]*avg_leadtime]*num_products]*nodes_per_echelon['factories'],
                               'wholesalers':[[[240]*avg_leadtime]*num_products]*nodes_per_echelon['wholesalers'],
                               'retailers':[[[240]*avg_leadtime]*num_products]*nodes_per_echelon['retailers']}
        
        if not processing_capacities: 
            processing_capacities=[840*num_products]*nodes_per_echelon['factories']

        if not processing_costs: 
            processing_costs=[[12]*num_products]*nodes_per_echelon['factories']

        nodes_info = self._create_chain(nodes_per_echelon, initial_stocks, stock_capacities, stock_costs, initial_supply,
                                        supply_capacities, supply_costs, dest_cost, ship_capacity, initial_shipments,
                                        processing_capacities, processing_costs)

        super().__init__(nodes_info, num_products=num_products, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                         exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, demand_range=demand_range,
                         demand_std=demand_std, demand_sen_peaks=demand_sen_peaks, avg_demand_range=avg_demand_range,
                         total_time_steps=total_time_steps, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime, 
                         seed=seed, build_info=build_info, demand_perturb_norm=demand_perturb_norm)
