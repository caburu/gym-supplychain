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

    def __init__(self, num_products=2,
                 initial_stocks=[[800,600]]*8, 
                 stock_capacities=[[1600]*2]+[[1800]*2]+[[6400]*2,[7200]*2]+[[1600]*2]+[[1800]*2]+[[1600]*2]+[[1800]*2],
                 stock_costs=1,
                 initial_supply=[[[600]*2]*2]+[[[840]*2]*2], 
                 supply_capacities=[[600]*2]+[[840]*2],
                 supply_costs=[[6]*2]+[[4]*2],
                 dest_cost=[[2]*2]*2,
                 ship_capacity=[2000,2000],
                 initial_shipments=[[[600]*2]*2,[[840]*2]*2]+[[[240]*2]*2]*4,
                 processing_capacities=[1600,2000], 
                 processing_costs=[[12]*2,[10]*2],
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

        nodes_info = self._create_chain(initial_stocks, stock_capacities, stock_costs, initial_supply,
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