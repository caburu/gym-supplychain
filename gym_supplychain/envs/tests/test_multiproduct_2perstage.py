import numpy as np

from ..supplychain_env import SupplyChainEnv

class TestMultiproduct2PerStage():

    def _create_chain(self):
        """ Cria uma cadeia com a configuração bem diversa """
        nodes_info = {}

        nodes_info['Supplier1'] = {'initial_stock':[11,1], 'stock_capacity':[20,10], 'stock_cost':[1,2],
                                    'initial_supply':[[1,4],[2,3]], 'supply_capacity':[50,60], 'supply_cost':[10,11],
                                    'destinations':['Factory1','Factory2'], 'dest_costs':[[1,2],[0,1]], 
                                    'ship_capacity':[100,101]}

        nodes_info['Supplier2'] = {'initial_stock':[12,2], 'stock_capacity':[21,11], 'stock_cost':[3,4],
                                    'initial_supply':[[3,1],[4,2]], 'supply_capacity':[100,110], 'supply_cost':[20,11],
                                    'destinations':['Factory1','Factory2'], 'dest_costs':[[3,4],[2,3]], 
                                    'ship_capacity':[102,103]}

        nodes_info['Factory1'] = {'initial_stock':[13,3], 'stock_capacity':[22,12], 'stock_cost':[3,4],
                                    'initial_shipments':[[1,2],[3,4]],
                                    'processing_capacity':40, 'processing_cost':[15,16],
                                    'destinations':['Wholesal1','Wholesal2'], 'dest_costs':[[5,6],[4,5]], 
                                    'ship_capacity':[104,105]}

        nodes_info['Factory2'] = {'initial_stock':[14,4], 'stock_capacity':[23,13], 'stock_cost':[1,2],
                                    'initial_shipments':[[4,3],[2,1]],
                                    'processing_capacity':30, 'processing_cost':[20,21],
                                    'destinations':['Wholesal1','Wholesal2'], 'dest_costs':[[7,8],[6,7]], 
                                    'ship_capacity':[106,107]}

        nodes_info['Wholesal1'] = {'initial_stock':[15,5], 'stock_capacity':[24,14], 'stock_cost':[5,6],
                                    'initial_shipments':[[5,6],[7,8]],
                                    'destinations':['Retailer1','Retailer2'], 'dest_costs':[[9,10],[8,9]],
                                    'ship_capacity':[108,109]}

        nodes_info['Wholesal2'] = {'initial_stock':[16,6], 'stock_capacity':[25,15], 'stock_cost':[6,5],
                                    'initial_shipments':[[8,7],[6,5]],
                                    'destinations':['Retailer1','Retailer2'], 'dest_costs':[[11,12],[10,11]],
                                    'ship_capacity':[110,111]}

        nodes_info['Retailer1'] = {'initial_stock':[17,7], 'stock_capacity':[26,16], 'stock_cost':[7,8],
                                    'initial_shipments':[[0,5],[10,15]],
                                    'last_level':True}

        nodes_info['Retailer2'] = {'initial_stock':[18,8], 'stock_capacity':[27,17], 'stock_cost':[8,7],
                                    'initial_shipments':[[15,10],[5,0]],
                                    'last_level':True}

        return nodes_info

    def _create_env(self, total_time_steps=5, build_info=False):
        num_products = 2

        nodes_info = self._create_chain()

        env = SupplyChainEnv(nodes_info, num_products=num_products, unmet_demand_cost=100, 
                 exceeded_stock_capacity_cost=101, exceeded_process_capacity_cost=102, exceeded_ship_capacity_cost=103,
                 demand_range=(0,100), demand_std=None, demand_sen_peaks=None, avg_demand_range=None, 
                 processing_ratio=[2,3], stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2,
                 total_time_steps=total_time_steps, seed=None, build_info=build_info, demand_perturb_norm=False)
        
        return env
    
    def test_basic_dynamics(self):
        env = self._create_env()

        env.seed(0)
        env.reset()

        assert np.all(env.customer_demands[:2].flatten() == [44,47,64,67,67,9,83,21])

        # ação para fornecer e enviar metade do material possível
        half_action = np.array(8*[0.5])
        half_action = 2*half_action-1

        env.step(half_action) # timestep=1

        # conferir estoques
        assert np.allclose(env.nodes[0].stock, [6, 1.5])
        assert np.allclose(env.nodes[1].stock, [7.5, 3])
        assert np.allclose(env.nodes[2].stock, [7, 2.5])
        assert np.allclose(env.nodes[3].stock, [9, 3.5])
        assert np.allclose(env.nodes[4].stock, [10, 5.5])
        assert np.allclose(env.nodes[5].stock, [12, 6.5])
        assert np.allclose(env.nodes[6].stock, [8.5, 6])
        assert np.allclose(env.nodes[7].stock, [16.5, 9])

        # conferir shipments

        # conferir custos

        # conferir estado