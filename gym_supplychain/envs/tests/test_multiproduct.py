import numpy as np
from ..supplychain_env import SupplyChainEnv
from .utils import check_build_info

class TestMultiproduct():

    def _create_simple_chain(self, num_products=1, initial_stock=[10,20], stock_capacity=[100,200], stock_cost=[1,2], 
                             dest_cost=[[2],[3]], supply_cost=[5,10], supply_capacity=[50,50], 
                             processing_cost=[10,20], processing_capacity=50, ship_capacity=100):
        nodes_info = {}
        nodes_info['Supplier'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                    'destinations':['Factory'], 'dest_costs':dest_cost, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Factory'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                    'destinations':['Wholesal'], 'dest_costs':dest_cost, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Wholesal'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'destinations':['Retailer'], 'dest_costs':dest_cost, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Retailer'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'last_level':True}
        return nodes_info
    
    def _create_env(self, total_time_steps=5, build_info=False):
        num_products = 2
        nodes_info = self._create_simple_chain(num_products=num_products, initial_stock=[10,20], stock_capacity=[100,200], 
                             stock_cost=[1,2], dest_cost=[[2],[3]], supply_cost=[5,10],
                             supply_capacity=[50,50], processing_cost=[10,20], processing_capacity=50, ship_capacity=100)

        env = SupplyChainEnv(nodes_info, num_products=num_products, unmet_demand_cost=1000, 
                 exceeded_stock_capacity_cost=101, exceeded_process_capacity_cost=102, exceeded_ship_capacity_cost=103,
                 demand_range=(0,5), demand_std=None, demand_sen_peaks=None, avg_demand_range=None, 
                 processing_ratio=2, stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2,
                 total_time_steps=total_time_steps, seed=None, build_info=build_info, demand_perturb_norm=False)
        
        return env

    def test_initial_stocks(self):
        """ Testa a capacidade, o custo e o valor do estoque ao resetar o ambiente estão corretos.
        """
        env = self._create_env()

        env.seed(0)
        env.reset()
        for node in env.nodes:
            assert node.stock == [10,20]
            assert node.stock_capacities == [100,200]
            assert node.stock_cost == [1,2]

    def test_simpleenv(self):
        env = self._create_env()   

        env.seed(0)
        env.reset() # timestep=0

        assert np.all(env.customer_demands.flatten() == [4, 5, 0, 3, 3, 3, 1, 3, 5, 2, 4, 0])

        for node in env.nodes:
            assert node.shipments_by_prod == [[], []]
        
        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,1,0,0,0,0,0,0])
        supply_action = 2*supply_action-1
        
        env.step(supply_action) # timestep=1

        assert env.nodes[0].shipments_by_prod == [[(3,50.0)],[(3,50.0)]]
        for node in env.nodes[1:]:
            assert node.shipments_by_prod == [[], []]
        for node in env.nodes[:-1]:
            assert np.allclose(node.stock, [10.0,20.0])
        assert np.allclose(env.nodes[-1].stock, [6, 15])

        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(8*[1])
        send_all_action = 2*send_all_action-1
        
        env.step(send_all_action) # timestep=2

        assert env.nodes[0].shipments_by_prod == [[(3,50), (4,50)],[(3,50), (4,50)]]
        assert env.nodes[1].shipments_by_prod == [[(4,10)],[(4,20)]]
        assert env.nodes[2].shipments_by_prod == [[(4,5)],[(4,10)]]
        assert env.nodes[3].shipments_by_prod == [[(4,10)],[(4,20)]]
        for node in env.nodes[:-1]:
            assert np.allclose(node.stock, [0.0, 0.0])
        assert np.allclose(env.nodes[-1].stock, [6,12])
        
        env.step(send_all_action) # timestep=3
        
        assert env.nodes[0].shipments_by_prod == [[(4,50), (5,50)],[(4,50), (5,50)]]
        assert env.nodes[1].shipments_by_prod == [[(4,10), (5,50)],[(4,20), (5,50)]]
        assert env.nodes[2].shipments_by_prod == [[(4, 5)],[(4, 10)]]
        assert env.nodes[3].shipments_by_prod == [[(4,10)],[(4,20)]]
        for node in env.nodes[:-1]:
            assert np.allclose(node.stock, [0.0, 0.0])
        assert np.allclose(env.nodes[-1].stock, [3,9])
        
        env.step(send_all_action) # timestep=4
        
        assert env.nodes[0].shipments_by_prod == [[(5,50), (6,50)],[(5,50), (6,50)]]
        assert env.nodes[1].shipments_by_prod == [[(5,50), (6,50)],[(5,50), (6,50)]]
        assert env.nodes[2].shipments_by_prod == [[(6, 5)],[(6, 10)]]
        assert env.nodes[3].shipments_by_prod == [[(6, 5)],[(6, 10)]]
        for node in env.nodes[:-1]:
            assert np.allclose(node.stock, [0.0, 0.0])
        assert np.allclose(env.nodes[-1].stock, [12,26])
        
        env.step(send_all_action) # timestep=5
        
        assert env.nodes[0].shipments_by_prod == [[(6,50), (7,50)],[(6,50), (7,50)]]        
        assert env.nodes[1].shipments_by_prod == [[(6,50), (7,50)],[(6,50), (7,50)]]
        assert env.nodes[2].shipments_by_prod == [[(6, 5), (7,25)],[(6, 10)]]
        assert env.nodes[3].shipments_by_prod == [[(6, 5)],[(6, 10)]]
        
        assert np.allclose(env.nodes[0].stock, [0.0,  0.0])
        assert np.allclose(env.nodes[1].stock, [0.0, 50.0])
        assert np.allclose(env.nodes[0].stock, [0.0,  0.0])
        
        assert np.allclose(env.nodes[-1].stock, [7,24])

    def test_basic_costs(self):
        env = self._create_env(build_info=True)        

        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,1,0,0,0,0,0,0])
        supply_action = 2*supply_action-1
        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(8*[1])
        send_all_action = 2*send_all_action-1
        
        env.seed(0)
        env.reset() # timestep=0

        env.step(supply_action) # timestep=1
        env.step(send_all_action) # timestep=2        
        env.step(send_all_action) # timestep=3        
        _,_,_,info = env.step(send_all_action) # timestep=4

        units = info['sc_episode']['units']
        costs = info['sc_episode']['costs']

        assert units['stock'] == [57,122]
        assert costs['stock'] == [57,244]

        assert units['stock_pen'] == [0,0]
        assert costs['stock_pen'] == [0,0]

        assert units['supply'] == [200,200]
        assert costs['supply'] == [1000,2000]

        assert units['process'] == [20,40]
        assert costs['process'] == [200,800]

        assert units['process_pen'] == [0,0]
        assert costs['process_pen'] == [0,0]

        assert units['ship'] == [135,170]
        assert costs['ship'] == [270,510]

        assert units['ship_pen'] == [0,0]
        assert costs['ship_pen'] == [0,0]

        assert units['unmet_dem'] == [0,0]
        assert costs['unmet_dem'] == [0,0]
    
    def test_pen_costs(self):
        env = self._create_env(build_info=True)        

        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,1,0,0,0,0,0,0])
        supply_action = 2*supply_action-1
        
        env.seed(0)
        env.reset() # timestep=0

        env.step(supply_action) # timestep=1
        env.step(supply_action) # timestep=2
        env.step(supply_action) # timestep=3
        _,_,_,info = env.step(supply_action) # timestep=4

        units = info['sc_episode']['units']
        costs = info['sc_episode']['costs']

        assert units['stock_pen'] == [10,0]
        assert costs['stock_pen'] == [101*10,0]

        assert np.allclose(env.nodes[0].stock, [100,120])

        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(8*[1])
        send_all_action = 2*send_all_action-1

        _,_,_,info = env.step(send_all_action) # timestep=5

        units = info['sc_episode']['units']
        costs = info['sc_episode']['costs']

        assert units['ship_pen'] == [0,70]
        assert costs['ship_pen'] == [0,103*70]

        assert np.allclose(env.nodes[0].stock, [0,70])

        assert env.nodes[1].shipments_by_prod == [[(7,100)],[(7,100)]]

        assert units['unmet_dem'] == [3,0]
        assert costs['unmet_dem'] == [3*1000,0]

    def test_processpen_costs(self):
        env = self._create_env(total_time_steps=6, build_info=True)        

        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,1,0,0,0,0,0,0])
        supply_action = 2*supply_action-1
        # ação para fornecer o máximo de material possível e enviar o máximo de material possível dos fornecedores
        supplier_full_action = np.array([1,1,1,1,0,0,0,0])
        supplier_full_action = 2*supplier_full_action-1
        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(8*[1])
        send_all_action = 2*send_all_action-1
        
        env.seed(0)
        env.reset() # timestep=0

        env.step(supply_action) # timestep=1
        env.step(supplier_full_action) # timestep=2
        env.step(supplier_full_action) # timestep=3
        env.step(supplier_full_action) # timestep=4
        env.step(supplier_full_action) # timestep=5
        _,_,_,info = env.step(send_all_action) # timestep=6

        units = info['sc_episode']['units']
        costs = info['sc_episode']['costs']

        assert units['process_pen'] == [50,140]
        assert costs['process_pen'] == [102*50,102*140]