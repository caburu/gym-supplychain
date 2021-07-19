import os
import numpy as np

from ..supplychain_env import SupplyChainEnv
from .utils import check_build_info

class TestSupplyChainEnv:
    def _data_folder(self):
        return os.path.dirname(os.path.abspath(__file__))+'/data/'

    def _create_simple_chain(self, num_products = 1, initial_stock = 0, stock_capacity = 10, stock_cost = 1, dest_cost = 2, supply_cost = 5,
                             supply_capacity = 100, processing_cost = 10, processing_capacity = 100, ship_capacity = 10):
        nodes_info = {}
        nodes_info['Supplier'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                    'destinations':['Factory'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Factory'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                    'destinations':['Wholesal'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Wholesal'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'destinations':['Retailer'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                    'ship_capacity':[ship_capacity]*2}
        nodes_info['Retailer'] = {'initial_stock':initial_stock, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                    'last_level':True}
        return nodes_info
    
    def _create_env(self, num_products=1, stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=4, build_info=False):
        nodes_info = self._create_simple_chain(num_products=num_products, initial_stock=10, stock_capacity=100, 
                             stock_cost=1, dest_cost=2, supply_cost=5,
                             supply_capacity=50, processing_cost=10, processing_capacity=100, ship_capacity = 100)

        env = SupplyChainEnv(nodes_info, num_products=num_products, unmet_demand_cost=1000, 
                 exceeded_stock_capacity_cost=1000, exceeded_process_capacity_cost=1000, exceeded_ship_capacity_cost=1000,
                 demand_range=(0,5), demand_std=None, demand_sen_peaks=None, avg_demand_range=None, 
                 processing_ratio=2, stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime,
                 total_time_steps=5, seed=None, build_info=build_info, demand_perturb_norm=False)
        
        return env

    def _create_simpleenv(self, build_info = False):
        return self._create_env(stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2, build_info=build_info)

    def _create_stocleadtimes_simpleenv(self, build_info = False):
        return self._create_env(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4, build_info=build_info)

    def test_initial_stocks(self):
        """ Testa a capacidade, o custo e o valor do estoque ao resetar o ambiente estão corretos.
        """
        env = self._create_simpleenv()

        env.seed(0)
        env.reset()
        for node in env.nodes:
            assert node.stock == [10]
            assert node.stock_capacities == [100]
            assert node.stock_cost == [1]

    def test_simpleenv(self):
        env = self._create_simpleenv()   

        # testando se ao iniciar o ambiente não há nenhum transporte em andamento
        env.seed(0)
        env.reset() # timestep=0

        assert np.all(env.customer_demands.flatten() == [4, 5, 0, 3, 3, 3])

        for node in env.nodes:
            assert node.shipments_by_prod == [[]]
        
        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,0,0,0,0,0])
        supply_action = 2*supply_action-1
        
        env.step(supply_action) # timestep=1

        assert env.nodes[0].shipments_by_prod[0] == [(3,50)]
        for node in env.nodes[1:]:
            assert len(node.shipments_by_prod[0]) == 0
        for node in env.nodes[:-1]:
            assert node.stock == 10
        assert env.nodes[-1].stock == 10 - env.customer_demands[0]

        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(6*[1])
        send_all_action = 2*send_all_action-1
        
        env.step(send_all_action) # timestep=2

        assert env.nodes[0].shipments_by_prod[0] == [(3,50), (4,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(4,10)]
        assert env.nodes[2].shipments_by_prod[0] == [(4,5)]
        assert env.nodes[3].shipments_by_prod[0] == [(4,10)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=3
        
        assert env.nodes[0].shipments_by_prod[0] == [(4,50), (5,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(4,10), (5,50)]
        assert env.nodes[2].shipments_by_prod[0] == [(4, 5)]
        assert env.nodes[3].shipments_by_prod[0] == [(4,10)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=4
        
        assert env.nodes[0].shipments_by_prod[0] == [(5,50), (6,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(5,50), (6,50)]
        assert env.nodes[2].shipments_by_prod[0] == [(6, 5)]
        assert env.nodes[3].shipments_by_prod[0] == [(6, 5)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10+10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=5
        
        assert env.nodes[0].shipments_by_prod[0] == [(6,50), (7,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(6,50), (7,50)]
        assert env.nodes[2].shipments_by_prod[0] == [(6, 5), (7,25)]
        assert env.nodes[3].shipments_by_prod[0] == [(6, 5)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10+10 - sum(env.customer_demands[:env.time_step]))

    def test_stocleadtimes_simpleenv(self):
        env = self._create_stocleadtimes_simpleenv()        

        # testando se ao iniciar o ambiente não há nenhum transporte em andamento
        env.seed(0)
        env.reset() # timestep=0

        assert np.all(env.customer_demands.flatten() == [4, 5, 0, 3, 3, 3])
        
        leadtimes = env.leadtimes
        
        # como estamos usando semente fixa, os leadtimes devem ser exatamente esses
        assert np.all(leadtimes == [[3, 1, 1, 1],
                                    [3, 2, 1, 2],
                                    [2, 3, 2, 3],
                                    [2, 2, 2, 1],
                                    [1, 1, 1, 2]])

        for node in env.nodes:
            assert len(node.shipments_by_prod[0]) == 0
        
        # ação para fornecer o máximo de material possível
        supply_action = np.array([1,0,0,0,0,0])
        supply_action = 2*supply_action-1
        
        env.step(supply_action) # timestep=1

        assert env.nodes[0].shipments_by_prod[0] == [(4,50)]
        for node in env.nodes[1:]:
            assert len(node.shipments_by_prod[0]) == 0
        for node in env.nodes[:-1]:
            assert node.stock == 10
        assert env.nodes[-1].stock == 10 - env.customer_demands[0]

        # ação para fornecer o máximo de material possível e enviar o máximo de material possível também
        send_all_action = np.array(6*[1])
        send_all_action = 2*send_all_action-1
        
        env.step(send_all_action) # timestep=2

        assert env.nodes[0].shipments_by_prod[0] == [(4,50), (5,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(4,10)]
        assert env.nodes[2].shipments_by_prod[0] == [(3, 5)]
        assert env.nodes[3].shipments_by_prod[0] == [(4,10)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=3
        
        assert env.nodes[0].shipments_by_prod[0] == [(4,50), (5,50), (5,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(4,10)]
        assert env.nodes[2].shipments_by_prod[0] == []
        assert env.nodes[3].shipments_by_prod[0] == [(4,10), (6,5)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=4
        
        assert env.nodes[0].shipments_by_prod[0] == [(5,50), (5,50), (6,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(6,50)]
        assert env.nodes[2].shipments_by_prod[0] == [(6, 5)]
        assert env.nodes[3].shipments_by_prod[0] == [(6, 5)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10+10 - sum(env.customer_demands[:env.time_step]))
        
        env.step(send_all_action) # timestep=5
        
        assert env.nodes[0].shipments_by_prod[0] == [(6,50), (6,50)]
        assert env.nodes[1].shipments_by_prod[0] == [(6,50), (6,100)]
        assert env.nodes[2].shipments_by_prod[0] == [(6, 5),]
        assert env.nodes[3].shipments_by_prod[0] == [(6, 5)]
        for node in env.nodes[:-1]:
            assert node.stock == 0
        assert env.nodes[-1].stock == max(0, 10+10 - sum(env.customer_demands[:env.time_step]))

    def test_demands_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = self._create_simpleenv()

        # Trecho para salvar as demandas:
        # 
        # d = np.zeros((10,10,6,1))
        # for seed in range(seeds):
        #     env.seed(seed)
        #     for epis in range(episodes):
        #         env.reset()
        #         d[seed,epis] = env.customer_demands.copy()
        #         done = False
        #         while not done:
        #             _, _, done, _ = env.step(env.action_space.sample())
        # np.save(self._data_folder()+'demands_simple_stochastic_chain', d)

        demands = np.load(self._data_folder()+'demands_simple_chain.npy')

        for seed in range(seeds):
            env.seed(seed)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[seed,epis] == env.customer_demands)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_demands_consecutive_episodes_stocleadtimes(self):
        
        seeds = 10
        episodes = 10        

        env = self._create_stocleadtimes_simpleenv()

        demands = np.load(self._data_folder()+'demands_simple_chain_stocleadtimes.npy')

        for seed in range(seeds):
            env.seed(seed)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[seed,epis] == env.customer_demands)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_leadtimes_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = self._create_stocleadtimes_simpleenv()

        # Trecho para salvar os leadtimes:
        # 
        # lt = np.zeros((100,5,env.count_leadtimes_per_timestep))
        # for seed in range(seeds):
        #     env.seed(seed)
        #     for epis in range(episodes):
        #         env.reset()
        #         lt[10*seed+epis] = env.leadtimes.copy()
        #         done = False
        #         while not done:
        #             _, _, done, _ = env.step(env.action_space.sample())
        # np.save(self._data_folder()+'leadtimes_simple_chain', lt)

        leadtimes = np.load(self._data_folder()+'leadtimes_simple_chain.npy')

        for seed in range(seeds):
            env.seed(seed)
            for epis in range(episodes):
                env.reset()
                assert np.all(leadtimes[10*seed+epis] == env.leadtimes)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())    
   
    def test_build_info_simpleenv(self):
        env = self._create_simpleenv(build_info=True)
        check_build_info(env)

    def test_build_info_stocleadtimes(self):
        env = self._create_stocleadtimes_simpleenv(build_info=True)
        check_build_info(env)
