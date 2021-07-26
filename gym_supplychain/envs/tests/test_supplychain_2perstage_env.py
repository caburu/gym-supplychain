import os
import numpy as np

# TODO: Testar dinâmica da cadeia sazonal (com leadtime)

from ..supplychain_2perstage_env import SupplyChain2perStageEnv, SupplyChain2perStageSeasonalEnv
from .utils import check_rewards, check_build_info

class TestSupplyChain2perStageEnv:
    def _data_folder(self):
        return os.path.dirname(os.path.abspath(__file__))+'/data/'

    def test_initial_stocks(self):
        """ Testa a capacidade, o custo e o valor do estoque ao resetar o ambiente estão corretos.
        """
        env = SupplyChain2perStageEnv()

        env.seed(0)
        env.reset()
        for id, node in enumerate(env.nodes):
            assert sum(node.stock) == 0
            if id % 2 == 0:
                assert node.stock_capacities[0] == 200
            else:
                assert node.stock_capacities[0] == 300
            assert sum(node.stock_cost) == 1
    
    # @pytest.mark.skip
    def test_chain_dynamics(self):
        env = SupplyChain2perStageEnv(total_time_steps=5, ship_capacity=250, build_info=True)
        
        
        env.seed(0)
        obs = env.reset() # timestep=0
        rewards = 0

        assert np.allclose(obs, [ 0.  , -1.  , -1.  ,  0.  ,  0.  , -1.  , -0.2 , -0.2 , -1.  ,
                                 -0.76, -0.76, -1.  , -0.76, -0.76, -1.  , -0.92, -0.92, -1.  ,
                                 -0.92, -0.92, -1.  , -0.92, -0.92, -1.  , -0.92, -0.92,  1.  ])

        assert np.allclose(env.customer_demands.flatten(), [15, 10, 13, 13, 17, 19, 13, 15, 12, 14, 17, 16])

        for node in env.nodes[:4]:
            assert node.shipments_by_prod[0] == [(1,60), (2,60)]
        for node in env.nodes[4:]:
            assert node.shipments_by_prod[0] == [(1,20), (2,20)]

        # ação para fornecer o máximo de material possível no primeiro fornecedor
        supply_action = np.array([1]+[0]*(env.action_space.shape[0]-1))
        supply_action = 2*supply_action-1

        obs, rew, _, info = env.step(supply_action) # timestep=1
        rewards += rew
        check_rewards(rewards, info, env.num_products)

        assert np.allclose(obs, [-0.4       , -0.4       , -0.4       ,  0.        ,  1.        ,
                                -0.6       , -0.2       , -1.        , -0.4       , -0.76      ,
                                -1.        , -0.6       , -0.76      , -1.        , -0.8       ,
                                -0.92      , -1.        , -0.86666667, -0.92      , -1.        ,
                                -0.95      , -0.92      , -1.        , -0.93333333, -0.92      ,
                                -1.        ,  0.6       ])
        assert rew == -1015.0

        assert env.nodes[0].shipments_by_prod[0] == [(2,60),(3,120)]
        assert sum(env.nodes[0].stock) == 60
        for i, node in enumerate(env.nodes[1:4]):
            assert node.shipments_by_prod[0] == [(2,60)]
            if i <= 2:
                assert sum(node.stock) == 60
            else:
                assert sum(node.stock) == 20
        for node in env.nodes[4:6]:
            assert node.shipments_by_prod[0] == [(2,20)]
            assert sum(node.stock) == 20
        for i, node in enumerate(env.nodes[-2:]):
            assert sum(node.stock) == 20 - env.customer_demands[0,i]

        # ação não fornecer material possível e enviar o máximo de material possível
        send_all_action = np.array([0,1,1]*2+[1]*2*4)
        send_all_action = 2*send_all_action-1
        
        obs, rew, _, info = env.step(send_all_action) # timestep=2
        rewards += rew
        check_rewards(rewards, info, env.num_products)
        
        assert np.allclose(obs, [ 0.4       ,  0.8       , -1.        ,  1.        , -1.        ,
                                -1.        , -1.        , -1.        , -1.        , -1.        ,
                                -0.04      , -1.        , -1.        , -1.        , -1.        ,
                                -1.        , -0.68      , -1.        , -1.        , -1.        ,
                                -0.88      , -1.        , -0.68      , -0.88666667, -1.        ,
                                -1.        ,  0.2       ])
        assert rew == -3469.0

        assert env.nodes[0].shipments_by_prod[0] == [(3,120)]
        assert env.nodes[1].shipments_by_prod[0] == []
        assert env.nodes[2].shipments_by_prod[0] == [(4,120), (4,120)]
        assert env.nodes[3].shipments_by_prod[0] == []
        assert env.nodes[4].shipments_by_prod[0] == [(4,40), (4,40)]
        assert env.nodes[5].shipments_by_prod[0] == []
        assert env.nodes[6].shipments_by_prod[0] == [(4,40), (4,40)]
        assert env.nodes[7].shipments_by_prod[0] == []

        for node in env.nodes[:-2]:
            assert sum(node.stock) == 0
        assert sum(env.nodes[-2].stock) == 12
        assert sum(env.nodes[-1].stock) == 17
        
        # ação não fornecer material possível e enviar metade do material para cada destino
        send_half_action = np.array([0,0.5,1]*2+[0.5,1]*4)
        send_half_action = 2*send_half_action-1
        obs, rew, _, info = env.step(send_half_action) # timestep=3
        rewards += rew
        check_rewards(rewards, info, env.num_products)

        assert np.allclose(obs, [-0.4 ,  0.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                -0.04, -0.76, -1.  , -1.  , -0.76, -1.  , -0.68, -1.  , -1.  ,
                                -1.  , -1.  , -1.  , -0.68, -1.  , -1.  , -1.  , -1.  , -0.2 ])
        assert rew == -1752.0
        
        for i in [0,1,5,7]:
            assert env.nodes[i].shipments_by_prod[0] == []
        assert env.nodes[2].shipments_by_prod[0] == [(4,120), (4,120), (5,60)]
        assert env.nodes[3].shipments_by_prod[0] == [(5,60)]
        assert env.nodes[4].shipments_by_prod[0] == [(4,40), (4,40)]
        assert env.nodes[6].shipments_by_prod[0] == [(4,40), (4,40)]        
        for node in env.nodes:
            assert sum(node.stock) == 0
        
        obs, rew, _, info = env.step(send_half_action) # timestep=4
        rewards += rew
        check_rewards(rewards, info, env.num_products)

        assert np.allclose(obs, [-0.6 , -0.2 , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                -0.76, -1.  , -1.  , -0.76, -1.  , -1.  , -1.  , -0.86666667, -1.  ,
                                -1.  , -0.86666667, -0.33, -1.  , -0.84, -1.  , -1.  , -0.84, -0.6 ])
        assert np.round(rew,3) == -6400.333
        
        for node in env.nodes[:2]:
            assert node.shipments_by_prod[0] == []
        for node in env.nodes[2:4]:
            assert node.shipments_by_prod[0] == [(5,60)]
        for node in env.nodes[4:6]:
            assert np.allclose(node.shipments_by_prod[0], [(6,33.333)])
        for node in env.nodes[6:]:
            assert node.shipments_by_prod[0] == [(6,40)]
        for node in env.nodes[:-2]:
            assert node.stock == 0
        assert sum(env.nodes[-2].stock) == 67
        assert sum(env.nodes[-1].stock) == 0
        
        obs, rew, done, info = env.step(send_half_action) # timestep=5
        rewards += rew
        check_rewards(rewards, info, env.num_products)

        assert np.allclose(obs, [ 0.4 ,  0.2 , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                 -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -0.86666667, -0.92, -1.  ,
                                 -0.86666667, -0.92, -0.45, -0.84, -1.  , -1.  , -0.84, -1.  , -1.  ])
        assert rew == -4479.0
        assert done == True
        
        for node in env.nodes[:4]:
            assert node.shipments_by_prod[0] == []
        for node in env.nodes[4:6]:
            assert np.allclose(node.shipments_by_prod[0], [(6,33.333), (7,10), (7,10)])
        for node in env.nodes[-2:0]:
            assert node.shipments_by_prod[0] == [(6,40)]
        for node in env.nodes[:-2]:
            assert sum(node.stock) == 0
        assert sum(env.nodes[-2].stock) == 55
        assert sum(env.nodes[-1].stock) == 0

    def test_demands_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageEnv()

        demands = np.load(self._data_folder()+'demands_2perstage.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[10*seed+epis] == env.customer_demands[:360])
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_demands_consecutive_episodes_stocleadtimes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageEnv(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4)

        demands = np.load(self._data_folder()+'demands_2perstage_stocleadtimes.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[10*seed+epis] == env.customer_demands[:360])
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_leadtimes_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageEnv(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4)

        # Trecho para salvar os leadtimes:
        # 
        lt = np.zeros((100,360,env.count_leadtimes_per_timestep))
        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                lt[10*seed+epis] = env.leadtimes.copy()
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())
        np.save(self._data_folder()+'leadtimes_2perstage', lt)

        leadtimes = np.load(self._data_folder()+'leadtimes_2perstage.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(leadtimes[10*seed+epis] == env.leadtimes)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

class TestSupplyChain2perStageSeasonalEnv:
    def _data_folder(self):
        return os.path.dirname(os.path.abspath(__file__))+'/data/'

    def test_initial_stocks(self):
        """ Testa a capacidade, o custo e o valor do estoque ao resetar o ambiente estão corretos.
        """
        env = SupplyChain2perStageSeasonalEnv()

        env.seed(0)
        env.reset()
        for id, node in enumerate(env.nodes):
            assert sum(node.stock) == 800
            if id % 2 == 0:
                assert node.stock_capacities[0] == 1600
            else:
                assert node.stock_capacities[0] == 1800
            assert node.stock_cost[0] == 1

    def test_demands_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageSeasonalEnv()

        # Trecho para salvar as demandas:
        # 
        # d = np.zeros((100,360,2))
        # for seed in range(seeds):
        #     env.seed(seed+1)
        #     for epis in range(episodes):
        #         env.reset()
        #         d[10*seed+epis] = env.customer_demands[:360].copy()
        #         done = False
        #         while not done:
        #             _, _, done, _ = env.step(env.action_space.sample())
        # np.save(self._data_folder()+'demands_2perstageSeasonal', d)

        demands = np.load(self._data_folder()+'demands_2perstageSeasonal.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[10*seed+epis] == env.customer_demands[:360])
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_demands_consecutive_episodes_stocleadtimes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageSeasonalEnv(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4)

        demands = np.load(self._data_folder()+'demands_2perstageSeasonal_stocleadtimes.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(demands[10*seed+epis] == env.customer_demands[:360])
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())

    def test_leadtimes_consecutive_episodes(self):
        
        seeds = 10
        episodes = 10

        env = SupplyChain2perStageSeasonalEnv(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4)

        # Trecho para salvar os leadtimes:
        # 
        # lt = np.zeros((100,360,env.count_leadtimes_per_timestep))
        # for seed in range(seeds):
        #     env.seed(seed+1)
        #     for epis in range(episodes):
        #         env.reset()
        #         lt[10*seed+epis] = env.leadtimes.copy()
        #         done = False
        #         while not done:
        #             _, _, done, _ = env.step(env.action_space.sample())
        # np.save(self._data_folder()+'leadtimes_2perstageSeasonal', lt)

        leadtimes = np.load(self._data_folder()+'leadtimes_2perstageSeasonal.npy')

        for seed in range(seeds):
            env.seed(seed+1)
            for epis in range(episodes):
                env.reset()
                assert np.all(leadtimes[10*seed+epis] == env.leadtimes)
                done = False
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())
                
    def test_build_info(self):
        env = SupplyChain2perStageSeasonalEnv(
                stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4,
                demand_perturb_norm=True, build_info=True)
        check_build_info(env)


