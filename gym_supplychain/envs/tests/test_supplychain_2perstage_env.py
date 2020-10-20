import os
import numpy as np
import pytest

# TODO: Testar dinâmica da cadeia sazonal (com leadtime)
# TODO: Testar caso de descarte por excesso de estoque

from gym_supplychain.envs.supplychain_2perstage_env import SupplyChain2perStageEnv, SupplyChain2perStageSeasonalEnv

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
            assert node.stock == 0
            if id % 2 == 0:
                assert node.stock_capacity == 200
            else:
                assert node.stock_capacity == 300
            assert node.stock_cost == 1
    
    # @pytest.mark.skip
    def test_chain_dynamics(self):
        env = SupplyChain2perStageEnv(total_time_steps=5)
        
        env.seed(0)
        obs = env.reset() # timestep=0

        assert np.allclose(obs, [ 0.  , -1.  , -1.  ,  0.  ,  0.  , -1.  , -0.2 , -0.2 , -1.  ,
                                 -0.76, -0.76, -1.  , -0.76, -0.76, -1.  , -0.92, -0.92, -1.  ,
                                 -0.92, -0.92, -1.  , -0.92, -0.92, -1.  , -0.92, -0.92,  1.  ])

        assert np.allclose(env.customer_demands, [[15, 10], [13, 13], [17, 19], [13, 15], [12, 14], [17, 16]])

        for node in env.nodes[:4]:
            assert node.shipments == [(1,60), (2,60)]
        for node in env.nodes[4:]:
            assert node.shipments == [(1,20), (2,20)]

        # ação para fornecer o máximo de material possível no primeiro fornecedor
        supply_action = np.array([1]+[0]*(env.action_space.shape[0]-1))
        supply_action = 2*supply_action-1

        obs, rew, _, _ = env.step(supply_action) # timestep=1

        assert np.allclose(obs, [-0.4       , -0.4       , -0.4       ,  0.        ,  1.        ,
                                -0.6       , -0.2       , -1.        , -0.8       , -0.76      ,
                                -1.        , -0.86666667, -0.76      , -1.        , -0.8       ,
                                -0.92      , -1.        , -0.86666667, -0.92      , -1.        ,
                                -0.95      , -0.92      , -1.        , -0.93333333, -0.92      ,
                                -1.        ,  0.6       ])
        assert rew == -2255.0

        assert env.nodes[0].shipments == [(2,60),(3,120)]
        assert env.nodes[0].stock == 60
        for i, node in enumerate(env.nodes[1:4]):
            assert node.shipments == [(2,60)]
            if i == 0:
                assert node.stock == 60
            else:
                assert node.stock == 20
        for node in env.nodes[4:6]:
            assert node.shipments == [(2,20)]
            assert node.stock == 20
        for i, node in enumerate(env.nodes[-2:]):
            assert node.stock == 20 - env.customer_demands[0,i]

        # ação não fornecer material possível e enviar o máximo de material possível
        send_all_action = np.array([0,1,1]*2+[1]*2*4)
        send_all_action = 2*send_all_action-1
        
        obs, rew, _, _ = env.step(send_all_action) # timestep=2

        assert np.allclose(obs, [ 0.4       ,  0.8       , -1.        ,  1.        , -1.        ,
                                -1.        , -1.        , -1.        , -1.        , -1.        ,
                                -0.04      , -1.        , -1.        , -1.        , -1.        ,
                                -1.        , -0.68      , -1.        , -1.        , -1.        ,
                                -0.88      , -1.        , -0.68      , -0.88666667, -1.        ,
                                -1.        ,  0.2       ])
        assert rew == -2149.0

        assert env.nodes[0].shipments == [(3,120)]
        assert env.nodes[1].shipments == []
        assert env.nodes[2].shipments == [(4,120), (4,120)]
        assert env.nodes[3].shipments == []
        assert env.nodes[4].shipments == [(4,40), (4,40)]
        assert env.nodes[5].shipments == []
        assert env.nodes[6].shipments == [(4,40), (4,40)]
        assert env.nodes[7].shipments == []

        for node in env.nodes[:-2]:
            assert node.stock == 0
        assert env.nodes[-2].stock == 12
        assert env.nodes[-1].stock == 17
        
        # ação não fornecer material possível e enviar metade do material para cada destino
        send_half_action = np.array([0,0.5,1]*2+[0.5,1]*4)
        send_half_action = 2*send_half_action-1
        obs, rew, _, _ = env.step(send_half_action) # timestep=3

        assert np.allclose(obs, [-0.4 ,  0.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                -0.04, -0.76, -1.  , -1.  , -0.76, -1.  , -0.68, -1.  , -1.  ,
                                -1.  , -1.  , -1.  , -0.68, -1.  , -1.  , -1.  , -1.  , -0.2 ])
        assert rew == -1752.0
        
        for i in [0,1,5,7]:
            assert env.nodes[i].shipments == []
        assert env.nodes[2].shipments == [(4,120), (4,120), (5,60)]
        assert env.nodes[3].shipments == [(5,60)]
        assert env.nodes[4].shipments == [(4,40), (4,40)]
        assert env.nodes[6].shipments == [(4,40), (4,40)]        
        for node in env.nodes:
            assert node.stock == 0
        
        obs, rew, _, _ = env.step(send_half_action) # timestep=4

        assert np.allclose(obs, [-0.6 , -0.2 , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                -0.76, -1.  , -1.  , -0.76, -1.  , -1.  , -1.  , -0.84, -1.  ,
                                -1.  , -0.84, -0.33, -1.  , -0.84, -1.  , -1.  , -0.84, -0.6 ])
        assert rew == -6507.0
        
        for node in env.nodes[:2]:
            assert node.shipments == []
        for node in env.nodes[2:4]:
            assert node.shipments == [(5,60)]        
        for node in env.nodes[4:]:
            assert node.shipments == [(6,40)]
        for node in env.nodes[:-2]:
            assert node.stock == 0
        assert env.nodes[-2].stock == 67
        assert env.nodes[-1].stock == 0
        
        obs, rew, done, _ = env.step(send_half_action) # timestep=5

        assert np.allclose(obs, [ 0.4 ,  0.2 , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -1.  ,
                                 -1.  , -1.  , -1.  , -1.  , -1.  , -1.  , -0.84, -0.92, -1.  ,
                                 -0.84, -0.92, -0.45, -0.84, -1.  , -1.  , -0.84, -1.  , -1.  ])
        assert rew == -4479.0
        assert done == True
        
        for node in env.nodes[:4]:
            assert node.shipments == []
        for node in env.nodes[4:6]:
            assert node.shipments == [(6,40), (7,10), (7,10)]
        for node in env.nodes[-2:0]:
            assert node.shipments == [(6,40)]
        for node in env.nodes[:-2]:
            assert node.stock == 0
        assert env.nodes[-2].stock == 55
        assert env.nodes[-1].stock == 0

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
            assert node.stock == 800
            if id % 2 == 0:
                assert node.stock_capacity == 1600
            else:
                assert node.stock_capacity == 1800
            assert node.stock_cost == 1

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

        env = SupplyChain2perStageEnv(stochastic_leadtimes=True, avg_leadtime=2, max_leadtime=4)

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