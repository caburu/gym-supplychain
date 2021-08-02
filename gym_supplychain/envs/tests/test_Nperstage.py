import numpy as np

from ..supplychain_env import SupplyChainEnv
from ..supplychain_Nperstage_env import SupplyChainNPerStage

# TODO: teste atuais NperStage são apenas de execução direta (faltam validações detalhadas)

class TestNPerStage():
    
    def _run_episode(self, env, seed=0, expected_rewards=None):
        env.seed(seed)
        env.reset()
        done = False
        rewards = 0
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            rewards += reward
        
        if expected_rewards:
            assert np.allclose(expected_rewards, rewards)

    def test3perStage(self):
        env = SupplyChainNPerStage(nodes_per_echelon=3)
        self._run_episode(env, expected_rewards=-60038768.011493534)

    def test3perStageSeasonalDemands(self):
        env = SupplyChainNPerStage(nodes_per_echelon=3,
                                   demand_std=60,
                                   demand_sen_peaks=4,
                                   avg_demand_range=(100,300),
                                   demand_perturb_norm=True,)
        self._run_episode(env, expected_rewards=-57730855.89812181)

    def test3perStage3Products(self):
        env = SupplyChainNPerStage(nodes_per_echelon=3, num_products=3)
        self._run_episode(env, expected_rewards=-88943757.80027954)

    def test10perStage(self):
        env = SupplyChainNPerStage(nodes_per_echelon=10)
        self._run_episode(env, expected_rewards=-197097090.01279718)
    
    def testChain_3_2_3_5(self):
        env = SupplyChainNPerStage(nodes_per_echelon=[3,2,3,5])
        self._run_episode(env, expected_rewards=-120404116.66453858)
    
    def testChain_5_4_7_10(self):
        env = SupplyChainNPerStage(nodes_per_echelon=[5,4,7,10])
        self._run_episode(env, expected_rewards=-251255147.76827675)
    
    def testChain_5_4_7_10_and_4products(self):
        env = SupplyChainNPerStage(nodes_per_echelon=[5,4,7,10], num_products=4)
        self._run_episode(env, expected_rewards=-501101931.2484466)