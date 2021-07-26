import numpy as np

from ..supplychain_env import SupplyChainEnv
from ..supplychain_multiproduct_env import SupplyChainMultiProduct

class TestMultiproduct2PerStage():

    def _create_chain(self):
        """ Cria uma cadeia com a configuração bem diversa """
        nodes_info = {}

        nodes_info['Supplier1'] = {'initial_stock':[11,1], 'stock_capacity':[20,10], 'stock_cost':[1,2],
                                    'initial_supply':[[1,4],[2,3]], 'supply_capacity':[50,60], 'supply_cost':[10,11],
                                    'destinations':['Factory1','Factory2'], 'dest_costs':[[1,2],[0,1]], 
                                    'ship_capacity':[100,101]}

        nodes_info['Supplier2'] = {'initial_stock':[12,2], 'stock_capacity':[21,11], 'stock_cost':[3,4],
                                    'initial_supply':[[3,1],[4,2]], 'supply_capacity':[100,110], 'supply_cost':[20,21],
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
    
    def test_basic_dynamics(self):
        env = self._create_env(build_info=True)

        env.seed(0)
        env.reset()

        assert np.all(env.customer_demands[:2].flatten() == [44,47,64,67,67,9,83,21])

        # ação para fornecer e enviar metade do material possível (sendo 1/4 pra cada destino)
        half_action = np.array(2*[0.5,0.5,0.25,0.5,0.25,0.5]+4*[0.25,0.5,0.25,0.5])
        half_action = 2*half_action-1

        obs, reward, _, info = env.step(half_action) # timestep=1

        # conferir estoques
        assert np.allclose(env.nodes[0].stock, [6, 1.5])
        assert np.allclose(env.nodes[1].stock, [7.5, 3])
        assert np.allclose(env.nodes[2].stock, [7, 3])
        assert np.allclose(env.nodes[3].stock, [9, 3])
        assert np.allclose(env.nodes[4].stock, [10, 6])
        assert np.allclose(env.nodes[5].stock, [12, 6])
        assert np.allclose(env.nodes[6].stock, [0, 0])
        assert np.allclose(env.nodes[7].stock, [0, 0])

        # conferir shipments

        assert env.nodes[0].shipments_by_prod == [[(2,4), (3,25)],[(2,3), (3,30)]]
        assert env.nodes[1].shipments_by_prod == [[(2,1), (3,50)],[(2,2), (3,55)]]
        assert env.nodes[2].shipments_by_prod == [[(2,2), (3,3.75), (3,3)], [(2,4), (3,1.5), (3,0.75)]]
        assert env.nodes[3].shipments_by_prod == [[(2,3), (3,3.75), (3,3)], [(2,1), (3,1.5), (3,0.75)]]
        assert env.nodes[4].shipments_by_prod == [[(2,6), (3,2.25), (3,1.75)], [(2,8), (3,0.5), (3,0.5)]]
        assert env.nodes[5].shipments_by_prod == [[(2,7), (3,2.25), (3,1.75)], [(2,5), (3,0.5), (3,0.5)]]
        assert env.nodes[6].shipments_by_prod == [[(2,5), (3,6), (3,5)], [(2,15), (3,3), (3,3)]]
        assert env.nodes[7].shipments_by_prod == [[(2,10), (3,6), (3,5)], [(2,0), (3,3), (3,3)]]

        # conferir custos

        units = info['sc_episode']['units']
        costs = info['sc_episode']['costs']

        assert units['stock'] == [6+7.5+7+9+10+12,
                                  1.5+3+3+3+6+6]
        assert costs['stock'] == [6*1+7.5*3+7*3+9*1+10*5+12*6,
                                  1.5*2+3*4+3*4+3*2+6*6+6*5]

        assert units['stock_pen'] == [6,1]
        assert costs['stock_pen'] == [101*6,101*1]

        assert units['supply'] == [25+50, 30+55]
        assert costs['supply'] == [25*10+50*20, 30*11+55*21]

        assert units['process'] == [7+9, 3+3]
        assert costs['process'] == [7*15+9*20, 3*16+3*21]

        assert units['process_pen'] == [0,0]
        assert costs['process_pen'] == [0,0]

        assert units['ship'] == [3.75+3+3.75+3+2.25+1.75+2.25+1.75+6+5+6+5,
                                 1.5+0.75+1.5+0.75+0.5+0.5+0.5+0.5+3+3+3+3]
        assert costs['ship'] == [3.75*3+3*1+3.75*4+3*2+2.25*7+1.75*5+2.25*8+1.75*6+6*11+5*9+6*12+5*10,
                                 1.5*2+0.75*0+1.5*3+0.75*1+0.5*6+0.5*4+0.5*7+0.5*5+3*10+3*8+3*11+3*9]

        assert units['ship_pen'] == [0,0]
        assert costs['ship_pen'] == [0,0]

        assert units['unmet_dem'] == [44-(17+0) + 64-(18+15-6), 47-(7+10-1) + 67-(8+5)]
        assert costs['unmet_dem'] == [100*units['unmet_dem'][0],100*units['unmet_dem'][1]]

        total_costs = 0
        for key in costs:
            total_costs += sum(costs[key])
        assert reward == -total_costs

        # conferir estado

        expected_obs = [67/100, 9/100, 83/100, 21/100, # demandas do próximo período
                        # Supplier1
                          6/20, 1.5/10, # estoques
                          4/50, 25/50, 3/60, 30/60, # fornecimentos
                        # Supplier2
                          7.5/21, 3/11, # estoques
                          1/100, 50/100, 2/110, 55/110, # fornecimentos
                        # Factory1
                          7/22, 3/12, # estoques
                          2/(100+102), (3+3.75)/(100+102), 4/(100+102), (1.5+0.75)/(100+102), # transportes
                        # Factory2
                          9/23, 3/13, # estoques
                          3/(101+103), (3+3.75)/(101+103), 1/(101+103), (1.5+0.75)/(101+103), # transportes
                        # Wholesal1
                          10/24, 6/14, # estoques
                          6/(104+106), (2.25+1.75)/(104+106), 8/(104+106), (0.5+0.5)/(104+106), # transportes
                        # Wholesal2
                          12/25, 6/15, # estoques
                          7/(105+107), (2.25+1.75)/(105+107), 5/(105+107), (0.5+0.5)/(105+107), # transportes
                        # Retailer1
                          0/26, 0/16, # estoques
                          5/(108+110), (6+5)/(108+110), 15/(108+110), (3+3)/(108+110), # transportes
                        # Retailer2
                          0/27, 0/17, # estoques
                          10/(109+111), (6+5)/(109+111), 0/(109+111), (3+3)/(109+111), # transportes
                        # Tempo para terminar episódio
                        (5-1)/5
                        ]

        expected_obs = 2*np.array(expected_obs) - 1
        assert np.allclose(obs, expected_obs)

        # ação para fornecer e enviar:
        # - Para primeiro produto: todo o material possível (sendo 1/2 pra cada destino)
        # - Para segundo produto: metade do material possível (sendo 1/4 pra cada destino)
        action = np.array(2*[1.0,0.5,0.5,1.0,0.25,0.5]+4*[0.5,1.0,0.25,0.5])
        action = 2*action-1

        obs, reward, _, info = env.step(action) # timestep=2

        # conferir estoques
        assert np.allclose(env.nodes[0].stock, [0, (1.5+3)/2])
        assert np.allclose(env.nodes[1].stock, [0, (3+2)/2])
        assert np.allclose(env.nodes[2].stock, [0, (3+4)/2])
        assert np.allclose(env.nodes[3].stock, [0, (3+1)/2])
        assert np.allclose(env.nodes[4].stock, [0, (6+8)/2])
        assert np.allclose(env.nodes[5].stock, [0, (6+5)/2])
        assert np.allclose(env.nodes[6].stock, [0, 0+15-9])
        assert np.allclose(env.nodes[7].stock, [0, 0])

        # conferir shipments

        assert env.nodes[0].shipments_by_prod == [[(3,25),(4, 50)], [(3,30),(4,30)]]
        assert env.nodes[1].shipments_by_prod == [[(3,50),(4,100)], [(3,55),(4,55)]]
        assert env.nodes[2].shipments_by_prod == [[(3,3),(3,3.75),(4,(7.5+1)/2),(4,(6+4)/2)], [(3,0.75),(3,1.5),(4,(3+2)/4),(4,(1.5+3)/4)]]
        assert env.nodes[3].shipments_by_prod == [[(3,3),(3,3.75),(4,(7.5+1)/2),(4,(6+4)/2)], [(3,0.75),(3,1.5),(4,(3+2)/4),(4,(1.5+3)/4)]]
        assert env.nodes[4].shipments_by_prod == [[(3,1.75),(3,2.25),(4,(9+3)/2/2),(4,(7+2)/2/2)], [(3,0.5),(3,0.5),(4,(3+1)/3/4),(4,(3+4)/3/4)]]
        assert env.nodes[5].shipments_by_prod == [[(3,1.75),(3,2.25),(4,(9+3)/2/2),(4,(7+2)/2/2)], [(3,0.5),(3,0.5),(4,(3+1)/3/4),(4,(3+4)/3/4)]]
        assert env.nodes[6].shipments_by_prod == [[(3,5),(3,6),(4,(12+7)/2),(4,(10+6)/2)], [(3,3),(3,3),(4,(6+5)/4),(4,(6+8)/4)]]
        assert env.nodes[7].shipments_by_prod == [[(3,5),(3,6),(4,(12+7)/2),(4,(10+6)/2)], [(3,3),(3,3),(4,(6+5)/4),(4,(6+8)/4)]]

  
    def test_SupplyChainMultiProduct(self):
        env = SupplyChainMultiProduct()
        self._run_episode(env, expected_rewards=-34704704.078214735)
        
    
    def test_scenario_mp_N20(self):
        env = SupplyChainMultiProduct(demand_range=(0, 400),
                                      avg_demand_range=[100, 300],
                                      demand_std=20,
                                      demand_sen_peaks=4,
                                      demand_perturb_norm=True,
                                      stochastic_leadtimes=True,
                                      avg_leadtime=2,
                                      max_leadtime=4)
        self._run_episode(env, expected_rewards=-33914245.32990393)

    def test_scenario_mp_rN50(self):
        env = SupplyChainMultiProduct(demand_range=(0, 400),
                                      avg_demand_range=[100, 300],
                                      demand_std=50,
                                      demand_perturb_norm=True,
                                      stochastic_leadtimes=True,
                                      avg_leadtime=2,
                                      max_leadtime=4)
        self._run_episode(env, expected_rewards=-33511405.156877503)

      
    def test_SupplyChainMultiProduct_3products(self):
        env = SupplyChainMultiProduct(num_products=3)
        self._run_episode(env, expected_rewards=-52509572.65837007)

    def test_scenario_m3p_N20(self):
        env = SupplyChainMultiProduct(num_products=3,
                                      demand_range=(0, 400),
                                      avg_demand_range=[100, 300],
                                      demand_std=20,
                                      demand_sen_peaks=4,
                                      demand_perturb_norm=True,
                                      stochastic_leadtimes=True,
                                      avg_leadtime=2,
                                      max_leadtime=4)
        self._run_episode(env, expected_rewards=-51585258.57599297)

    def test_scenario_m3p_rN50(self):
        env = SupplyChainMultiProduct(num_products=3,
                                      demand_range=(0, 400),
                                      avg_demand_range=[100, 300],
                                      demand_std=50,
                                      demand_perturb_norm=True,
                                      stochastic_leadtimes=True,
                                      avg_leadtime=2,
                                      max_leadtime=4)
        self._run_episode(env, expected_rewards=-51132357.668103226)
    
    def test_SupplyChainMultiProduct_10products(self):
        env = SupplyChainMultiProduct(num_products=10)
        self._run_episode(env, expected_rewards=-173415102.8513805)