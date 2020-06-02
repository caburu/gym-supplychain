import numpy as np
from collections import deque

import gym
from gym import spaces

class SC_Action:
    """ Define uma ação do ambiente da Cadeia de Suprimentos.

        As ações podem ser dos seguintes tipos:
        - SUPPLY: decide quanto fornecer de material para o próprio nó da cadeia
        - SHIP: decide quanto enviar de material para cada possível destino.

        Essa classe tem por objetivo converter um valor de ação real (entre 0 e 1)
        em um valor inteiro correspondente a ser realmente aplicado para a aquela ação.
    """
    def __init__(self, action_type, capacity=None, costs=None):
        """ Cria a ação, armazenando o tipo passado

            :param action_type: (str) Define o tipo da ação
            :param capacity: (int) Define o valor máximo da ação. Se `None` (padrão) não existe valor máximo.
        """
        # confere se o tipo de ação passado é válido
        assert action_type in ['SUPPLY', 'SHIP']

        # confere se ação de fornecimento e estoque tem capacidade válida e ação de envio tem destino válidos
        if action_type == 'SUPPLY':
            assert capacity is not None

        self.action_type = action_type
        self.capacity = capacity
        self.costs = costs

    def apply(self, action_values, max=None):
        """ Aplica o valor de ação recebido, podendo ter um limite variável.

            :param action_value: (float or list) valor (ou lista de valores) entre 0 e 1 (porcentagem) referente à ação a ser aplicada.
            :param max: (int) valor máximo da ação para essa decisão específica.
            :return: (int or list) retorna o(s) valor(es) inteiro(s) correspondente(s) à ação.
        """
        if self.action_type == 'SUPPLY':
            # No caso de fornecimento basta usar o valor percentual no máximo de
            # valor inteiro possível (mínimo entre capacidade e máximo recebido)
            if max is None:
                limit = self.capacity
            else:
                limit = min(self.capacity, max)
            supplied_amount = int(round(action_values*limit))
            return supplied_amount, supplied_amount*self.costs
        elif self.action_type == 'SHIP':
            
            # Definindo a quantidade máxima de material disponível para envio
            if self.capacity is None:
                limit = max
            else:
                limit = min(self.capacity, max)
            available = limit
                        
            # Inicializando listas a serem retornadas (quantidades e custos)
            returns = [0]*len(action_values)
            costs = [0]*len(action_values)
                
            # Se tem material para ser enviado
            if available > 0:
                # formando tuplas com os valores das ações e o índice do destino
                tuples = [(action_values[i], i) for i in range(len(action_values))]
                # ordenando as tuplas pelos valores das ações
                tuples.sort()
                
                # o primeiro corte será de zero até o primeiro valor
                initial_cut = 0
                # Para cada valor e destino
                for value, idx in tuples:
                    # o corte vai até o valor da ação desse destino
                    final_cut = value 
                    # precisamos arredondar a quantidade de material pois ela é inteira
                    rounded_amount = int(round((final_cut-initial_cut)*limit))
                    # tratamento para evitar problemas de arredondamento
                    if rounded_amount > available:
                        rounded_amount = available
                    # guardando a quantidad (e o custo) referentes a o destino em questão
                    returns[idx]= rounded_amount
                    costs[idx] = rounded_amount*self.costs[idx]
                    # descontamos agora a quantidade de material disponível
                    available -= rounded_amount
                    # e o próximo corte começa onde terminou esse
                    initial_cut = final_cut

            return returns, costs
        else:
            raise NotImplementedError('Unknown action type:' + self.action_type)

    def get_type(self):
        return self.action_type

class SC_Node:
    """ Define um nó da Cadeia de Suprimentos

        Cada nó tem:
        - Um identificador.
        - Uma capacidade de estoque.
        - Uma capacidade de fornecimento.
        - Uma indicação se é nó do último nível (estágio).
        - Uma quantidade de material em estoque.
        - Uma lista de ações possíveis.
        - O número total de valores de ação esperados.
        - Uma lista de destinos.
        - Uma fila de material para chegar.
    """
    def __init__(self, label, initial_stock=0, initial_supply=None, initial_shipments=None, 
                 stock_capacity=0, supply_capacity=0, processing_ratio=0,
                 last_level=False, stock_cost=0, supply_cost=0, processing_cost=0, 
                 exceeded_capacity_cost=1,unmet_demand_cost=1):
        self.label = label
        self.supply_action = None
        self.ship_action = None
        self.num_actions = 0
        if supply_capacity > 0:
            self.supply_action = SC_Action('SUPPLY', capacity=supply_capacity, costs=supply_cost)
            self.num_actions += 1
        self.last_level = last_level
        
        self.initial_stock = initial_stock
        self.initial_supply = initial_supply
        self.initial_shipments = initial_shipments
        
        self.stock_capacity = stock_capacity
        self.stock_cost = stock_cost
        self.exceeded_capacity_cost = exceeded_capacity_cost
        self.unmet_demand_cost = unmet_demand_cost
        
        self.processing_ratio = processing_ratio
        self.processing_cost = processing_cost
        self.pending_material = 0
        self.dest = None
        self.shipments = deque()        

    def define_destinations(self, dests, costs):
        self.dests = dests
        self.ship_action = SC_Action('SHIP', costs=costs)
        self.num_actions += len(self.dests)

    def act(self, action_values, leadtime, time_step, customer_demand=None):
        total_cost = 0

        arrived_material = 0
        # O primeiro passo é receber o material que está pra chegar
        while self.shipments and  self.shipments[-1][0] == time_step:
            _, amount = self.shipments.pop()
            arrived_material += amount
            
        # Se é uma fábrica, é necessário processar a matéria-prima, gerando produto
        if self.processing_ratio > 0:
            # A matéria-prima que chegou é somada com matéria-prima anterior que
            # pode ter sobrado como resto do processamento
            material = arrived_material+self.pending_material
            amount = material // self.processing_ratio
            # Calcula o novo resto do processamento
            self.pending_material = material % self.processing_ratio
            # Calcula o custo de processamento (calculado por unidade de matéria-prima usada)
            self.est_costs['processing'] = (material-self.pending_material) * self.processing_cost
            self.est_units['processing'] = (material-self.pending_material)
            total_cost += self.est_costs['processing']            
            self.stock += amount
        else:
            self.stock += arrived_material
            
        if self.stock > self.stock_capacity:
            self.est_costs['stock_penalty'] = self.exceeded_capacity_cost*(self.stock - self.stock_capacity)
            self.est_units['stock_penalty'] = self.stock - self.stock_capacity
            total_cost += self.est_costs['stock_penalty']
            self.stock = self.stock_capacity
        else:
            self.est_costs['stock_penalty'] = 0
            self.est_units['stock_penalty'] = 0
                
        #debug = ''
        next_action_idx = 0
        # O próximo passo é executar as ações referentes ao nó da cadeia
        if self.supply_action:
            amount, cost = self.supply_action.apply(action_values[next_action_idx])
            next_action_idx += 1
            if amount > 0:
                self.shipments.appendleft((time_step+leadtime, amount))
            #debug += str(amount)+'='+str(cost)+' + '
            self.est_costs['supply'] = cost
            self.est_units['supply'] = amount
            total_cost += self.est_costs['supply']                   

        if self.ship_action:
            amounts, costs = self.ship_action.apply(action_values[next_action_idx:], max=self.stock)
            self.stock -= sum(amounts)
            for i in range(len(self.dests)):
                if amounts[i] > 0:
                    self.dests[i]._ship_material(time_step+leadtime, amounts[i])
            #debug += str(sum(amounts))+'='+str(sum(costs))+' + '
            self.est_costs['ship'] = sum(costs)
            self.est_units['ship'] = sum(amounts)
            total_cost += self.est_costs['ship']  
            
        elif self.last_level:
            max_possible = min(self.stock, customer_demand)
            self.stock -= max_possible
            self.est_costs['unmet_dem'] = self.unmet_demand_cost * (customer_demand - max_possible)
            self.est_units['unmet_dem'] = customer_demand - max_possible
            total_cost += self.est_costs['unmet_dem']     

        self.est_costs['stock'] = self.stock*self.stock_cost
        self.est_units['stock'] = self.stock
        total_cost += self.est_costs['stock']

        #debug += str(self.stock)+'='+str(self.stock*self.stock_cost)+' + '
        #print('CUSTO', self.label, debug)

        return total_cost

    def _ship_material(self, time, amount):
        self.shipments.appendleft((time,amount))

    def reset(self):
        self.stock = self.initial_stock
        self.shipments.clear()
        if self.initial_supply:
            for i in range(len(self.initial_supply)):
                self.shipments.appendleft((i+1, self.initial_supply[i]))
        if self.initial_shipments:
            for i in range(len(self.initial_shipments)):
                self.shipments.appendleft((i+1, self.initial_shipments[i]))
        
        if self.processing_ratio > 0:
            self.pending_material = 0
        
        # Atributos estatísticos para depuração
        self.est_costs = {'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'ship':0, 'unmet_dem':0}
        self.est_units = {'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'ship':0, 'unmet_dem':0} 

    def num_expected_actions(self):
        return self.num_actions

    def render(self):
        print(self.__str__(), end='')

    def is_last_level(self):
        return self.last_level

    def build_observation(self, shipments_range):
        """ Observação é [estoque, ship1, ship2, ...]
            Onde ship1 é a soma dos carregamentos que chegam no próximo período,
            ship2 no período seguinte, e assim por diante.

            Os valores são normalizados, ou seja, o valor de estoque é a porcentagem
            da capacidade do estoque.
            Os valores dos carregamentos, por enquanto, serão dados também como porcentagem
            da capacidade do estoque, assumindo como esse o limite superior para transporte.
        """
        # a primeira informação é o estoque atual.
        # - No caso das fábricas inclui a fração de matéria-prima pendente (se existir)
        current_stock = self.stock
        if self.processing_ratio > 0:
            current_stock += self.pending_material/self.processing_ratio

        obs = [current_stock/self.stock_capacity]

        # Se não tem nenhum carregamento pra chegar
        # Cria os carregamentos vazios
        if not self.shipments:
            obs += [0]*(shipments_range[1]-shipments_range[0]+1)
            return obs
        else:
            ship_idx = -1
            for time_step in range(shipments_range[0], shipments_range[1]+1):
                obs.append(0)
                while ship_idx >= -len(self.shipments) and self.shipments[ship_idx][0] == time_step:
                    obs[-1] += self.shipments[ship_idx][1]
                    ship_idx -= 1
                obs[-1] /= self.stock_capacity
            return obs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        desc = self.label + ' '
        for i in range(len(self.shipments)):
            desc += str(self.shipments[i]) + ' - '
        desc += '[' + str(self.stock) + '] '
        if self.processing_ratio > 0: desc += '+' + str(self.pending_material) + ' '
        return desc

class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}
    def __init__(self, nodes_info, unmet_demand_cost=1000, exceeded_capacity_cost=1000,
                 demand_range=(10,21), processing_ratio=3, leadtime=2, total_time_steps=360, seed=None):
                 
        self.DEBUG = True
        
        def create_nodes(nodes_info):
            nodes_dict = {}
            self.nodes = []
            self.last_level_nodes = []
            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                
                processing_cost = node_info.get('processing_cost', 0)
                if processing_cost > 0:
                    node_processing_ratio = processing_ratio
                else:
                    node_processing_ratio = 0
                    
                node = SC_Node(node_name,
                               initial_stock=node_info.get('initial_stock', 0),
                               initial_supply=node_info.get('initial_supply', None),
                               initial_shipments=node_info.get('initial_shipments', None),
                               stock_capacity=node_info.get('stock_capacity', float('inf')),
                               supply_capacity=node_info.get('supply_capacity', 0),
                               processing_ratio=node_processing_ratio,
                               last_level=node_info.get('last_level', False),
                               stock_cost=node_info.get('stock_cost', 0),
                               supply_cost=node_info.get('supply_cost', 0),
                               processing_cost=processing_cost,
                               exceeded_capacity_cost=exceeded_capacity_cost,
                               unmet_demand_cost=unmet_demand_cost)
                nodes_dict[node_name] = node
                self.nodes.append(node)
                if node.is_last_level():
                    self.last_level_nodes.append(node)

            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                node = nodes_dict[node_name]
                if 'destinations' in node_info:
                    node.define_destinations([nodes_dict[dest_name] for dest_name in node_info['destinations']],
                                             node_info['dest_costs'])

        create_nodes(nodes_info)
        self.total_time_steps = total_time_steps
        self.leadtime = leadtime
        self.rand_generator = np.random.RandomState(seed)
        self.demand_range = demand_range

        # Definição dos espaços de ações e de estados
        action_space_size = 0
        for node in self.nodes:
            action_space_size += node.num_expected_actions()
        obs_space_size = len(self.last_level_nodes)+len(self.nodes)*(1+leadtime)

        # O action_space é tratado como de [0,1] no código, então quando a ação é recebida o valor
        # é desnormalizado
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(action_space_size,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_space_size,))        

        self.current_state = None

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.time_step = 0

        self.current_reward = 0
        self.episode_rewards = 0
        # definindo as demandas do próximo período
        self.customer_demands = self.rand_generator.randint(low=self.demand_range[0],
                                                            high=self.demand_range[1], 
                                                            size=len(self.last_level_nodes))        
        
        self.current_state = self._build_observation()

        # Estatísticas do episódio
        self.est_episode = self._initial_est_episode()

        return self.current_state
    
    def _initial_est_episode(self):
        return {'rewards':0, 
                'costs':{
                    'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'ship':0, 'unmet_dem':0},
                'units':{
                    'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'ship':0, 'unmet_dem':0}
                }
        
    def _denormalize_action(self, action):
        return (action+1)/2
        
    def _normalize_obs(self, obs):
        return obs*2 - 1

    def step(self, action):
        action = self._denormalize_action(action)
        
        self.time_step += 1        

        total_cost = 0

        next_action_idx = 0
        next_customer = 0
        for node in self.nodes:
            actions_to_apply = action[next_action_idx:next_action_idx+node.num_expected_actions()]
            next_action_idx += node.num_expected_actions()
            if node.last_level:
                demand = self.customer_demands[next_customer]
                next_customer += 1
            else:
                demand = None
            cost = node.act(actions_to_apply, self.leadtime, self.time_step, customer_demand=demand)
            total_cost += cost

        # definindo as demandas do próximo período
        self.customer_demands = self.rand_generator.randint(low=self.demand_range[0],
                                                    high=self.demand_range[1], 
                                                    size=len(self.last_level_nodes))

        self.current_reward = -total_cost
        self.episode_rewards += self.current_reward
        self.current_state = self._build_observation()

        is_terminal = self.time_step == self.total_time_steps

        self._update_statistics()
        
        self.current_info = self._build_return_info()
        
        #print(self.customer_demands[0], ',', self.customer_demands[1], ',', end=' ')

        return (self.current_state, self.current_reward, is_terminal, self.current_info)

    def _update_statistics(self):
        self.est_episode['rewards'] += self.current_reward
        costs = self.est_episode['costs']
        units = self.est_episode['units']
        for node in self.nodes:
            for key,value in node.est_costs.items():
                costs[key] += value
            for key,value in node.est_units.items():
                units[key] += value

    def _build_observation(self):
        """ Uma observação será formada:
            - Pelas demandas dos clientes
            - Pela obervação de cada nó. Que tem:
                - A quantidade de material em estoque.
                - O quanto de material está chegando nos períodos seguintes.
        """ 
        # Primeiro guardamos as demandas (normalizadas)
        if self.demand_range[0] != self.demand_range[1]-1:
            demands_obs = (self.customer_demands - self.demand_range[0])/(self.demand_range[1]-self.demand_range[0]-1)
        else:
            demands_obs = [0]*len(self.customer_demands)
            
        # Depois pegamos os dados de cada nó
        nodes_obs = []
        for node in self.nodes:
            nodes_obs += node.build_observation((self.time_step+1, self.time_step+self.leadtime))        

        # A observação tem a concatenação das demandas com os dados nos nós
        obs = np.concatenate((demands_obs, nodes_obs))

        # Por fim, acrescentamos quantos períodos faltam para terminar o episódio (normalizado)
        obs = np.append(obs, (self.total_time_steps-self.time_step)/self.total_time_steps)
        
        # Por fim, normalizamos o estado para a faixa [-1,1]
        norm_obs = self._normalize_obs(obs)

        return norm_obs

    def _build_return_info(self):
        # O prefixo 'sc_' serve pra identificar as chaves que se refere ao ambiente
        # porque a biblioteca Stable Baselines inclui informação adicional
        return {'sc_episode':self.est_episode}

    def render(self, mode='human'):
        print('TIMESTEP:', self.time_step)
        for node in self.nodes:
            node.render()
            print()                 
        print('Next demands  :', self.customer_demands)
        print('Current state :', self.current_state)
        print('Current reward:', round(self.current_reward,3))
        #if self.DEBUG:
        #    for node in self.nodes:
        #        print('Custos:', node.est_costs)
        print('='*30)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)

if __name__ == '__main__':
    demand_range     = (10,21)
    stock_capacity   = 300
    supply_capacity  = 50
    processing_ratio = 3
    leadtime    = 2
    stock_cost  = 1
    dest_cost   = 2*stock_cost
    supply_cost = 5*stock_cost
    processing_cost   = 2*supply_cost
    # Quanto custa para produzir e entregar uma unidade de produto (sem usar estoque)
    # Obs: na verdade custo de transporte seria 2*leadtime*cost, porque não paga transporte no fornecimento
    product_cost = supply_cost + 3*leadtime*dest_cost + processing_cost 
    # O custo de demanda não atendida é duas vezes o custo de produzir (como se comprasse do concorrente).
    unmet_demand_cost = 2*product_cost
    # O custo de excesso de estoque talvez pudesse nem existir, já que o custo já incorrido no material
    # é perdido. Mas podemos considerar também que existiria um custo de desfazer do material.
    exceeded_capacity_cost = 10*stock_cost
    
    total_time_steps = 5
    
    nodes_info = {}
    nodes_info['Supplier 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Supplier 2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  1'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 2'] = {'initial_stock':15, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Retailer 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}
    nodes_info['Retailer 2'] = {'initial_stock':20, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}

    env = SupplyChainEnv(nodes_info, demand_range=demand_range, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_capacity_cost=exceeded_capacity_cost, processing_ratio=processing_ratio, 
                         leadtime=leadtime, total_time_steps=total_time_steps)
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

