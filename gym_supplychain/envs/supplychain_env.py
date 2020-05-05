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
            supplied_amount = round(action_values*limit)
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
                    rounded_amount = round((final_cut-initial_cut)*limit)
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
    def __init__(self, label, initial_stock=0, stock_capacity=0, supply_capacity=0, last_level=False,
                 stock_cost=0.0, supply_cost=0.0, exceeded_capacity_cost=1.0):
        self.label = label
        self.supply_action = None
        self.ship_action = None
        self.num_actions = 0
        if supply_capacity > 0:
            self.supply_action = SC_Action('SUPPLY', capacity=supply_capacity, costs=supply_cost)
            self.num_actions += 1
        self.last_level = last_level
        self.initial_stock = initial_stock
        self.stock_capacity = stock_capacity
        self.stock_cost = stock_cost
        self.exceeded_capacity_cost = exceeded_capacity_cost
        self.dest = None
        self.shipments = deque()
        
        # Atributos estatísticos para depuração
        self.est_stock_cost = 0
        self.est_stock_penalty = 0
        self.est_supply_cost = 0
        self.est_ship_cost = 0

    def define_destinations(self, dests, costs):
        self.dests = dests
        self.ship_action = SC_Action('SHIP', costs=costs)
        self.num_actions += len(self.dests)

    def act(self, action_values, leadtime, time_step):
        total_cost = 0

        # O primeiro passo é receber o material que está pra chegar
        while self.shipments and  self.shipments[-1][0] == time_step:
            ship = self.shipments.pop()
            self.stock += ship[1]
            if self.stock > self.stock_capacity:
                total_cost += self.exceeded_capacity_cost*(self.stock - self.stock_capacity)
                self.stock = self.stock_capacity                
                self.est_stock_penalty = self.exceeded_capacity_cost*(self.stock - self.stock_capacity)

        #debug = ''
        next_action_idx = 0
        # O próximo passo é executar as ações referentes ao nó da cadeia
        if self.supply_action:
            amount, cost = self.supply_action.apply(action_values[next_action_idx])
            next_action_idx += 1
            if amount > 0:
                self.shipments.appendleft((time_step+leadtime, amount))
            #debug += str(amount)+'='+str(cost)+' + '
            total_cost += cost            
            self.est_supply_cost = cost

        if self.ship_action:
            amounts, costs = self.ship_action.apply(action_values[next_action_idx:], max=self.stock)
            self.stock -= sum(amounts)
            for i in range(len(self.dests)):
                if amounts[i] > 0:
                    self.dests[i]._ship_material(time_step+leadtime, amounts[i])
            #debug += str(sum(amounts))+'='+str(sum(costs))+' + '
            total_cost += sum(costs)            
            self.est_ship_cost = sum(costs)

        total_cost += self.stock*self.stock_cost
        self.est_stock_cost = self.stock*self.stock_cost

        #debug += str(self.stock)+'='+str(self.stock*self.stock_cost)+' + '
        #print('CUSTO', self.label, debug)

        return total_cost

    def meet_demand(self, amount):
        # Se o nó é de último nível atende à demanda do cliente
        assert self.last_level

        max_possible = min(self.stock, amount)
        self.stock -= max_possible

        return max_possible

    def _ship_material(self, time, amount):
        self.shipments.appendleft((time,amount))

    def reset(self):
        self.stock = self.initial_stock
        self.shipments.clear()

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
        obs = [self.stock/self.stock_capacity]

        # Se não tem nenhum carregamento pra chegar
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
        return desc

class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}
    def __init__(self, nodes_info, unmet_demand_cost=1.0, exceeded_capacity_cost=1.0,
                 demand_range=(0,11), leadtime=1, total_time_steps=1000, seed=None):
        def create_nodes(nodes_info):
            nodes_dict = {}
            self.nodes = []
            self.last_level_nodes = []
            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                node = SC_Node(node_name,
                               initial_stock=node_info.get('initial_stock', 0),
                               stock_capacity=node_info.get('stock_capacity', float('inf')),
                               supply_capacity=node_info.get('supply_capacity', 0),
                               last_level=node_info.get('last_level', False),
                               stock_cost=node_info.get('stock_cost', 0.0),
                               supply_cost=node_info.get('supply_cost', 0.0),
                               exceeded_capacity_cost=exceeded_capacity_cost)
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
        self.unmet_demand_cost = unmet_demand_cost

        # Definição dos espaços de ações e de estados
        action_space_size = 0
        for node in self.nodes:
            action_space_size += node.num_expected_actions()
        obs_space_size = len(self.last_level_nodes)+len(self.nodes)*(1+leadtime)

        # O action_space é tratado como de [0,1] no código, então quando a ação é recebida o valor
        # é desnormalizado
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(action_space_size,))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,))

        self.current_state = None

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.time_step = 0
        self.customer_demands = []

        self.current_reward = 0.0
        # definindo as demandas do próximo período
        self.customer_demands = self.rand_generator.randint(low=self.demand_range[0],
                                                            high=self.demand_range[1], 
                                                            size=len(self.last_level_nodes))
        
        # Prefixo 'est' indica que é uma estatística. Ou seja, é útil para monitoramento
        # mas não é necessário para funcionamento.
        self.est_unmet_demands = np.zeros(len(self.last_level_nodes))
        
        self.current_state = self._build_observation()

        return self.current_state
        
    def _denormalize_action(self, action):
        return (action+1)/2

    def step(self, action):
        action = self._denormalize_action(action)
        
        self.time_step += 1        

        total_cost = 0.0

        next_action_idx = 0
        for node in self.nodes:
            actions_to_apply = action[next_action_idx:next_action_idx+node.num_expected_actions()]
            next_action_idx += node.num_expected_actions()
            cost = node.act(actions_to_apply, self.leadtime, self.time_step)
            total_cost += cost

        for i in range(len(self.last_level_nodes)):
            met_dem = self.last_level_nodes[i].meet_demand(self.customer_demands[i])            
            total_cost += self.unmet_demand_cost * (self.customer_demands[i] - met_dem)
            self.est_unmet_demands[i] = (self.customer_demands[i] - met_dem)

        # definindo as demandas do próximo período
        self.customer_demands = self.rand_generator.randint(low=self.demand_range[0],
                                                    high=self.demand_range[1], 
                                                    size=len(self.last_level_nodes))

        self.current_reward = -total_cost
        self.current_state = self._build_observation()

        is_terminal = self.time_step == self.total_time_steps

        return (self.current_state, self.current_reward, is_terminal, {})

    def _build_observation(self):
        """ Uma observação será formada:
            - Pelas demandas dos clientes
            - Pela obervação de cada nó. Que tem:
                - A quantidade de material em estoque.
                - O quanto de material está chegando nos períodos seguintes.
        """ 
        nodes_obs = []
        for node in self.nodes:
            nodes_obs += node.build_observation((self.time_step+1, self.time_step+self.leadtime))
        demands_obs = self.customer_demands/(self.demand_range[1]-1)
        obs = np.concatenate((demands_obs, nodes_obs))
        return obs

    def render(self, mode='human'):
        print('TIMESTEP:', self.time_step)        
        for node in self.nodes:
            node.render()
            print()
        print('Next demands  :', self.customer_demands)
        print('Current state :', self.current_state)
        print('Current reward:', round(self.current_reward,2))
        print('='*30)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)

if __name__ == '__main__':
    stock_capacity  = 1000
    supply_capacity = 20
    stock_cost  = 0.001
    supply_cost = 0.005
    dest_cost   = 0.002
    nodes_info = {}
    nodes_info['Supplier 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Supplier 2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  1'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 2'] = {'initial_stock':15, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Retailer 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}
    nodes_info['Retailer 2'] = {'initial_stock':20, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}

    env = SupplyChainEnv(nodes_info, unmet_demand_cost=0.1, exceeded_capacity_cost=1.0, total_time_steps=5, leadtime=2)
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

