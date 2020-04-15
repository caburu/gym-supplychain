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
            # As ações de envio de material são a porcentagem do estoque a ser
            # enviada para aquele destino, atendendo um por vez. Exemplo: se
            # para o primeiro destino o valor for 0.8, então 80% do que tiver no
            # estoque é enviado para aquele destino; se para o segundo destino o
            # valor for 0.3, então 30% do que sobrou no estoque é enviado para
            # aquele destino, e assim, sucessivamente. O que sobrar de material
            # fica estocado.
            # Obs.: lembrando que é necessário tratar arredondamentos.
            if self.capacity is None:
                limit = max
            else:
                limit = min(self.capacity, max)
            returns = []
            costs = []
            for i in range(len(action_values)):
                rounded_amount = round(action_values[i]*limit)
                if rounded_amount > limit:
                    rounded_amount = limit
                returns.append(rounded_amount)
                costs.append(rounded_amount*self.costs[i])
                limit -= rounded_amount

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
                 stock_cost=0.0, supply_cost=0.0):
        self.label = label
        self.supply_action = None
        self.ship_action = None
        self.num_actions = 0
        if supply_capacity > 0:
            self.supply_action = SC_Action('SUPPLY', capacity=supply_capacity, costs=supply_cost)
            self.num_actions += 1
        self.last_level = last_level
        self.initial_stock = initial_stock
        self.stock_cost = stock_cost
        self.dest = None
        self.shipments = deque()

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

        debug = ''
        next_action_idx = 0
        # O próximo passo é executar as ações referentes ao nó da cadeia
        if self.supply_action:
            amount, cost = self.supply_action.apply(action_values[next_action_idx])
            next_action_idx += 1
            if amount > 0:
                self.shipments.appendleft((time_step+leadtime, amount))
            debug += str(amount)+'='+str(cost)+' + '
            total_cost += cost

        if self.ship_action:
            amounts, costs = self.ship_action.apply(action_values[next_action_idx:], max=self.stock)
            self.stock -= sum(amounts)
            for i in range(len(self.dests)):
                if amounts[i] > 0:
                    self.dests[i]._ship_material(time_step+leadtime, amounts[i])
            debug += str(sum(amounts))+'='+str(sum(costs))+' + '
            total_cost += sum(costs)

        total_cost += self.stock*self.stock_cost
        debug += str(self.stock)+'='+str(self.stock*self.stock_cost)+' + '
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

    def is_last_level():
        return self.last_level

    def build_observation(self, shipments_range):
        """ Observação é [estoque, ship1, ship2, ...]
            Onde ship1 é a soma dos carregamentos que chegam no próximo período,
            ship2 no período seguinte, e assim por diante.
        """
        obs = [self.stock]

        # Se não tem nenhum carregamento pra chegar
        if not self.shipments:
            obs += [0]*(shipments_range[1]-shipments_range[0]+1)
            return obs
        else:
            ship_idx = -1
            for time_step in shipments_range:
                obs.append(0)
                while ship_idx >= -len(self.shipments) and self.shipments[ship_idx][0] == time_step:
                    obs[-1] += self.shipments[ship_idx][1]
                    ship_idx -= 1
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
    def __init__(self, nodes={}, demand_range=(0,10), leadtime=1, unmet_demand_cost=1.0, total_time_steps=100, seed=None):
        # TODO: definir valores iniciais de estoque
        # Estrutura a receber de entrada
        #   nodes={
        #     'Supplier 1': {
        #       'label':,
        #       'initial_stock':20,
        #       'stock_capacity':1000,
        #       'supply_capacity':0,
        #       'last_level':False,
        #       'stock_cost':0.01,
        #       'supply_cost':0.0,
        #       'destinations':['Factory 1', 'Factory 2'],
        #       'dest_costs': [0.03, 0.03]
        #     }
        #   }
        sup1 = SC_Node('Supplier 1', initial_stock=20, stock_capacity=1000, stock_cost=0.01, supply_capacity=100, supply_cost=0.05)
        sup2 = SC_Node('Supplier 2', initial_stock=0, stock_capacity=1000, stock_cost=0.01, supply_capacity=100, supply_cost=0.05)
        fac1 = SC_Node('Factory  1', initial_stock=20, stock_capacity=1000, stock_cost=0.01)
        fac2 = SC_Node('Factory  2', initial_stock=0, stock_capacity=1000, stock_cost=0.01)
        who1 = SC_Node('Wholesal 1', initial_stock=20, stock_capacity=1000, stock_cost=0.01)
        who2 = SC_Node('Wholesal 2', initial_stock=0, stock_capacity=1000, stock_cost=0.01)
        ret1 = SC_Node('Retailer 1', initial_stock=10, stock_capacity=1000, stock_cost=0.01, last_level=True)
        ret2 = SC_Node('Retailer 2', initial_stock=20, stock_capacity=1000, stock_cost=0.01, last_level=True)
        sup1.define_destinations([fac1, fac2], [0.02, 0.02])
        sup2.define_destinations([fac1, fac2], [0.02, 0.02])
        fac1.define_destinations([who1, who2], [0.02, 0.02])
        fac2.define_destinations([who1, who2], [0.02, 0.02])
        who1.define_destinations([ret1, ret2], [0.02, 0.02])
        who2.define_destinations([ret1, ret2], [0.02, 0.02])
        self.nodes = [sup1, sup2, fac1, fac2, who1, who2, ret1, ret2]
        self.last_level_nodes = [ret1, ret2]
        self.total_time_steps = total_time_steps
        self.leadtime = leadtime
        self.rand_generator = np.random.RandomState(seed)
        self.demand_range = demand_range
        self.unmet_demand_cost = unmet_demand_cost

        # Definição dos espaços de ações e de estados
        action_space_size = 0
        for node in self.nodes:
            action_space_size += node.num_expected_actions()
        obs_space_size = len(self.nodes)*(1+leadtime)
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(action_space_size,))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,))

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.time_step = 0
        self.customer_demands = []

        self.current_reward = 0.0
        self.current_state = self._build_observation()

        return self.current_state

    def step(self, action):
        self.time_step += 1
        self.customer_demands = self.rand_generator.randint(low=self.demand_range[0],
                             high=self.demand_range[1], size=len(self.last_level_nodes))

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

        self.current_reward = -total_cost
        self.current_state = self._build_observation()

        is_terminal = self.time_step == self.total_time_steps

        return (self.current_state, self.current_reward, is_terminal, {})

    def _build_observation(self):
        obs = []
        for node in self.nodes:
            obs += node.build_observation((self.time_step+1, self.time_step+self.leadtime))
        return obs

    def render(self, mode='human'):
        print('TIMESTEP:', self.time_step)
        print('customer demands:', self.customer_demands)
        for node in self.nodes:
            node.render()
            print()
        print('Current state:', self.current_state)
        print('Current reward:', round(self.current_reward,2))
        print('='*20)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)

if __name__ == '__main__':
    env = SupplyChainEnv(total_time_steps=5, leadtime=2 )
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

# TODO: tratar parametrização do ambiente
# TODO: tratar fábricas (transformação de matéria-prima em produto)
#
# TODO: representação de ações SHIP é muito desbalanceada (Isso é um problema?)
#       Para mandar a mesma quantidade de material para todos os destinos os valores das ações são muito díspares.
#
# TODO: se for usar leadtimes variáveis tem que repensar representação do estado
