import numpy as np
from collections import deque

import gym
from gym import spaces

class SC_Action:
    """ Define uma ação do ambiente da Cadeia de Suprimentos.

        As ações podem ser dos seguintes tipos:
        - SUPPLY: decide quanto fornecer de material para o próprio nó da cadeia
        - STOCK: decide quanto estocar de material no nó da cadeia.
        - SHIP: decide quanto enviar de material para cada possível destino.

        Essa classe tem por objetivo converter um valor de ação real (entre 0 e 1)
        em um valor inteiro correspondente a ser realmente aplicado para a aquela ação.
    """
    def __init__(self, action_type, capacity=None):
        """ Cria a ação, armazenando o tipo passado

            :param action_type: (str) Define o tipo da ação
            :param capacity: (int) Define o valor máximo da ação. Se `None` (padrão) não existe valor máximo.
        """
        # confere se o tipo de ação passado é válido
        assert action_type in ['SUPPLY', 'STOCK', 'SHIP']

        # confere se ação de fornecimento e estoque tem capacidade válida e ação de envio tem destino válidos
        if action_type in ['SUPPLY', 'STOCK']:
            assert capacity is not None

        self.action_type = action_type
        self.capacity = capacity

    def apply(self, action_values, max=None):
        """ Aplica o valor de ação recebido, podendo ter um limite variável.

            :param action_value: (float or list) valor (ou lista de valores) entre 0 e 1 (porcentagem) referente à ação a ser aplicada.
            :param max: (int) valor máximo da ação para essa decisão específica.
            :return: (int or list) retorna o(s) valor(es) inteiro(s) correspondente(s) à ação.
        """
        if self.action_type in ['SUPPLY', 'STOCK']:
            # No caso de fornecimento e estoque basta usar o valor percentual no
            # máximo de valor inteiro possível (mínimo entre capacidade e máximo recebido)
            if max is None:
                limit = self.capacity
            else:
                limit = min(self.capacity, max)
            return np.round(action_values*limit)
        elif self.action_type == 'SHIP':
            # No caso de envio de material, se for 3 destinos, por exemplo, a ação
            # se refere aos dois pontos de corte percentuais. Suponha que sejam
            # 0.2 e 0.5. Nesse caso 20% do máximo possível é enviado para o primeiro
            # destino, 30% para o segundo e 50% para o último.
            if self.capacity is None:
                limit = max
            else:
                limit = min(self.capacity, max)
            returns = [round(value*limit) for value in action_values]
            returns.append(limit - sum(returns))
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
    def __init__(self, ID, stock_capacity=0, supply_capacity=0, last_level=False):
        self.ID = ID
        self.actions = []
        self.num_actions = 0
        if supply_capacity > 0:
            self.actions.append(SC_Action('SUPPLY', capacity=supply_capacity))
            self.num_actions += 1
        if stock_capacity > 0:
            self.actions.append(SC_Action('STOCK', capacity=stock_capacity))
            self.num_actions += 1
        self.last_level = last_level
        self.stock = 0
        self.dest = None
        self.shipments = deque()

    def define_destinations(self, dest):
        self.dests = dest
        self.actions.append(SC_Action('SHIP'))
        self.num_actions += len(self.dests)

    def act(self, action_values, lead_times, time_step):
        # O primeiro passo é receber o material que está pra chegar
        if self.shipments:
            while self.shipments[-1][0] == time_step:
                ship = self.shipments.pop()
                self.stock += ship[1]

        # O próximo passo é executar as ações referentes ao nó da cadeia
        for i in range(len(self.actions)):
            if self.actions[i].get_type() == 'SUPPLY':
                amount = self.actions[i].apply(action_values[i])
                self.shipments.appendleft((time_step+lead_times[i], amount))
            elif  self.actions[i].get_type() == 'STOCK':
                amount = self.actions[i].apply(action_values[i])
                self.stock += amount
            elif  self.actions[i].get_type() == 'SHIP':
                amounts = self.actions[i].apply(action_values[i:i+self.num_actions])
                self.stock -= sum(amount)
                for i in range(self.dests):
                    self.dests[i]._ship_material(time_step+lead_times[i], amount[i])
            else:
                raise NotImplementedError('Unknown action type:' + self.type)

        # Se o nó é de último nível atende à demanda do cliente
        if self.last_level:
            self.stock -= min(self.stock, action_values[-1])


    def _ship_material(self, time, amount):
        self.shipments.appendleft((time,amount))

    def reset(self):
        self.stock = 0
        self.shipments.clear()

    def num_expected_actions(self):
        return self.num_actions

    def render(self):
        print('ID:', self.ID, end=' ')
        for i in range(len(self.shipments)):
            print(self.shipments[i], '-', end=' ')
        print('[', self.stock, ']', sep='', end=' ')


class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}
    def __init__(self, total_time_steps=100):
        node1 = SC_Node(1, stock_capacity=1000, supply_capacity=100, last_level=False)
        node2 = SC_Node(2, last_level=True)
        node1.define_destinations([node2])
        self.nodes = [node1, node2]
        self.total_time_steps = total_time_steps
        self.lead_time = 2

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.time_step = 0

        self.current_state = None

        return self.current_state

    def step(self, action):
        self.time_step += 1

        ## TODO: ver como tratar ação fake de atender a demanda do cliente
        action.append(10)

        # TODO: tratar limites das ações de estoque e envio (não está passando max)

        next_action_idx = 0
        for node in self.nodes:
            actions_to_apply = action[next_action_idx:node.num_expected_actions()]
            next_action_idx = node.num_expected_actions()
            node.act(actions_to_apply, [self.lead_time]*len(actions_to_apply), self.time_step)

        reward = 0
        self.current_state = None

        is_terminal = self.time_step == self.total_time_steps

        return (self.current_state, reward, is_terminal, {})

    def render(self, mode='human'):
        for node in self.nodes:
            node.render()
            print()

if __name__ == '__main__':
    env = SupplyChainEnv(total_time_steps=3)
    env.reset()
    env.render()
    done = False
    while not done:
        action = [10,0,0]
        _, _, done, _ = env.step(action)
        env.render()
