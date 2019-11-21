import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SupplyChainEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def array_to_int(self, array, num_codes):
        code = 0
        for i in range(len(array)):
            code += array[i]*(num_codes**i)
        return code

    def int_to_array(self, code, num_levels, num_codes):
        array = [0]*num_levels
        for i in range(num_levels-1,0,-1):
            array[i] = code // num_codes**i
            code = code % num_codes**i
        array[0] = code
        return array

    def observation(self, state):
        return self.array_to_int(state, self.state_codes)

    def decode_actions(self, action):
        return self.int_to_array(action, self.levels, self.action_codes)

    def code_state(self, inventory):
        state = [0]*len(inventory)
        for i in range(len(inventory)):
            if inventory[i] < -6:
                state[i] = 0
            elif inventory[i] < -3:
                state[i] = 1
            elif inventory[i] < 0:
                state[i] = 2
            elif inventory[i] < 3:
                state[i] = 3
            elif inventory[i] < 6:
                state[i] = 4
            elif inventory[i] < 10:
                state[i] = 5
            elif inventory[i] < 15:
                state[i] = 6
            elif inventory[i] < 20:
                state[i] = 7
            else:
                state[i] = 8
        return np.asarray(state)


    def __init__(self, env_init_info={}):
        '''Initial inventory is a list with the initial inventory position for each level
        '''

        # Número de níveis da cadeia
        self.levels = 4
        # A CODIFICAÇÃO DE ESTADOS E AÇÕES DEVERIA ESTAR NO AGENTE E NÃO NO AMBIENTE
        self.state_codes = 9
        self.action_codes = 4
        self.inv_cost = 1
        self.backlog_cost = 2

        # Nos vetores, a posição zero é do retailer

        # Quantidade de itens em cada estoque

        # Demanda dos clientes a cada semana
        self.customer_demand = np.asarray(env_init_info['customer_demand'])
        # Quantidade inicial de estoque em cada nível
        self.initial_inventory = 12 + np.zeros(self.levels)
        # Número de semanas a simular
        self.max_weeks = len(self.customer_demand)
        # Máximo leadtime de entrega   ## MUDAR PARA O CASO DO ARTIGO
        self.max_ship_delay = 2
        # Valor inicial de ítens em transporte
        self.initial_shipment_values = 4
        # Por quanto tempo essas unidades em transporte vão chegar
        self.initial_shipment_times = 2
        # Pedidos colocados inicialmente
        self.initial_orders_value = 4


        # Estrutura para guardar todas as entregas. Por tempo, por nível.
        self.initial_shipment = np.zeros((self.max_weeks+self.max_ship_delay+1, self.levels), dtype=int)
        # Tratando as entregas pendentes já no momento inicial
        self.initial_shipment[1:1+self.initial_shipment_times][:] = self.initial_shipment_values

        # Pedidos colocados para o nível acima
        self.initial_orders_placed = self.initial_orders_value + np.zeros(self.levels, dtype=int)
        # Pedidos pendentes que chegaram do nível abaixo (posição zero é a demanda do cliente,
        # e valor colocado aqui é ignorado depois)
        self.initial_incoming_orders = self.initial_orders_value + np.zeros(self.levels, dtype=int)

        self.current_state = None

        # Tratando variáveis do OpenAI Gym environment (# Ver uso do MultiDiscrete)
        self.action_space = spaces.Discrete(self.action_codes**self.levels)
        self.observation_space = spaces.Discrete(self.state_codes**self.levels)


    def step(self, action):
        Y_actions = np.asarray(self.decode_actions(action))
        #Y_actions = np.asarray(self.decode_actions(0))

        self.week += 1

        # 1. Receive inventory and advance shipment delay

        # Os estoques recebem o que estava previsto nas entregas
        self.inventory += self.shipments[self.week][:]
        # Não é necessário "avançar" as entregas, pois estamos guardando todas
        # no vetor (por tempo)

        # 2. Fill orders

        # Retailer obtém a demanda do cliente
        self.incoming_orders[0] = self.customer_demand[self.week-1]

        # Valores negativos do estoque indicam o backlog.
        # logo, tratamos aqui as entregas que chegam (e o backlog está indiretamente
        # tratado no recebimento do passo anterior e aqui)

        # Faz entrega
        self.shipments[self.week+self.max_ship_delay][:-1] =  \
            np.maximum(np.zeros(self.levels-1, dtype=int),
                       np.minimum(self.inventory[1:], self.incoming_orders[1:]))
        #print('shipments incoming orders:\n',self.shipments)
        # Desconta do estoque
        self.inventory -= self.incoming_orders


        # 3. Record the invetory or backlog

        # Já feito :)

        # 4. Advance the order slips

        # Movendo os pedidos colocados para os pedidos chegando
        # Obs: a informação do cliente não é atualizada aqui (valor fica lixo
        #        até próximo passo)
        self.incoming_orders[1:] = self.orders_placed[:-1]

        # Pedidos para a fábrica (último nível) são colocados para entrega
        self.shipments[self.week+self.max_ship_delay][-1] = self.orders_placed[-1]
        #print('shipments factory orders:\n',self.shipments)

        # 5. Place orders

        # cada nível passa para o nível acima um pedido de tamanho X+Y
        # onde X é o que recebeu de demanda e Y é a quantidade a decidir pelo agente

        self.orders_placed = self.incoming_orders + Y_actions

        # 6. Tratando agora as questões de Aprendizado (recompensa e próximo estado)

        self.current_state = self.code_state(self.inventory)

        # A quantidade de itens disponíveis são os valores positivos do estoque
        inventory_onhand = np.clip(self.inventory,0,None)
        # O backlog é a quantidade de itens que falta no estoque (valores negativos)
        backlog = -np.clip(self.inventory,None,0)
        # A recompensa é a soma ponderada dos custos de estoque e backlog
        reward = -np.sum(self.inv_cost*inventory_onhand + self.backlog_cost*backlog)

        is_terminal = self.week == self.max_weeks

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.week = 0
        self.inventory = np.copy(self.initial_inventory)
        self.orders_placed = np.copy(self.initial_orders_placed)
        self.incoming_orders = np.copy(self.initial_incoming_orders)
        self.shipments = np.copy(self.initial_shipment)

        self.current_state = self.code_state(self.inventory)

        return self.observation(self.current_state)

    def render(self, mode='human'):
        print('\n' + '='*20)
        print('Week:\t', self.week)
        print('Inventory:\t', self.inventory)
        print('Incoming order:\t', '[?]', self.incoming_orders[1:])
        print('Orders placed:\t', self.orders_placed)
        if self.week < self.max_weeks:
            print('Next customer demand:\t', self.customer_demand[self.week])
        #print('Print shipments:\n', self.shipments)
        #print('self.shipments[', self.week+1, ':', self.week+self.max_ship_delay+1, ']')
        print('Next shipments:\t', list(self.shipments[self.week+1:self.week+self.max_ship_delay+1]))
        pass

    def close(self):
        pass
