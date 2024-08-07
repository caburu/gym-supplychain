import gym
from gym import spaces
import numpy as np

class BeerGameEnv2(gym.Env):
    """ Ver documentação da classe no Notebook em:
        https://github.com/caburu/RL-agents/blob/master/stable-baselines/Ambiente%20BeerGame.ipynb
    """

    def __init__(self, max_stock=100, max_order=30, weeks=35, levels=4,
                    customer_demand=[4]*4 + [8]*31, initial_inventory=[12,12,12,12],
                    inv_cost=1, backlog_cost=2, exceeded_capacity_penalty=100,
                    shipment_delays=2, initial_shipment=4, initial_orders=4,
                    seed=None):
        ''' Build the environment. The default parameters values are from standard beer game
            (except exceeded_capacity_penalty, max_stock and max_order).
        '''
        # Flag para informações de depuração
        self.DEBUG = False

        # Número de níveis da cadeia
        self.levels    = levels
        # Capacidade dos estoques
        self.max_stock = max_stock

        # Definição dos espaços de ações e de estados
        self.action_space      = spaces.MultiDiscrete(self.levels * [max_order])
        self.observation_space = spaces.MultiDiscrete(self.levels * [2*self.max_stock])

        # Custo por unidade em estoque por semana
        self.inv_cost     = inv_cost
        # Custo por unidade em backlog por semana
        self.backlog_cost = backlog_cost
        # Custo de penalização por unidade de produto em estoque ou backlog além da capacidade
        self.exceeded_capacity_penalty = exceeded_capacity_penalty
        
        # Quantidade de semanas a simular (tamanho do episódio)
        self.max_weeks = weeks

        # Demanda dos clientes a cada semana
        if isinstance(customer_demand, tuple) or (isinstance(customer_demand, list) and len(customer_demand)==2):
            self.stochastic_demand_range = customer_demand            
        else:
            self.stochastic_demand_range = None
            self.customer_demand = np.asarray(customer_demand, dtype=int)
        
        # Leadtimes de entrega
        self.stochastic_shipdelays_range = None
        if isinstance(shipment_delays, int):
            self.shipment_delays = np.asarray([2] + self.max_weeks*[shipment_delays], dtype=int)
        elif isinstance(shipment_delays, tuple) or (isinstance(shipment_delays, list) and len(shipment_delays)==2):
            self.stochastic_shipdelays_range = shipment_delays            
            self.shipment_delays = None
        else:
            self.shipment_delays = np.asarray([2] + shipment_delays, dtype=int)

        if self.stochastic_demand_range or self.stochastic_shipdelays_range:
            self.rand_generator = np.random.RandomState(seed)
        
        # Quantidade inicial de estoque em cada nível
        self.initial_inventory = np.asarray(initial_inventory, dtype=int)
        # Valor inicial de itens em transporte
        self.initial_shipment_value = initial_shipment
        # Pedidos colocados inicialmente
        self.initial_orders_value   = initial_orders    

        # Pedidos colocados para o nível acima
        self.initial_orders_placed = self.initial_orders_value + np.zeros(self.levels, dtype=int)
        # Pedidos pendentes que chegaram do nível abaixo (posição zero é a demanda do cliente,
        # e valor colocado aqui é ignorado depois)
        self.initial_incoming_orders = self.initial_orders_value + np.zeros(self.levels, dtype=int)

        # Definindo variável de estado atual
        self.current_state = None

    def _generate_stochastic_data(self, arange, asize):
        return self.rand_generator.randint(low=arange[0], high=arange[1], size=asize)

    def reset(self):
        self.week = 0
        self.inventory = np.copy(self.initial_inventory)
        self.backlog = np.zeros(self.levels, dtype=int)
        self.orders_placed = np.copy(self.initial_orders_placed)
        self.incoming_orders = np.copy(self.initial_incoming_orders)

        # Se a demanda é estocástica
        if self.stochastic_demand_range:
            self.customer_demand = self._generate_stochastic_data(self.stochastic_demand_range, self.max_weeks)
        # Se os leadtimes são estocásticos
        if self.stochastic_shipdelays_range:
            self.shipment_delays = self._generate_stochastic_data(self.stochastic_shipdelays_range, self.max_weeks)
            self.shipment_delays = np.insert(self.shipment_delays, 0, 2)
        
        # Estrutura para guardar todas as entregas. Por tempo, por nível.
        max_shipment_week = self.max_weeks+np.max(self.shipment_delays)+1
        self.shipments = np.zeros((max_shipment_week+1, self.levels), dtype=int)
        # Tratando as entregas pendentes já no momento inicial
        self.shipments[1:1+self.shipment_delays[0]][:] = self.initial_shipment_value

        self.inventory_costs = np.zeros(self.levels, dtype=float)
        self.backlog_costs = np.zeros(self.levels, dtype=float)
        self.penalty_costs = np.zeros(self.levels, dtype=float)

        self.all_orders_placed = np.zeros((self.levels,self.max_weeks+1), dtype=int)
        self.all_orders_placed[:,0] = self.initial_orders_value

        self.current_state = self._observation()

        return self.current_state

    def _observation(self):
        return self.max_stock + self.inventory - self.backlog

    def step(self, action):
        self.week += 1

        # 1. Receive inventory and advance shipment delay

        # Os estoques recebem o que estava previsto nas entregas
        self.inventory += self.shipments[self.week][:]
        # Não é necessário "avançar" as entregas, pois estamos guardando todas
        # no vetor (por tempo)

        # 2. Fill orders

        # Retailer obtém a demanda do cliente
        self.incoming_orders[0] = self.customer_demand[self.week-1]
        # Demais níveis recebem demanda do nível inferior (# 4. Advance the order slips)
        self.incoming_orders[1:] = self.orders_placed[:-1]

        # Os pedidos a serem tratados são o que veio dos níveis abaixo da cadeia
        # mais o que tinha pendente de backlog
        orders_to_fill = self.incoming_orders + self.backlog

        # O quanto será enviado será o total de pedidos a serem tratados, se
        # possível, ou tudo que tem, se não for possível.
        orders_to_deliver = np.minimum(self.inventory, orders_to_fill)

        # Já a entrega considera apenas o que realmente pode ser entregue.
        # Se o tempo de delay é zero, entrega direto nos estoques dos níveis abaixo
        if self.shipment_delays[self.week] == 0:
            self.inventory[:-1] += orders_to_deliver[1:] # A primeira posição é para o cliente
        else: # Se delay é maior que zero, agenda a entrega
            self.shipments[self.week+self.shipment_delays[self.week]][:-1] += orders_to_deliver[1:]

        # 3. Record the inventory or backlog

        # Desconta do estoque tudo que foi enviado para entrega
        self.inventory -= orders_to_deliver
        # Guarda como backlog tudo que não foi possível entregar
        self.backlog = orders_to_fill - orders_to_deliver

        # 4. Advance the order slips

        # Movendo os pedidos colocados para os pedidos chegando
        # Já feito no Passo 2

        # Pedidos para a fábrica (último nível) são colocados para entrega
        if self.shipment_delays[self.week] == 0:
            self.inventory[-1] += self.orders_placed[-1]
        else:
            self.shipments[self.week+self.shipment_delays[self.week]][-1] += self.orders_placed[-1]

        # 5. Place orders

        # Cada nível passa para o nível acima um pedido igual ao valor da ação

        self.orders_placed = np.array(action)

        self.all_orders_placed[:,self.week] = self.orders_placed.T

        # 6. Tratando agora as questões de Aprendizado (recompensa e próximo estado)

        self.current_state = self._observation()

        # A recompensa é a soma ponderada dos custos de estoque e backlog mais...
        reward = -np.sum(self.inv_cost*self.inventory + self.backlog_cost*self.backlog)
        # ... a penalização por produto que passarem da capacidade de estoque ou backlog
        exceeded_product = np.clip(self.inventory-self.max_stock,0,None) + np.clip(self.backlog-self.max_stock,0,None)
        reward += -np.sum(self.exceeded_capacity_penalty*exceeded_product)

        self.inventory_costs += self.inv_cost*self.inventory
        self.backlog_costs += self.backlog_cost*self.backlog
        self.penalty_costs += self.exceeded_capacity_penalty*exceeded_product

        is_terminal = self.week == self.max_weeks

        #if is_terminal: print(round(np.sum(self.inventory_costs),2), round(np.sum(self.backlog_costs),2), round(np.sum(self.penalty_costs),2), '=', round(np.sum(self.inventory_costs+self.backlog_costs+self.penalty_costs),2))

        if self.DEBUG: print('env.step()', self.inventory - self.backlog, self.current_state, reward.item())

        return (self.current_state, reward.item(), is_terminal, {})

    def render(self, mode='human'):
        print('\n' + '='*20)
        print('Week:\t', self.week)
        print('Inventory/back:\t', self.inventory, self.backlog, self.inventory - self.backlog)
        print('Incoming order:\t', self.incoming_orders)
        print('Orders placed:\t', self.orders_placed)
        if self.week < self.max_weeks:
            print('Next customer demand:\t', self.customer_demand[self.week])
        #print('Print shipments:\n', self.shipments)
        #print('self.shipments[', self.week+1, ':', self.week+self.shipment_delays[self.week]+1, ']')
        print('Next shipments:\t', [(i,list(self.shipments[i])) for i in range(self.week+1, self.week+6) if i < len(self.shipments)])
        print('Current delay:\t', self.shipment_delays[self.week])
        print('Inventory costs:', self.inventory_costs)
        print('Backlog costs:\t', self.backlog_costs)
        print('Penalty costs:\t', self.penalty_costs)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)
