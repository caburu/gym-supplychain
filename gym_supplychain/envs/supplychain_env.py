import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}

    def __init__(self, env_init_info={}):
        '''Initial inventory is a list with the initial inventory position for each level
        '''
        self.DEBUG = False

        iib = SupplyChain_InitInfoBuilder(penalization=10)

        # Número de níveis da cadeia
        self.levels       = iib.define_num_levels(env_init_info)
        # Custo por unidade em estoque por semana
        self.inv_cost     = iib.define_inventory_costs(env_init_info)
        # Custo por unidade em backlog por semana
        self.backlog_cost = iib.define_backlog_costs(self.inv_cost)

        # Demanda dos clientes a cada semana
        self.customer_demand = np.asarray(iib.define_customer_demand(env_init_info), dtype=int)
        # Quantidade inicial de estoque em cada nível
        self.initial_inventory = np.asarray(iib.define_initial_inventory(env_init_info), dtype=int)
        # Número de semanas a simular
        self.max_weeks = len(self.customer_demand)
        # Máximo leadtime de entrega
        self.shipment_delays = np.asarray(iib.define_shipment_delays(env_init_info))

        # Estrutura para guardar todas as entregas. Por tempo, por nível.
        max_shipment_week = self.max_weeks + max(self.shipment_delays) + 1
        self.initial_shipment = np.zeros((max_shipment_week, self.levels), dtype=int)

        self.transf_factor = iib.define_transf_factor(env_init_info)

        self.current_state = None

        # Tratando variáveis do OpenAI Gym environment (# Ver uso do MultiDiscrete)
        #self.action_space = spaces.Discrete(self.action_codes**self.levels)
        #self.observation_space = spaces.Discrete(self.state_codes**self.levels)

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
        for level in range(self.levels-1):
            # Se o tempo de delay é zero, entrega direto nos estoques dos níveis abaixo
            if self.shipment_delays[level] == 0:
                self.inventory[level] += orders_to_deliver[level+1] # A primeira posição é para o cliente
            else: # Se delay é maior que zero, agenda a entrega
                self.shipments[self.week+self.shipment_delays[level]][level] += orders_to_deliver[level+1]

        # 3. Record the invetory or backlog

        # Desconta do estoque tudo que foi enviado para entrega
        self.inventory -= orders_to_deliver
        # Guarda como backlog tudo que não foi possível entregar
        self.backlog = orders_to_fill - orders_to_deliver

        # 4. Advance the order slips

        # Movendo os pedidos colocados para os pedidos chegando
        # Já feito no Passo 2

        # Pedidos para a fábrica (último nível) são colocados para entrega
        if self.shipment_delays[-1] == 0:
            self.inventory[-1] += self.orders_placed[-1]
        else:
            self.shipments[self.week+self.shipment_delays[-1]][-1] += self.orders_placed[-1]

        # 5. Place orders

        # Cada nível passa para o nível acima um pedido de tamanho X+Y
        # onde X é o que recebeu de demanda e Y é a quantidade a decidir pelo agente

        self.orders_placed = self.transf_factor*self.incoming_orders + action

        self.all_orders_placed[:,self.week] = self.orders_placed.T

        # 6. Tratando agora as questões de Aprendizado (recompensa e próximo estado)

        self.current_state = self._observation()

        # A recompensa é a soma ponderada dos custos de estoque e backlog
        reward = -np.sum(self.inv_cost*self.inventory + self.backlog_cost*self.backlog)
        self.inventory_costs += self.inv_cost*self.inventory
        self.backlog_costs += self.backlog_cost*self.backlog

        is_terminal = self.week == self.max_weeks

        if self.DEBUG: print('env.step()', self.inventory - self.backlog, self.current_state)

        return self.current_state, reward, is_terminal, {}

    def reset(self):
        self.week = 0
        self.inventory = np.copy(self.initial_inventory)
        self.backlog = np.zeros(self.levels, dtype=int)
        self.orders_placed = np.zeros(self.levels, dtype=int)
        self.incoming_orders = np.zeros(self.levels, dtype=int)
        self.shipments = np.copy(self.initial_shipment)

        self.inventory_costs = np.zeros(self.levels, dtype=float)
        self.backlog_costs = np.zeros(self.levels, dtype=float)

        self.all_orders_placed = np.zeros((self.levels,self.max_weeks+1), dtype=int)

        self.current_state = self._observation()

        return self.current_state

    def render(self, mode='human'):
        print('\n' + '='*20)
        print('Week:\t', self.week)
        print('Inventory:\t', self.inventory, self.backlog, self.inventory - self.backlog)
        print('Incoming order:\t', self.incoming_orders)
        print('Orders placed:\t', self.orders_placed)
        if self.week < self.max_weeks:
            print('Next customer demand:\t', self.customer_demand[self.week])
        #print('Print shipments:\n', self.shipments)
        #print('self.shipments[', self.week+1, ':', self.week+self.shipment_delays[self.week]+1, ']')
        print('Next shipments:\t', [(i,list(self.shipments[i])) for i in range(self.week+1, self.week+6) if i < len(self.shipments)])
        print('Inventory costs:\t', self.inventory_costs)
        print('Backlog costs:\t', self.backlog_costs)
        #if (self.week == self.max_weeks):
        #    print('All orders placed:')
        #    for level in range(self.levels):
        #        print(level, self.all_orders_placed[level])

    def close(self):
        pass

    def _observation(self):
        return self.inventory - self.backlog


class SupplyChain_InitInfoBuilder:

    def __init__(self, penalization=10):
        self.penalization = penalization

    def define_num_levels(self, data):
        """ São 3 níveis:
            - A frente/fornecedor:
              - Com delay=1 para matéria-prima chegar no estoque (por eqto = 0)
            - A central/fábrica:
              - Tratando transformação de matéria-prima em produto.
              - Com delay=2 para produto chegar no estoque.
            - O ponto de demanda/revendedor:
              - Com delay=1 para chegar no estoque.
              - E com altos custos de estoque e backlog para evitar ao máximo que aconteça.
        """
        return len(data['chain_settings']['levels'])

    def define_inventory_costs(self, data):
        """ Os custos de estoque é por nível e o custo de cada nível é dado pela
            média apenas dos custos de estoque. Exceto o custo do revendedor que
            é bem mais alto para se penalizar tentanto evitar que ele tenha estoque.

            Não vamos incluir os custos de fabricação e transporte porque eles serão
            sempre necessários (acontecerão de qualquer jeito para toda unidade
            de produto necessárias). O que pode ajudar a definir uma solução melhor
            ou pior é quanto vamos estocar.
        """
        suppliers_avg_cost = (sum(data['chain_settings']['costs']['suppliers']['stock'])
                              / data['chain_settings']['levels']['suppliers'])

        factories_avg_cost = (sum(data['chain_settings']['costs']['factories']['stock'])
                              / data['chain_settings']['levels']['factories'])

        retailers_avg_cost = self.penalization*(suppliers_avg_cost+factories_avg_cost)

        return np.array([retailers_avg_cost, factories_avg_cost, suppliers_avg_cost], dtype=float)

    def define_backlog_costs(self, inv_costs):
        """ Backlog na verdade não pode acontecer no nosso cenário da cadeia de
            suprimentos. Logo, esse custo precisar ser visto como penalização para
            evitar que ele ocorra.

            É necessário pensar em algum valor inicial e depois realizar experimentos
            para defini-lo melhor.
        """
        return self.penalization*inv_costs

    def define_customer_demand(self, data):
        """ A ideia original seria aprender a lidar com as variações de demanda
            que ocorrem depois do plano otimizado.

            Mas a ideia de tentar resolver o problema como uma instância do Beer
            Game perde essa noção. Pois aprenderia apenas baseado no histórico
            que já atende a demanda usando a heurística implementada.

            Por enquanto, vai aprender usando apenas a informação da demanda real.
        """
        periods = data['chain_settings']['periods']
        #periods = 30
        # Por equando vamos simplesmente considerar a demanda real (que pode ter
        # sofrido variação).
        real_demand = [0]*periods
        for i in range(periods):
            # A demanda real de todos os pontos de demanda
            real_demand[i] = np.sum(np.asarray(data['periods'][str(i)]['retailers']['actual_demand']))

        return real_demand

    def define_initial_inventory(self, data):
        """ Quantidade inicial nos estoques de cada nível. Soma os estoques de
            cada nível."""

        inventory = [0]
        inventory.append(sum(data['chain_settings']['materials']['initial_stocks']['fact_stocks']))
        inventory.append(sum(data['chain_settings']['materials']['initial_stocks']['supp_stocks']))
        return inventory

    def define_shipment_delays(self, data):
        """ Os delays de entrega no nosso caso não são variáveis no tempo. Eles são
            fixos, mas diferentes em cada parte da cadeia.

            Nas frentes/fornecedores o delay é 1.
            Nas centrais/fábricas o delay é 2.
            Nos revendedores o delay é 1 (por eqto = 0).
        """
        return [0,2,1]

    def define_transf_factor(self, data):
        """ O fator de transformação é usada para a transformação da quantidade
            de matéria-prima em quantidade de produto.

            Para deixar a implementação mais flexível, usaremos um vetor para
            tranformação em todos os níveis da cadeia, e deixamos o valor 1 para
            os níveis da cadeia que não têm transformação.
        """
        r2p = data['chain_settings']['materials']['raw_to_product'][0][0]

        return np.array([1, r2p, 1])
