from gym.envs.registration import register

register(
    id='beergame-v0',
    entry_point='gym_supplychain.envs:BeerGameEnv',
)
