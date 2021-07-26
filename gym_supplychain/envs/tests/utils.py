import numpy as np

def check_rewards(acumm_rewards, info, num_products):
    # Confere se a soma de recompensas retornadas, bate com o total nas estatísticas do ambiente
    assert np.allclose(acumm_rewards, info['sc_episode']['rewards'])
    total = 0
    for key in info['sc_episode']['costs']:
        for prod in range(num_products):
            total += info['sc_episode']['costs'][key][prod]
    # Agora confere com o soma dos custos por tipo        
    assert np.allclose(acumm_rewards, -total)

def check_build_info(env):
    env.seed(1)
    env.reset()

    done = False
    rewards = 0        
    while not done:
        _, r, done, info = env.step(env.action_space.sample())            
        rewards += r
        check_rewards(rewards, info, env.num_products)
