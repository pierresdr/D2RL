import gym, tqdm
import numpy as np
from gym import Wrapper
import copy, os, datetime, json, fire
import gym_bicycle
from utils.plot_results import plots_std
# import mujoco_py
from utils.stochastic_wrapper import StochActionWrapper

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

class RobustWrapper(Wrapper):
    def __init__(self, env, rnd_action=0., noise=0.):
        super(RobustWrapper, self).__init__(env)


        # Create State and Observation Space
        self.state_space = self.observation_space

        self.rnd_action = rnd_action
        self.noise = noise

        
    def step(self, action, render=False):
        
        if(np.random.uniform()<self.rnd_action): # waveing
            action = self.env.action_space.sample()

        action[0] += np.random.normal(0,self.noise)

        new_state,reward,done,info = self.env.step(action)

        if(render):
            self.render()
        return new_state,reward,done,info


def test_model(model, env, test_steps, test_traj_len):
    t = 0
    ret = []
    done = False
    traj_t = 0 # The first step of the current trajectory
    while t<test_steps:
        # if the env is reinitialized, sample the first delay actions without following the policy
        if (t-traj_t)%test_traj_len==0 or done:
            obs = env.reset()
            done = False
            traj_t = t

        action, _ = model.predict(obs)

        obs, reward, done, _ = env.step(action)
        ret.append(reward)
        t += 1

    return ret


def train_undelayed(env, algo='SAC', total_timesteps=50000, n_saves=1, noise=0., rnd_action=0., save_path='trained_agent/test', 
            test_steps=1000, test_traj_len=200, stoch_mdp_distrib=None, stoch_mdp_param=None,):
    # Get the parameters which will be modified through tests
    params = copy.deepcopy(locals())
    for k,v in params.items():
        if isinstance(v,type): 
            params[k] = v.__name__

    # Create save folder
    suffix = ''
    if stoch_mdp_distrib is not None:
        suffix = '_noise_{}_param_{}'.format(stoch_mdp_distrib,stoch_mdp_param)
    save_path = os.path.join(save_path,'{}_{}{}'.format(algo.lower(), env, suffix))
    head, tail = os.path.split(save_path)
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_") + tail
    save_path = os.path.join(head, tail)
    try:
        os.makedirs(save_path)
    except:
        pass
    with open(os.path.join(save_path,'parameters.json'), 'w') as f:
        json.dump(params, f)



    env = gym.make(env)
    if stoch_mdp_distrib is not None:
        env = StochActionWrapper(env, distrib=stoch_mdp_distrib, param=stoch_mdp_param)
    # rob_env = RobustWrapper(env, rnd_action=rnd_action, noise=noise)
    env.reset()

    if algo=='SAC':
        model = SAC("MlpPolicy", env, verbose=1,)
    elif algo=='PPO':
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo=='TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    else:
        raise ValueError


    save_every = total_timesteps//n_saves

    ret = []
    for i in tqdm.tqdm(range(0,total_timesteps,save_every)):
        model.learn(total_timesteps=save_every, log_interval=10)
        ret.append(test_model(model, env, test_steps, test_traj_len))
    model.save(os.path.join(save_path,'policy'))


    series = {
            'returns': [{'mean': np.array([np.mean(r) for r in ret]),'std': np.array([np.std(r) for r in ret])}],
            }
    plots_std(series, save_path=os.path.join(save_path,'learning_curve'), title='Return as a function of training rounds')
    


if __name__ == "__main__":
    fire.Fire()