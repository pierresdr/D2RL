import gym, fire, os, copy, datetime, json, logging
import numpy as np
from stable_baselines3 import SAC, PPO, TD3
from utils.neural_networks import mlp, cnn
from utils.miscellaneous import get_space_dim, index_sampler
from utils.delays import DelayWrapper
import torch
from torch.optim import RMSprop, Adam
from utils.save_results import save_training
from utils.plot_results import plots_std
from utils.stochastic_wrapper import StochActionWrapper
from tqdm import tqdm

# Logging level
logging.basicConfig(level='INFO', format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

def get_dagger_policy(beta, delayed_policy, expert_policy, all_actions=False):
    def  dagger_policy(obs, delayed_obs, actions):
        expert_action = expert_policy.predict(obs)[0]
        if np.random.uniform()<beta:
            return expert_action, expert_action
        else:
            return delayed_policy(delayed_obs.unsqueeze(0), actions.unsqueeze(0)).reshape(-1).detach().numpy(), expert_action
    return dagger_policy



def train_dagger(env, delay=7, algo_expert='sac', training_rounds=4, n_steps=1000, traj_len=250, gamma=1, n_neurons=[100, 100,], 
            learning_rate=1e-3, optimizer="RMSprop", batch_size=32, beta_routine=None, noise_routine=None, 
            random_action_routine=None, save_path='test', exact_path=False, seed=0, test_steps=1000, 
            expert_sample=True, n_channels=[4,4], kernel_size=3, policy='mlp', keep_dataset=True, 
            all_actions=False, stoch_mdp_distrib=None, stoch_mdp_param=None, **env_kwargs):

    if not exact_path:
        # Get the parameters which will be modified through tests
        params = copy.deepcopy(locals())
        for k,v in params.items():
            if isinstance(v,type): 
                params[k] = v.__name__

        # Create save folder
        head, tail = os.path.split(save_path)
        tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_") + tail
        save_path = os.path.join(head, tail)
        try:
            os.makedirs(save_path)
        except:
            pass
        with open(os.path.join(save_path,'parameters.json'), 'w') as f:
            json.dump(params, f)

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env, **env_kwargs)
    if stoch_mdp_distrib is not None:
        env = StochActionWrapper(env, distrib=stoch_mdp_distrib, param=stoch_mdp_param)
    denv = DelayWrapper(copy.deepcopy(env), delay=delay)
    if all_actions:
        if delay<5:
            test_delays = np.arange(delay).astype(int)
        else:
            test_delays = np.linspace(1,delay,5,dtype=int)
    else:
        test_delays = [delay]
    
    algo_expert = algo_expert.lower()
    expert_loader = eval(algo_expert.upper())
    if env.unwrapped.spec.id == "Pendulum-v0":
        if stoch_mdp_distrib is not None:
            expert_policy = expert_loader.load(os.path.join('trained_agent', '{}_Pendulum-v0_noise_{}_param_{}'.format(algo_expert,stoch_mdp_distrib,stoch_mdp_param), 'policy'))
        else:
            expert_policy = expert_loader.load(os.path.join('trained_agent', '{}_Pendulum-v0'.format(algo_expert), 'policy'))
    elif env.unwrapped.spec.id == "LunarLanderContinuous-v2":
        expert_policy = expert_loader.load(os.path.join('trained_agent', '{}_LunarLanderContinuous-v2'.format(algo_expert),'policy'))
    elif env.unwrapped.spec.id == "Walker2d-v2":
        expert_policy = expert_loader.load(os.path.join('trained_agent', '{}_Walker2d-v2'.format(algo_expert), 'policy'))
    else:
        raise ValueError


    state_dim = get_space_dim(env.observation_space)
    ext_state_dim = get_space_dim(denv.observation_space)
    action_dim = get_space_dim(env.action_space)
    if policy=='mlp':
        n_neurons = [i for i in n_neurons]
        if all_actions:
            n_neurons = [ext_state_dim] + n_neurons + [action_dim*delay]
        else:
            n_neurons = [ext_state_dim] + n_neurons + [action_dim]
        delayed_policy = mlp(n_neurons,all_actions=all_actions)
    elif policy=='cnn':
        n_channels = [i for i in n_channels]
        l_out = cnn.output_size(delay, n_channels, kernel_size)
        n_channels = [action_dim] + n_channels
        n_neurons = [i for i in n_neurons]
        n_neurons = [state_dim+l_out*n_channels[-1]] + n_neurons + [action_dim]
        delayed_policy = cnn(n_channels, n_neurons, kernel_size)
    else:
        raise ValueError

    delayed_policy.eval()


    # Define beta routine
    if beta_routine is None:
        beta_routine = [1] + [0]*(training_rounds-1) 
    elif beta_routine == 'linear':
        beta_routine = np.linspace(1,0,training_rounds)

    # Define random action selection routine 
    random_action_routine = [0]*training_rounds if random_action_routine is None else random_action_routine

    # Define random action selection routine 
    noise_routine = [0]*training_rounds if noise_routine is None else noise_routine


    state_buffer = np.zeros((training_rounds*n_steps, state_dim), dtype=float)
    mask_buffer = np.zeros(training_rounds*n_steps, dtype=bool)
    action_buffer = np.zeros((training_rounds*n_steps,action_dim), dtype=float)
    action_buffer_del = np.zeros((training_rounds*n_steps,action_dim), dtype=float)
    returns = np.zeros(training_rounds)
    test_returns = np.zeros((training_rounds, len(test_delays)))
    test_traj_returns = np.zeros((training_rounds, len(test_delays)))
    failed_reset = np.zeros((training_rounds, len(test_delays)))
    losses = np.zeros(training_rounds)
    traj_ret = []


    if optimizer=='RMSprop':
        set_optimizer = "RMSprop(delayed_policy.parameters(),  lr=learning_rate,  alpha=0.9, eps=1e-10)"
    elif optimizer=='Adam':
        set_optimizer = "Adam(delayed_policy.parameters(),  lr=learning_rate,)"
    else:
        raise ValueError
    optimizer = eval(set_optimizer)
    

    for round in range(training_rounds):
        policy = get_dagger_policy(beta_routine[round], delayed_policy, expert_policy, all_actions=all_actions)

        t = 0
        ret = 0
        done = False
        traj_t = 0 # The first step of the current trajectory
        traj_ret.append([])
        logging.info('\n Dagger training round {} with beta {}.'.format(round, beta_routine[round]))        
        pbar = tqdm(total = n_steps)
        while t<n_steps:
            # if the env is reinitialized, sample the first delay actions without following the policy
            if t+delay>=n_steps:
                t=n_steps
                pbar.update(delay)
            elif (t-traj_t)%traj_len==0 or done:
                obs = env.reset()
                done = False
                traj_t = t
                traj_ret[-1].append(0)
                state_buffer[round*n_steps+t] = obs
                for h in range(delay):
                    action = env.action_space.sample()
                    obs, reward, done, _ = env.step(action)
                    action_buffer[round*n_steps+t+h] = action
                    action_buffer_del[round*n_steps+t+h] = action
                    state_buffer[round*n_steps+t+h+1] = obs
                    ret += reward
                t += delay
                mask_buffer[round*n_steps+t-delay] = True
                pbar.update(delay)
            else:
                if np.random.uniform()<random_action_routine[round]:
                    action = env.action_space.sample()
                    expert_action = expert_policy.predict(obs)[0]
                else:
                    del_s = torch.from_numpy(state_buffer[round*n_steps+t-delay]).float()
                    s = torch.from_numpy(state_buffer[round*n_steps+t]).float()
                    if expert_sample:
                        e = torch.from_numpy(np.hstack((action_buffer[round*n_steps+t-delay:round*n_steps+t]))).float().reshape(-1,action_dim)
                    else:
                        e = torch.from_numpy(np.hstack((action_buffer_del[round*n_steps+t-delay:round*n_steps+t]))).float().reshape(-1,action_dim)
                    action, expert_action = policy(s, del_s, e)
                obs, reward, done, _ = env.step(action)
                ret += reward
                traj_ret[-1][-1] += reward
                state_buffer[round*n_steps+t+1] = obs
                action_buffer[round*n_steps+t] = expert_action + np.random.normal(0, noise_routine[round])
                action_buffer_del[round*n_steps+t] = action
                t += 1
                pbar.update(1)
                mask_buffer[round*n_steps+t-delay] = True
        pbar.close()
        returns[round] = ret/n_steps
        logging.info('\n Dagger training round {} DONE with mean reward {}.'.format(round, returns[-1]))        
        
        if keep_dataset:
            idx = np.arange(training_rounds*n_steps)
        else:
            idx = np.arange(round*n_steps,(round+1)*n_steps)
        losses[round], optimizer = imitation_learning(delay, delayed_policy, state_buffer, 
                action_buffer, action_buffer_del, mask_buffer, 
                batch_size=batch_size, optimizer=optimizer, all_actions=all_actions)
        test_returns[round], test_traj_returns[round], failed_reset[round] = test_policy(delayed_policy, 
                env, test_delays, test_steps, traj_len, all_actions=all_actions)
    traj_ret = np.array([np.mean(t) for t in traj_ret])
    save_training(delayed_policy, test_delays, returns, test_returns, losses, traj_ret, test_traj_returns,
            failed_reset, seed=seed, save_path=save_path)
    return save_path

                
def test_policy(policy, env, delays, steps, traj_len, all_actions=False):
    mean_rew = np.zeros(len(delays))
    traj_ret = np.zeros(len(delays))
    failed_reset = np.zeros(len(delays))
    for d_i,d in enumerate(delays):
        denv = DelayWrapper(copy.deepcopy(env), delay=d)

        ret = 0
        done = False
        traj_t = 0 # The first step of the current trajectory
        traj_r = []
        for t in range(steps):
            # if the env is reinitialized, sample the first delay actions without following the policy
            if (t-traj_t)%traj_len==0 or done:
                obs = denv.reset()
                if obs[0] is None:
                    done = True
                    failed_reset[d_i] += 1
                else:
                    done = False
                traj_t = t
                traj_r.append(0)

            
            if not done:
                state = torch.from_numpy(obs[0]).float().unsqueeze(0)
                actions = torch.from_numpy(obs[1]).float().unsqueeze(0)
                action = policy(state,actions).reshape(-1).detach().numpy()

                obs, reward, done, _ = denv.step(action)
                ret += sum(reward)
                traj_r[-1] += sum(reward)
        traj_ret[d_i] = np.mean(traj_r)
        mean_rew[d_i] = ret/steps

    return mean_rew, traj_ret, failed_reset


def imitation_learning(delay, delayed_policy, state_buffer, action_buffer, action_buffer_del,
            mask_buffer, batch_size=32, optimizer=None, all_actions=False):
    index_batch = index_sampler(mask_buffer, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    losses = 0
    delayed_policy.train()
    for idx in index_batch:
        s = state_buffer[idx]
        e = np.stack([np.roll(action_buffer_del,-i, axis=0)[idx] for i in range(delay)])
        e = e.transpose(1,0,2)
        s = torch.from_numpy(s).float()
        e = torch.from_numpy(e).float()
        optimizer.zero_grad()
        a_pred = delayed_policy(s, e)
        if all_actions:
            a = np.stack([np.roll(action_buffer_del,-i, axis=0)[idx] for i in range(1,delay+1)])
            a = torch.from_numpy(a).float()
        else:
            a = torch.from_numpy(np.roll(action_buffer,-delay, axis=0)[idx]).float()
        loss = loss_fn(a_pred, a)

        loss.backward()
        optimizer.step()
        losses += loss.item()
    delayed_policy.eval()
    return losses/sum(mask_buffer), optimizer



def train_dagger_seeds(env, delay=7, training_rounds=4, n_steps=1000, traj_len=250, gamma=1, n_neurons=[100, 100,], 
            learning_rate=1e-3, optimizer="RMSprop", batch_size=32, beta_routine=None, noise_routine=None, 
            random_action_routine=None, save_path='test', seeds=[0,1,2], expert_sample=True, **env_kwargs):
    exact_path = False
    print(seeds)
    for s in seeds:
        logging.info('\n Training with seed {}.'.format(s))        
        save_path = train_dagger(env, delay=delay, training_rounds=training_rounds, n_steps=n_steps, traj_len=traj_len, gamma=gamma, n_neurons=n_neurons, 
            learning_rate=learning_rate, optimizer=optimizer, batch_size=batch_size, beta_routine=beta_routine, noise_routine=noise_routine, 
            random_action_routine=random_action_routine, save_path=save_path, exact_path=exact_path, seed=s, expert_sample=expert_sample,  **env_kwargs)
        exact_path = True

    r = []; l = []
    for s in seeds:
        with open(os.path.join(save_path,'{}_{}.txt'.format('returns',s)), 'rb') as f:
            r.append(np.load(f))
        with open(os.path.join(save_path,'{}_{}.txt'.format('losses',s)), 'rb') as f:
            l.append(np.load(f))
    r = np.stack(r); l = np.stack(l)
    
    series = {
            'losses': [{'mean': l.mean(0),'std':l.std(0)}],
            'returns': [{'mean': r.mean(0), 'std':r.std(0)}],
            }
    plots_std(series, save_path=os.path.join(save_path,'training_mean'), title='Imitation loss and return over {} seeds'.format(len(seeds)))
    



if __name__ == "__main__":
    fire.Fire()
