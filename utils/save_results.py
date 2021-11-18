

import os, json 
import numpy as np
from torch import save as torchsave
from utils.plot_results import plots

def save_numpy(variable, name, save_path, seed=None):
    suffix = '_{}'.format(seed) if seed is not None else ''
    if variable is not None:
        with open(os.path.join(save_path,'{}{}.txt'.format(name,suffix)), 'wb') as f:
            np.save(f, variable, allow_pickle=False,)

def save_training(delayed_policy, delays, returns, test_returns, losses, traj_ret, test_traj_returns, failed_reset, seed, save_path):

    with open(os.path.join(save_path,'policy_class_{}.txt'.format(seed)), 'w') as f:
        json.dump(delayed_policy.__class__.__name__,f)
    torchsave(delayed_policy.state_dict(), os.path.join(save_path,'policy_{}.pt'.format(seed)))

    # Save numpy variables
    list_save = [
                {'variable': returns, 'name': 'returns'},
                {'variable': losses, 'name': 'losses'},
                {'variable': traj_ret, 'name': 'trajectory_returns'},
                ]
    for delay in delays:
        list_save.append({'variable': test_traj_returns, 'name': 'test_traj_returns_delay_{}'.format(delay)})
        list_save.append({'variable': test_returns, 'name': 'test_returns_delay_{}'.format(delay)})
        list_save.append({'variable': failed_reset, 'name': 'failed_reset_delay_{}'.format(delay)})
    
    for d in list_save:
        save_numpy(**d, save_path=save_path, seed=seed)


    series = {
            'losses': [losses],
            'returns': [returns],
            'trajectory_returns': [traj_ret],
            'test returns': [test_returns[-1]],
            'test trajectory returns': [test_traj_returns[-1]],
            }
    plots(series, os.path.join(save_path,'training_seed_{}'.format(seed)), title='Imitation loss and return')
    
    series = {
            'test returns': [{'x':np.arange(len(test_returns[d_i])), 'y':test_returns[d_i], 'label':'delay {}'.format(d)} for d_i,d in enumerate(delays)],
            'test returns': [{'x':np.arange(len(test_traj_returns[d_i])), 'y':test_traj_returns[d_i], 'label':'delay {}'.format(d)} for d_i,d in enumerate(delays)]
            }
    plots(series, os.path.join(save_path,'training_all_delays_seed_{}'.format(seed)), title='Returns for different delays')
