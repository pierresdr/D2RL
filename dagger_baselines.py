import fire 
import os




# def run_dagger_delays(range_low=None, range_high=None, save_path=None,):
#     import itertools
#     from numpy import prod, arange
#     # range =  arange(0,101,step=5,dtype=int)
#     range =  arange(0,21,step=5,dtype=int)
#     range[0] = 1
#     hyperparam = {
#         'delay' : range,
#     }

#     n_runs = []
#     for v in hyperparam.values():
#         n_runs.append(len(v))
#     print('Size of gird: {}'.format(prod(n_runs)))
#     if range_low is None:
#         range_low = 0 
#     if range_high is None:
#         range_high = prod(n_runs)
#     print('Range ({}-{})'.format(range_low,range_high))

#     if save_path is None:
#         save_path = os.path.join('results','Pendulum-v0_dagger_delays')
        
#     for i, values in enumerate(itertools.product(*hyperparam.values())):
#         cur_path = os.path.join(save_path, "delay_{}".format(values[0]))
#         if i>=range_low and i<=range_high:
#             print(values)
#             os.system('python train_dagger.py train_dagger_seeds \
#             --save_path "{0}" --env "Pendulum-v0" --training_rounds 30 \
#             --delay {1} --n_steps 5000 --traj_len 250 \
#             --learning_rate 1e-3 --n_seeds 3 --expert_sample "False" --optimizer "Adam"'.format(cur_path, *values))

def run_dagger_delays(env, delays, algo_expert='sac', n_seeds=None, seeds=None, n_steps=5000, range_low=None, 
            n_neurons=[100,100,10], batch_size=32, range_high=None, save_path=None,):
    import itertools
    from numpy import prod
    if n_seeds is not None:
        seeds = [i for i in range(n_seeds)]
    elif seeds is None:
        seeds = [i for i in range(3)]
    else: 
        seeds = [i for i in seeds]
    hyperparam = {
        'delay' : delays,
    }

    n_runs = []
    for v in hyperparam.values():
        n_runs.append(len(v))
    print('Size of gird: {}'.format(prod(n_runs)))
    if range_low is None:
        range_low = 0 
    if range_high is None:
        range_high = prod(n_runs)
    print('Range ({}-{})'.format(range_low,range_high))

    if save_path is None:
        save_path = os.path.join('results','{}_dagger_delays'.format(env))
        
    names = list(hyperparam.keys())
    for i, values in enumerate(itertools.product(*hyperparam.values())):
        cur_path = os.path.join(save_path, "delay_{}".format(values[0]))
        if i>=range_low and i<=range_high:
            print(list(zip(names,values)))
            os.system('python train_dagger.py train_dagger_seeds \
            --save_path "{0}" --env "{1}" --training_rounds 30\
            --delay {7} --n_steps {3} --traj_len 250 --n_neurons "{4}" \
            --learning_rate 1e-3 --seeds "{2}" --expert_sample "False" \
            --optimizer "Adam" --batch_size {5} --algo_expert {6}'.format(cur_path, env, seeds, n_steps, 
            n_neurons, batch_size, algo_expert, *values))


if __name__=='__main__':
    fire.Fire()