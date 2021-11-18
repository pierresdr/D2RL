# Code for the thesis 

## How to train an expert undelayed policy ?
The command is as follows,
python train_undelayed.py train_undelayed --env [the environment] --total_timesteps [number of training steps] --algo [algorithm to run] --n_saves [number of test while training to see training evolution] --save_path "trained_agent" --test_steps [number of steps during the training] --test_traj_len [length of the trajectories during tests]

One can choose the algorithm in the list of implemented algorithms: SAC, PPO and TD3.

Example of command:
python train_undelayed.py train_undelayed --env "Pendulum-v0" --total_timesteps 50000 --algo "SAC" --n_saves 100 --noise 0.0 --rnd_action 0.0 --save_path "trained_agent" --test_steps 1000 --test_traj_len 200



## How to train an imitation learning agent on delayed environment? 
The command is as follows:
python train_dagger.py train_dagger_seeds --save_path [path to save directory] --env [name of gym environment] --training_rounds [number of training rounds] --delay [length of the delay] --n_steps [number of steps] --traj_len [length of trajectories] --n_neurons [number of neurons per layer] --learning_rate [learning rate] --seeds [seeds to be run] --expert_sample "False" --optimizer "Adam" --batch_size [batch size] 
> To train with noise, use the arguments: --stoch_mdp_param [parameter for the noise, see utils/stochastic_wrapper.py] and --stoch_mdp_distrib [name of the noise, see utils/stochastic_wrapper.py]

Example of command:
python train_dagger.py train_dagger_seeds --save_path "results/pendulum-noise/beta" --env "Pendulum-v0" --training_rounds 50 --delay 5 --n_steps 10000 --traj_len 250 --n_neurons 100,100,10 --learning_rate 1e-3 --seeds 6,7,8,9 --expert_sample "False" --optimizer "Adam" --batch_size 64 --stoch_mdp_param 1.0 --stoch_mdp_distrib "Beta"

