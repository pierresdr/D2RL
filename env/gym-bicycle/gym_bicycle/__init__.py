from gym.envs.registration import register

register(
    id='Bicycle-v0',
    entry_point='gym_bicycle.envs:BicycleEnv',
)