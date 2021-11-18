import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import os

class TradingMain(gym.Env):
    """
        Abstract class which implements the trading actions. Must be extended for
        different types of observations and rewards. Extends the vec environment
    """

    metadata = {'render.modes'}

    def __init__(self, data=None, n_envs=1, fees = 0., window=60, maximum_drowdown =None):
        # Check data (prices CSV)
        if data is not None:
            if isinstance(data, pd.DataFrame):
               self.data = data
            else:
              self.data = pd.read_csv(data)
        # Initialize parameters
        self.fees = fees        
        self.window=window
        self.maximum_drowdown = maximum_drowdown
        # Internal variables
        self.n_envs = n_envs
        self.dones = [True] * self.n_envs
        self.prices = None
        self.current_timestep = 0
        
        # Prices
        self.current_prices = None
        self.previous_prices = None
        # Portfolio
        self.current_portfolios = None
        self.previous_portfolios = None

    def seed(self, seed = None):
        np.random.seed(seed)
        random.seed(seed)

    def _observation(self):
        raise Exception('Not implemented in abstract class.')

    def _reward(self):
        raise Exception('Not implemented in abstract class.')

    def step(self, action):
        """
            Act on the environment. Target can be either a float from -1 to +1
            of an integer in {-1, 0, 1}. Float represent partial investments.
        """
        # Check if the environment has terminated.
        if self.done:
            return self._observation, 0.0, self.done, {}

        # Transpose action if action space is discrete [0, 2] => [-1, +1]
        # if isinstance(self.action_space, spaces.Discrete):
            # action = action - 1

        # Check actions are in range [-1, +1]
        assert -1 <= action <= 1, "Actions not in range!"

        # Update price
        self.current_timestep += 1
        self.previous_price, self.current_price = self.current_price,  self.data['open'].iloc[self.current_timestep]
        self.prices = np.insert(np.delete(self.prices,0), self.window, self.current_price)
        
        # Update time
        timestamp = self.data['timestamp'].iloc[self.current_timestep]
        self.current_time = int(timestamp[9:11])*60+int(timestamp[11:13])
        # Check if day has ended
        if self.current_timestep >= len(self.data):
            self.done = True

        # Perform action
        self.previous_portfolio, self.current_portfolio = self.current_portfolio, action

        # Compute the reward and update the profit
        r = self._reward()
        reward = r if not self.done else 0
        self.profit += reward

        # Check if drawdown condition is met
        if self.maximum_drowdown is not None:
            self.done = self.done or self.profit < self.maximum_drowdown
        return self._observation(), reward, self.done, {}

    def reset(self):
        self.current_timestep = self.window # starting by window to avoid nans
        # Init internals
        self.prices = np.array(self.data['open'].iloc[self.current_timestep-self.window:self.current_timestep+1])
        self.current_price = self.prices[-1]
        timestamp = self.data['timestamp'].iloc[self.current_timestep]
        self.current_time = int(timestamp[9:11])*60+int(timestamp[11:13])
        self.previous_price = None
        self.current_portfolio = 0
        self.previous_portfolio = None
        self.done = False
        self.profit = 0
        # return self._observation()
