'''
    Subclass of trading main environment, observations are derivatives of prices
    in the previous N minutes.
'''

from env.fx_trading_env import TradingMain
from gym import spaces
import numpy as np

MAX_PRICE = 5.0
MAX_TIME = 23*60+59

class TradingPrices(TradingMain):

    def __init__(self, scale=1e3, gamma=1, use_weekdays=True, **kwargs):
        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        if use_weekdays:
            observation_low = np.concatenate([np.full((self.window), -MAX_PRICE), [0.0, -1.0, 0]])
            observation_high = np.concatenate([np.full((self.window), +MAX_PRICE), [6.0, +1.0, MAX_TIME]])
        else:
            observation_low = np.concatenate([np.full((self.window), -MAX_PRICE), [-1.0, 0]])
            observation_high = np.concatenate([np.full((self.window), +MAX_PRICE), [+1.0, MAX_TIME]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.action_space = spaces.Discrete(3)
        self.scale = scale
        # Required for FQI
        self.action_dim = 1
        self.state_dim = self.observation_space.shape[0]
        self.gamma = gamma

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        _portfolio = self.current_portfolio
        _time = self.current_time
        return np.hstack((self.prices, _portfolio, _time))

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        r = self.current_portfolio * (self.current_price - self.previous_price) - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees
        return self.scale*r

    def reset(self):
        super().reset()
        return self._observation()


