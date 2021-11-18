'''
    Subclass of trading main environment, observations are derivatives of prices
    in the previous N minutes.
'''

from logging import raiseExceptions
from env.fx_trading_env import TradingMain
from gym import spaces
import numpy as np

MAX_DERIVATIVE = 5.0
MAX_TIME = float(23*60+59)

class TradingDerivatives(TradingMain):

    def __init__(self, scale=1e5,  gamma=1, use_weekdays=True, spread=False, spread_window=0, ten_step_window=0, pers_window=0, multiCurrency=False, use_diag=False, do_pca=False, n_pc=None, state_dim=None, **kwargs):

        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        if multiCurrency:
            if use_diag:
                if use_weekdays:
                    observation_low = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window + (spread_window +2 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [0.0, -1.0, -1.0, 0.0]])
                    observation_high = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window+ (spread_window +2 if spread else 0)),
                                                        +scale*MAX_DERIVATIVE), 
                                                [6.0, +1.0, +1.0, MAX_TIME]])
                else:
                    observation_low = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window + (spread_window +2 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [-1.0, -1.0, 0.0]])
                    observation_high = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window+ (spread_window +2 if spread else 0)), 
                                                        +scale*MAX_DERIVATIVE), 
                                                [+1.0, +1.0, MAX_TIME]])
            else:
                if use_weekdays:
                    observation_low = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window + (spread_window +1 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [0.0, -1.0, -1.0, 0.0]])
                    observation_high = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window+ (spread_window +1 if spread else 0)),
                                                        +scale*MAX_DERIVATIVE), 
                                                [6.0, +1.0, +1.0, MAX_TIME]])
                else:
                    observation_low = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window + (spread_window +1 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [-1.0, -1.0, 0.0]])
                    observation_high = np.concatenate([np.full(2*(self.window + ten_step_window + pers_window+ (spread_window +1 if spread else 0)), 
                                                        +scale*MAX_DERIVATIVE), 
                                                [+1.0, +1.0, MAX_TIME]])
        else:    
            if use_weekdays:
                observation_low = np.concatenate([np.full((self.window + ten_step_window + pers_window + (spread_window +1 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [0.0, -1.0, 0.0]])
                observation_high = np.concatenate([np.full((self.window + ten_step_window + pers_window+ (spread_window +1 if spread else 0)),
                                                        +scale*MAX_DERIVATIVE), 
                                                [6.0, +1.0, MAX_TIME]])
            else:
                observation_low = np.concatenate([np.full((self.window + ten_step_window + pers_window + (spread_window +1 if spread else 0)),
                                                        -scale*MAX_DERIVATIVE), 
                                                [-1.0, 0.0]])
                observation_high = np.concatenate([np.full((self.window + ten_step_window + pers_window+ (spread_window +1 if spread else 0)), 
                                                        +scale*MAX_DERIVATIVE), 
                                                [+1.0, MAX_TIME]])            
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)
        
        if multiCurrency:
            self.action_space = spaces.Discrete(5)
        else:       
            self.action_space = spaces.Discrete(3)
        
        # Internals
        self.scale = scale
        
        # Required for FQI
        if state_dim is not None:
            self.action_dim = 1
            self.state_dim = state_dim
        elif not(do_pca):
            self.action_dim = 1
            self.state_dim = self.observation_space.shape[0]
        
        else:
            if multiCurrency&use_weekdays&use_diag:
                self.action_dim = 1
                self.state_dim = n_pc + 8
            else:
                raise Exception('Option Not Implemented')        
        self.gamma = gamma

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        _portfolio = self.current_portfolio
        _time = self.current_time
        return np.hstack((self.delta, _portfolio, _time))

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        r = self.current_portfolio * (self.current_price - self.previous_price) - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees
        return self.scale*r

    def reset(self):
        super().reset()
        self.delta = np.flip(self.scale*(self.prices[1:]-self.prices[:-1])/self.prices[:-1])
        return self._observation()

    def step(self, action):
        _, reward, self.done, _ = super().step(action)
        self.delta = np.insert(np.delete(self.delta,self.window-1),  0, self.scale*(self.current_price-self.previous_price)/(self.previous_price))
        return self._observation(), reward, self.done, {}
        
        

class TradingDerivativesContinuous(TradingMain):

    def __init__(self, scale=1e5,  gamma=1, **kwargs):
        # Calling superclass init
        super().__init__(**kwargs)
        # Observation space and action space, appending PORTFOLIO and TIME
        observation_low = np.concatenate([np.full((self.window), -scale*MAX_DERIVATIVE), [-1.0, 0.0]])
        observation_high = np.concatenate([np.full((self.window), +scale*MAX_DERIVATIVE), [+1.0, MAX_TIME]])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high)
        self.action_space = spaces.Box(low=np.full(3,1), high=np.full(3,-1))
        
        # Internals
        self.scale = scale
        # Required for FQI
        self.action_dim = 1
        self.state_dim = self.observation_space.shape[0]
        self.gamma = gamma

    def _observation(self):
        # Pad derivatives with zeros for the first time_lag minutes
        _portfolio = self.current_portfolio
        _time = self.current_time
        return np.hstack((self.delta, _portfolio, _time))

    def _reward(self):
        # Instant reward: (current_portfolio) * delta_price - delta_portfolio * fee
        # In case of continuous portfolio fix
        r = self.current_portfolio * (self.current_price - self.previous_price) - \
                abs(self.current_portfolio - self.previous_portfolio) * self.fees
        return self.scale*r

    def reset(self):
        super().reset()
        self.delta = np.flip(self.scale*(self.prices[1:]-self.prices[:-1])/self.prices[:-1])
        return self._observation()

    def step(self, action):
        _, reward, self.done, _ = super().step(action)
        self.delta = np.insert(np.delete(self.delta,self.window-1),  0, self.scale*(self.current_price-self.previous_price)/(self.previous_price))
        return self._observation(), reward, self.done, {}