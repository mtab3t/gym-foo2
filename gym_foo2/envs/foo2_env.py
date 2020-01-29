import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class Foo2Env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.option_strike = 100
        self.current_price = 100  # price in percentage
        self.previous_price = 0
        self.volatility = 0.20  # vol in percentage
        self.time_to_maturity = 90  # time to maturity in days
        self.shares_held = 0
        self.cash_flow_balance = 0
        self.isTerminalState = False
        self.reward = 0
        self.current_step = 0

        self.reward_range = (-np.inf, np.inf)

        # Hedging Actions of the format Buy x%, Sell x%, Hold, etc. min amount of hedge ration is zero and max amount is 100
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.option_strike = 100
        self.current_price = 100  #price in percentage
        self.previous_price = 0
        self.volatility = 0.20 #vol in percentage
        self.time_to_maturity = 90 #time to maturity in days
        self.shares_held = 0
        self.cash_flow_balance = 0
        self.current_step =0
        self.isTerminalState = False

        obs = np.array([self.current_price, self.shares_held , self.cash_flow_balance, self.time_to_maturity])
        return obs

    def step(self, action):

        self._take_action(action)
        self.current_step += 1
        self.time_to_maturity -= 1

        if (self.time_to_maturity) == 0:
            self.cash_flow_balance += - np.maximum(0, self.current_price - self.option_strike)
            self.reward = -np.abs(self.cash_flow_balance)
            self.isTerminalState = True

        else:
            self.reward = 0

        obs = np.array([self.current_price, self.shares_held, self.cash_flow_balance, self.time_to_maturity])
        return obs, self.reward, self.isTerminalState, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        self.previous_price = self.current_price
        rnd = np.random.standard_normal()
        print('rnd', rnd)
        self.current_price = self.previous_price* np.exp(-0.5*self.volatility*self.volatility*1/365 +
                                                    rnd*self.volatility* np.sqrt(1/365))

        action_type = action[0]
        amount = action[1]
        if action_type < 1:
            # Buy amount % of balance in shares
            self.shares_held = self.shares_held + amount
            self.cash_flow_balance = self.cash_flow_balance - amount*self.current_price

        elif action_type < 2:
            # Sell amount % of shares held
            self.shares_held = self.shares_held - amount
            self.cash_flow_balance = self.cash_flow_balance + amount * self.current_price

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'CurrentPrice: {self.current_price}')
        print(f'Cash Balance: {self.cash_flow_balance}')
        print(f'Shares held: {self.shares_held}')
