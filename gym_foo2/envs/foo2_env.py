import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.stats import norm

class Foo2Env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.option_strike = 100
        self.current_price = 100  # price in percentage
        self.previous_price = 0
        self.volatility = 0.20  # vol in percentage
        self.time_to_maturity = 90  # time to maturity in days
        self.shares_held = 0
        self.isTerminalState = False
        self.reward = 0
        self.current_step = 0
        self.bs_delta_amount_held = 0
        self.bs_cash_flow_balance = 0
        self.bs_reward = 0

        d1 = (1 / (self.volatility * np.sqrt(self.time_to_maturity / 365))) * (
                    np.log(self.current_price / self.option_strike) +
                    0.5 * self.volatility * self.volatility * (self.time_to_maturity / 365))
        d2 = d1 - self.volatility*np.sqrt(self.time_to_maturity / 365)

        self.premium = self.current_price* norm.cdf(d1) - self.option_strike* norm.cdf(d2)
        print(self.premium)

        self.cash_flow_balance = self.premium
        self.bs_cash_flow_balance = self.premium

        self.reward_range = (-np.inf, 0)

        # Hedging Actions of the format Buy x%, Sell x%, Hold, etc. min amount of hedge ration is zero and max amount is 100
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1,1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=np.array([-10, 0]), high=np.array([10, 365]), dtype=np.float16)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.option_strike = 100
        self.current_price = 100  #price in percentage
        self.previous_price = 0
        self.volatility = 0.20 #vol in percentage
        self.time_to_maturity = 90 #time to maturity in days
        self.shares_held = 0
        self.cash_flow_balance = self.premium
        self.current_step =0
        self.isTerminalState = False
        self.reward = 0
        self.bs_delta_amount_held = 0
        self.bs_cash_flow_balance = self.premium
        self.bs_reward = 0

        obs = np.array([np.log(self.current_price/self.option_strike), self.time_to_maturity])
        return obs

    def step(self, action):

        self._take_action(action)
        self.current_step += 1
        self.time_to_maturity -= 1

        if (self.time_to_maturity) == 0:

            if self.current_price > self.option_strike:
                self.cash_flow_balance += -(1 - self.shares_held)*self.current_price + self.option_strike
                self.bs_cash_flow_balance += -(1 - self.bs_delta_amount_held)*self.current_price + self.option_strike
            else:
                self.cash_flow_balance += self.shares_held*self.current_price
                self.bs_cash_flow_balance += self.bs_delta_amount_held*self.current_price

            #self.cash_flow_balance += - np.maximum(0, self.current_price - self.option_strike)
            #print("self.cash_flow_balance", self.cash_flow_balance)
            self.reward = - np.abs(self.cash_flow_balance)
            self.bs_reward = - np.abs(self.bs_cash_flow_balance)

            #print("self.reward", self.reward)
            self.isTerminalState = True
        else:
            self.reward = [0]
            self.bs_reward = [0]

        obs = np.array([np.log(self.current_price/self.option_strike), self.time_to_maturity])
        return obs, self.reward, self.isTerminalState, self.shares_held, self.bs_delta_amount_held, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        self.previous_price = self.current_price
        rnd = np.random.standard_normal()

        self.current_price = self.previous_price* np.exp(-0.5*self.volatility*self.volatility*1/365 +
                                                    rnd*self.volatility* np.sqrt(1/365))
        #amount = action[0]
        if self.time_to_maturity>0:
            #action = np.float64(action[0])
            #action = action[0]
            #print(action)
            mu = action[0][0]
            sigma = action[0][1]

            #print("mu", mu)
            #print("sigma", sigma)
            amount = np.random.normal(mu, sigma, 1)
            #amount = np.clip(amount, self.action_space.low[0], self.action_space.high[0])
            amount_purchased = amount - self.shares_held
            self.shares_held = amount
            self.cash_flow_balance = self.cash_flow_balance - amount_purchased * self.current_price

            d1 = (1/(self.volatility*np.sqrt(self.time_to_maturity/365))) *(np.log(self.current_price/self.option_strike) +
                                                    0.5*self.volatility*self.volatility*(self.time_to_maturity/365))
            bs_delta = norm.cdf(d1)
            bs_purchase = bs_delta - self.bs_delta_amount_held
            self.bs_delta_amount_held = bs_delta
            self.bs_cash_flow_balance = self.bs_cash_flow_balance - bs_purchase * self.current_price


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.time_to_maturity==0:
            print('------------------------------------------------')
            print(f'Step: {self.current_step}')
            print(f'CurrentPrice: {self.current_price}')
            print(f'Cash Balance: {self.cash_flow_balance}')
            print(f'Shares held: {self.shares_held}')
            print(f'BS Shares held: {self.bs_delta_amount_held}')
            print(f'Time to maturity: {self.time_to_maturity}')
            print(f'done: {self.isTerminalState}')
            print(f'self.reward: {self.reward}')
            print(f'self.bs_reward: {self.bs_reward}')
            print('------------------------------------------------')

