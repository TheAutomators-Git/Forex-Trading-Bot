import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Custom Gym Environment
# when using a2c_trading_model_best: window_size=10, sample data minutely and use MinMax normalized data
class CSVTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size):
        super(CSVTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 1), dtype=np.float32)
        self.current_step = 0
        self._prepare_data()
        self.position = None
        self.entry_price = None
        self.total_profit = 0

    def _prepare_data(self):
        self.data = self.df['close'].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.total_profit = 0
        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        end = self.current_step + self.window_size
        obs = self.data[self.current_step:end]
        return obs.reshape(-1, 1)

    def step(self, action):
        reward = 0
        done = False
        truncated = False
        signal = 'hold'  # Default signal
        current_price = 0  # Default value for current_price

        self.current_step += 1

        if self.current_step + self.window_size >= len(self.data):
            done = True

        if not done:
            current_price = self.data[self.current_step + self.window_size - 1]

            if action == 1:  # Buy
                if self.position is None:  # Entering a new position
                    self.position = 'long'
                    self.entry_price = current_price
                    signal = 'open_long'
                elif self.position == 'short':  # Closing a short position
                    reward = float(self.entry_price - current_price)
                    self.total_profit += reward
                    self.position = 'long'
                    self.entry_price = current_price
                    signal = 'close_short_open_long'
                else:
                    signal = 'hold_long'

            elif action == 2:  # Sell
                if self.position is None:  # Entering a new position
                    self.position = 'short'
                    self.entry_price = current_price
                    signal = 'open_short'
                elif self.position == 'long':  # Closing a long position
                    reward = float(current_price - self.entry_price)
                    self.total_profit += reward
                    self.position = 'short'
                    self.entry_price = current_price
                    signal = 'close_long_open_short'
                else:
                    signal = 'hold_short'

            elif action == 0:  # Hold
                if self.position == 'long':
                    reward = float(current_price - self.entry_price)
                    self.total_profit += reward
                    self.position = None
                    self.entry_price = None
                    signal = 'close_long'
                elif self.position == 'short':
                    reward = float(self.entry_price - current_price)
                    self.total_profit += reward
                    self.position = None
                    self.entry_price = None
                    signal = 'close_short'
                else:
                    signal = 'hold'

        if done:
            if self.position == 'long':
                reward += float(current_price - self.entry_price)
                self.total_profit += reward
            elif self.position == 'short':
                reward += float(self.entry_price - current_price)
                self.total_profit += reward

        obs = self._next_observation()
        info = {'total_profit': self.total_profit, 'signal': signal}

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Position Trading Environment
# when using ppo_positiontrading_model: window_size=10, sample data hourly/minutely and use MinMax normalized data (best results)
# when using a2c_positiontrading_model: window_size=10, sample data hourly/minutely and use MinMax normalized data
class PositionTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, initial_balance=10000):
        super(PositionTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.min_price = df['close'].min()
        self.max_price = df['close'].max()
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 1), dtype=np.float32)
        self.current_step = 0
        self._prepare_data()
        self.position = None
        self.entry_price = None
        self.balance = initial_balance
        self.total_profit = 0

    def _prepare_data(self):
        self.data = self.df['close'].values.astype(np.float32)

    def _denormalize_price(self, normalized_price):
        return normalized_price * (self.max_price - self.min_price) + self.min_price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.balance = self.initial_balance
        self.total_profit = 0
        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        end = self.current_step + self.window_size
        obs = self.data[self.current_step:end]
        return obs.reshape(-1, 1)

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        self.current_step += 1

        if self.current_step + self.window_size >= len(self.data):
            done = True

        current_normalized_price = self.data[self.current_step + self.window_size - 1]
        current_price = self._denormalize_price(current_normalized_price)

        if action == 1:  # Buy
            if self.position is None:  # Entering a new position
                self.position = 'long'
                self.entry_price = current_price
                signal = 'open_long'
            elif self.position == 'short':  # Closing a short position
                reward = float(self.entry_price - current_price)
                self.total_profit += reward
                self.balance += reward
                self.position = 'long'
                self.entry_price = current_price
                signal = 'close_short_open_long'
            else:
                signal = 'hold_long'

        elif action == 2:  # Sell
            if self.position is None:  # Entering a new position
                self.position = 'short'
                self.entry_price = current_price
                signal = 'open_short'
            elif self.position == 'long':  # Closing a long position
                reward = float(current_price - self.entry_price)
                self.total_profit += reward
                self.balance += reward
                self.position = 'short'
                self.entry_price = current_price
                signal = 'close_long_open_short'
            else:
                signal = 'hold_short'

        elif action == 0:  # Hold
            if self.position == 'long':
                reward = float(current_price - self.entry_price)
                self.total_profit += reward
                self.balance += reward
                self.position = None
                self.entry_price = None
                signal = 'close_long'
            elif self.position == 'short':
                reward = float(self.entry_price - current_price)
                self.total_profit += reward
                self.balance += reward
                self.position = None
                self.entry_price = None
                signal = 'close_short'
            else:
                signal = 'hold'

        if done:
            if self.position == 'long':
                reward += float(current_price - self.entry_price)
                self.total_profit += reward
                self.balance += reward
            elif self.position == 'short':
                reward += float(self.entry_price - current_price)
                self.total_profit += reward
                self.balance += reward

        obs = self._next_observation()
        info = {'total_profit': self.total_profit, 'balance': self.balance, 'signal': signal}

        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}, Action: {action}, Signal: {signal}, Reward: {reward}, Total Profit: {self.total_profit}, Balance: {self.balance}")

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Scalping Trading Environment
class ScalpTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size):
        super(ScalpTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 3), dtype=np.float32)  # 3 features: Close, MA, RSI
        self.current_step = 0
        self._prepare_data()
        self.position = None
        self.entry_price = None
        self.total_profit = 0

    def _prepare_data(self):
        self.df['MA'] = self.df['close'].rolling(window=5).mean()
        self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
        self.df.dropna(inplace=True)
        self.data = self.df[['close', 'MA', 'RSI']].values.astype(np.float32)

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.total_profit = 0
        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        end = self.current_step + self.window_size
        obs = self.data[self.current_step:end]
        return obs

    def step(self, action):
        reward = 0
        done = False
        truncated = False
        signal = 'hold'  # Default signal

        self.current_step += 1

        if self.current_step + self.window_size >= len(self.data):
            done = True

        if not done:
            current_data = self.data[self.current_step + self.window_size - 1]
            current_price = current_data[0]

            if action == 1:  # Buy
                if self.position is None:  # Entering a new position
                    self.position = 'long'
                    self.entry_price = current_price
                    signal = 'open_long'
                elif self.position == 'short':  # Closing a short position
                    reward = float(self.entry_price - current_price)
                    self.total_profit += reward
                    self.position = 'long'
                    self.entry_price = current_price
                    signal = 'close_short_open_long'
                else:
                    signal = 'hold_long'

            elif action == 2:  # Sell
                if self.position is None:  # Entering a new position
                    self.position = 'short'
                    self.entry_price = current_price
                    signal = 'open_short'
                elif self.position == 'long':  # Closing a long position
                    reward = float(current_price - self.entry_price)
                    self.total_profit += reward
                    self.position = 'short'
                    self.entry_price = current_price
                    signal = 'close_long_open_short'
                else:
                    signal = 'hold_short'

            elif action == 0:  # Hold
                if self.position == 'long':
                    reward = float(current_price - self.entry_price)
                    self.total_profit += reward
                    self.position = None
                    self.entry_price = None
                    signal = 'close_long'
                elif self.position == 'short':
                    reward = float(self.entry_price - current_price)
                    self.total_profit += reward
                    self.position = None
                    self.entry_price = None
                    signal = 'close_short'
                else:
                    signal = 'hold'

        if done:
            if self.position == 'long':
                reward += float(current_price - self.entry_price)
                self.total_profit += reward
            elif self.position == 'short':
                reward += float(self.entry_price - current_price)
                self.total_profit += reward

        obs = self._next_observation()
        info = {'total_profit': self.total_profit, 'signal': signal}

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Custom callback class to log rewards and total profit
class RewardAndProfitCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardAndProfitCallback, self).__init__(verbose)
        self.rewards = []
        self.profits = []

    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.rewards.append(reward)
        total_profit = self.locals['infos'][0]['total_profit']
        self.profits.append(total_profit)
        return True