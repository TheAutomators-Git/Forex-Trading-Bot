import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler
from environments import CSVTradingEnv, RewardAndProfitCallback
import matplotlib.pyplot as plt

# Load and prepare CSV data
df = pd.read_csv('data/train/DAT_MT_EURUSD_M1_2020.csv', delimiter=',', header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df.columns = df.columns.str.lower()
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.set_index('datetime', inplace=True)
df.drop(['date', 'time'], axis=1, inplace=True)

# Sample data hourly (optional)
# df = df.resample('h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

# Normalize data
df[['open', 'high', 'low', 'close', 'volume']] = MinMaxScaler().fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

# Create the environment and load the model for further training
csv_env = DummyVecEnv([lambda: CSVTradingEnv(df, window_size=10)])
model = A2C.load('a2c_trading_model_best', env=csv_env, verbose=1)

# Create the callback
callback = RewardAndProfitCallback()

# Train the model
model.learn(total_timesteps=500000, callback=callback)

# Print the net profit
net_profit = callback.profits[-1] if callback.profits else 0
print(f"Net Profit: {net_profit}")

# Plot the rewards and total profits
fig, ax1 = plt.subplots()

ax1.set_xlabel('Timesteps')
ax1.set_ylabel('Reward', color='tab:blue')
ax1.plot(callback.rewards, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Total Profit', color='tab:red')
ax2.plot(callback.profits, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Rewards and Total Profit during Training')
plt.show()

# Save the model
model.save('a2c_trading_model')