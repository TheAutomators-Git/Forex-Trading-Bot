import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler
from torchinfo import summary
from environments import PositionTradingEnv  
import matplotlib.pyplot as plt
import time

# Load and prepare CSV data
df_val = pd.read_csv('data/val/DAT_MT_EURUSD_M1_202406.csv', delimiter=',', header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df_val.columns = df_val.columns.str.lower()
df_val['datetime'] = pd.to_datetime(df_val['date'] + ' ' + df_val['time'])
df_val.set_index('datetime', inplace=True)
df_val.drop(['date', 'time'], axis=1, inplace=True)

# Sample data hourly (optional)
# df = df.resample('h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

# Normalize data
df_val[['open', 'high', 'low', 'close', 'volume']] = MinMaxScaler().fit_transform(df_val[['open', 'high', 'low', 'close', 'volume']])

# Create the validation environment and load the model
val_env = DummyVecEnv([lambda: PositionTradingEnv(df_val, window_size=10)])  # Updated environment
model = PPO.load("ppo_positiontrading_model", val_env)

# Print the model summary
print("Model Summary:")
print(model)

# Inspecting Specific Layers
print("\nSpecific Layers:")
for name, param in model.policy.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}, Values: {param[:5]}")  # Display first 5 values for brevity

# Detailed Summary using torchinfo
print("\nDetailed Model Summary using torchinfo:")
summary(model.policy)
time.sleep(10)

# Initialize lists to store rewards and profits
rewards = []
profits = []

# Run the model on the validation environment
obs = val_env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = val_env.step(action)  # Updated to include truncated
    rewards.append(reward[0])
    profits.append(info[0]['total_profit'])
    signal = info[0]['signal']
    print(f"Step: {len(rewards)}, Action: {action[0]}, Signal: {signal}, Reward: {reward[0]}, Total Profit: {info[0]['total_profit']}")  # Debug print
    val_env.render()

net_profit_val = profits[-1] if profits else 0
print(f"Net Profit on Validation Set: {net_profit_val}")

# Plot the rewards and total profits for validation
fig, ax1 = plt.subplots()

ax1.set_xlabel('Timesteps')
ax1.set_ylabel('Reward', color='tab:blue')
ax1.plot(rewards, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Total Profit', color='tab:red')
ax2.plot(profits, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Rewards and Total Profit during Validation')
plt.show()