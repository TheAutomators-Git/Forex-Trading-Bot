import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import A2C
from sklearn.preprocessing import MinMaxScaler
from environments import CSVTradingEnv

# Initialize MetaTrader 5 connection
login = "YOUR_LOGIN"
server = "MetaQuotes-Demo"
password = "YOUR_PASSWORD"

if not mt5.initialize(login=login, server=server, password=password):
    print("initialize() failed")
    mt5.shutdown()

# Load the A2C model
model = A2C.load("a2c_trading_model_best.zip")

# Function to fetch real-time data
def get_real_time_data(symbol, timeframe, num_bars):
    utc_from = datetime.now() - timedelta(minutes=1) 
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, num_bars)
    if rates is None:
        print(f"No data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print(df)

    # Normalize the close prices
    df['close'] = MinMaxScaler().fit_transform(df['close'].values.reshape(-1, 1))

    return df

# Function to generate trading signals using the A2C model
def generate_signals(env, model):
    obs, _ = env.reset()  # Reset environment and get initial observation
    done = False
    signal = None

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)  # Perform action and get environment feedback
        signal = info['signal']  # Extract the signal from the environment
        print(f"Action: {action}, Signal: {signal}, States: {_states}")
        env.render()

    return action, signal

# Function to place an order
def place_order(symbol, lot, order_type, sl_points, tp_points):
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": price - sl_points * point if order_type == mt5.ORDER_TYPE_BUY else price + sl_points * point,
        "tp": price + tp_points * point if order_type == mt5.ORDER_TYPE_BUY else price - tp_points * point,
        "deviation": 20,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    print(result)
    if result is not None and result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode={}".format(result.retcode))
        return False
                
    return True

# Main trading bot function
def run_bot(symbol, model, env, lot=0.01, sl_points=50, tp_points=200, timeframe=mt5.TIMEFRAME_M1, num_bars=500):
    while True:
        data = get_real_time_data(symbol, timeframe, num_bars)
        if data is not None:
            env.df = data[10:]  # Update the environment data
            env._prepare_data()
            action, signal = generate_signals(env, model)
            print(f"Action: {action}, Signal: {signal}")
            
            # Decision making based on the signal
            if signal == 'open_long' or signal == 'close_short_open_long':
                place_order(symbol, lot, mt5.ORDER_TYPE_BUY, sl_points, tp_points)
            elif signal == 'open_short' or signal == 'close_long_open_short':
                place_order(symbol, lot, mt5.ORDER_TYPE_SELL, sl_points, tp_points)

            # Additional logic to close positions if necessary
            if signal == 'close_long' or signal == 'close_short':
                print(f"Closing position based on signal: {signal}")
                # Implement logic to close open positions as needed

        time.sleep(10)  # Wait for 10 seconds before fetching new data

# Example usage
symbol = "EURUSD"
data = get_real_time_data(symbol, mt5.TIMEFRAME_M1, 500)
env = CSVTradingEnv(data, window_size=10)
run_bot(symbol, model, env)

# Shut down connection to the MetaTrader 5 terminal
mt5.shutdown()