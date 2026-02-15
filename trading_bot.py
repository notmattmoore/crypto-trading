# imports {{{1
# general
from copy import deepcopy
from time import sleep, time
import datetime as dt

# ML data
import numpy as np
import pandas as pd

# custom modules
from cb_auth      import CB_API_KEY, CB_API_PRIVATE_KEY
from coinbase     import CoinBase
from mylibs.utils import load, file_newest
from mylibs.utils import map_dict, str_dict, str_iter
from mylibs.utils import datetime_iso, print_dyn_line
from trade        import TradeDecider, CBTrader
#----------------------------------------------------------------------------}}}1


# Initial balances and currency reserve -- will be overridden by any saved trading state.
SYMBOLS_CONFIG = {
  "BTC-USD"   : {"curr": 20000, "asset": 0},
  "ETH-USD"   : {"curr": 20000, "asset": 0},
  # "SOL-USD"   : {"curr": 10000, "asset": 0},
  # "MATIC-USD" : {"curr": 5000, "asset": 0, "trade_on_rec": False},
}
RESERVE_CURR, RESERVE_ASSET = 0, 0

# Time delays for the loop (all in seconds).
DATA_INTERVAL   = 60
LOOP_DELAY      = 15   # CB minute candles should get updated ~15s after the minute

# Verbosity levels and output control
VERBOSE_SUMMARY = True
VERBOSE_DECIDER = 0
VERBOSE_TRADER  = 0
SUMMARY_PROFIT_OFFSETS = {  # what timedeltas to calculate profits for in the summary
  dt.timedelta(hours=1)           : {"name" : "hour"},
  dt.timedelta(hours=24)          : {"name" : "day"},
  dt.timedelta(weeks=1)           : {"name" : "week"},
  dt.timedelta(weeks=52.17857/12) : {"name" : "month"},
}

# Trader defaults.
TRADER_DEFAULTS = {
  "data_interval"       : DATA_INTERVAL,
  "verbose"             : VERBOSE_TRADER,
  "max_volume_buy_perc" : 0.1,  # a single buy should be at most 10% of the average volume
  "halt_config"         : (24*60, -0.1, 7*24*60),  # if more than 10% decline over 24 hours, then halt trading for a week
  "history"             : dt.timedelta(days=365.25/12*1.1), # store history for a bit over 1 month
  "history_attribs"     : ["value_total"],  # store the total value of the trader so that we can calculate historical profit
}
SIMTRADER_KEYS = [   # params to keep from the simtrader
  "min_buy_perc", "max_buy_perc", "ratio_buy", "ratio_sell", "stop_loss_perc",
  "take_profit_perc", # "take_profit_ratchet",
]
CB = CoinBase(CB_API_KEY, CB_API_PRIVATE_KEY)

# Example of the filesystem structure:
#   BTC-USD/
#    data/
#      BTC-USD_60_2023-01-20T04:00:32.csv
#      BTC-USD_60_2023-01-23T13:35:32.csv
#    models-params/
#      Same org as data/, single file for model and parameters. Each file contains.
#      - model,
#      - params_data, params_fe, params_train, params_model, params_trader.
#    data.log
#    features.log
#    orders.log
#    positions.log
#    trading-bot_state.pickle


def trader_load(*, symbol_pair, balance_curr=0, balance_asset=0, load_state=True, verbose=True):  # {{{
  """
  Load saved model/parameters for symbol pair and initialize a trader for it. If
  $load_state, then try to load a saved trading state. Returns the trader and the
  filename of the saved model/parameters.
  """
  # Load parameters and model.
  filename_model_params = file_newest(f"{symbol_pair}/models_params/")
  if verbose:
    print(f"{symbol_pair}: Loading parameters and model from {filename_model_params}.")
  params_and_model = load(filename_model_params)
  params_data      = params_and_model["params_data"]
  params_fe        = params_and_model["params_fe"]
  params_train     = params_and_model["params_train"]
  params_model     = params_and_model["params_model"]
  params_simtrader = params_and_model["params_trader"]
  model            = params_and_model["model"]

  if verbose:
    print(f"{symbol_pair}: Setting up decider...")
  pred_wrap = lambda y, t=params_train.get("pred_thresh", 0.5): (y[:,1] >= t).astype(int)
  decider = TradeDecider(
    model=model, pred_wrap=pred_wrap, params_data=params_data, params_fe=params_fe,
    data_ticks=params_simtrader.get("decider_data_ticks"), symbol_pair=symbol_pair,
    logging=f"{symbol_pair}/features.log", verbose=VERBOSE_DECIDER
  )

  if verbose:
    print(f"{symbol_pair}: Setting up trader...")
  # Build params_trader from params_simtrader.
  symbol_asset, symbol_curr = symbol_pair.split('-')
  params_trader = deepcopy(TRADER_DEFAULTS)
  params_trader.update({
    "CB"            : CB,
    "decider"       : decider,
    "symbol_asset"  : symbol_asset,
    "symbol_curr"   : symbol_curr,

    "balance_asset" : balance_asset,  # possibly overridden by saved trading state
    "balance_curr"  : balance_curr,   # ""
    "reserve_curr"  : RESERVE_CURR,   # ""
    "reserve_asset" : RESERVE_ASSET,  # ""

    # "hold_length"   : params_simtrader["hold_length"] * params_data.get("data_backtest_resample", 1),

    "logging" : tuple(f"{symbol_pair}/{l}.log" for l in ["data", "orders", "positions", "verbose"])
  })
  params_trader.update({k: params_simtrader[k] for k in SIMTRADER_KEYS if k in params_simtrader.keys()})
  params_trader["min_buy_perc"] = max(
    params_trader.get("min_buy_perc", 0),
    100 / balance_curr if balance_curr != 0 else 1   # min buy should be no less than $100
  )
  trader = CBTrader(**params_trader)

  if load_state:
    try:
      trader.state_import(load=True)
      print(f"{symbol_pair}: Loaded previous trading state.")
    except FileNotFoundError:
      print(f"{symbol_pair}: No previous trading state found.")

  return trader, filename_model_params
#----------------------------------------------------------------------------}}}
def trader_hist_profits(T, offsets):  # {{{
  """
  Given trader T, calculate the historical profits (incl percentages) for each offset
  in the dictionary offsets (the keys are the offsets).
  """
  if T.history == False:
    return {k:{**v, "profit": np.nan, "profit_perc": np.nan} for (k,v) in offsets.items()}

  # Find the closest history key for each offset, then calculate the profit based on
  # that key.
  offset_profits = { k:{**v, "delta_keyhist": (dt.timedelta.max, None)} for (k,v) in offsets.items() }
  dt_now = dt.datetime.now()
  for k in T.history.keys():
    for (o, o_v) in offset_profits.items():
      o_v["delta_keyhist"] = min(o_v["delta_keyhist"], (abs((dt_now - o) - k), k))
  for (o, o_v) in offset_profits.items():
    o_v["profit"] = T.history[o_v["delta_keyhist"][1]].get("value_total", np.nan) - T.value_total
    o_v["profit_perc"] = o_v["profit"] / T.value_total

  return offset_profits
#----------------------------------------------------------------------------}}}
def traders_print(traders, filenames_model_params):  # {{{
  """
  Print a composite summary for a dictionary of multiple traders (keys are symbol
  pairs, values are traders).
  """
  # Build composite candle DataFrame, positions, and balances.
  candles = pd.DataFrame(
    columns=["date", "open", "high", "low", "close", "volume", "sell", "hold", "buy", "rec"],
    index=traders.keys()
  )
  df_positions = pd.DataFrame()
  balances = pd.DataFrame(
    columns=["Currency Budget", "Asset Balance", "Asset Value"] \
        + [f"Profit ({v['name']})" for v in SUMMARY_PROFIT_OFFSETS.values()] \
        + ["Profit (total)"],
    index=traders.keys()
  )
  currencies = set()
  formats = dict()
  for symbol_pair, T in traders.items():
    # candles
    data_T = T.data_most_recent_block(size=1, stale=np.inf).copy()
    if T.decider.params_fe["task"] == "binary":
      dec_keys = ["hold", "buy"]
    elif T.decider.params_fe["task"] == "multiclass":
      dec_keys = ["sell", "hold", "buy"]
    data_T[dec_keys] = T.decider.dec_prob
    data_T["rec"] = T.decider.dec
    candles.loc[symbol_pair] = data_T.iloc[0]

    # positions
    P = T.positions_desc_df(show_profit=True)
    P["Symbol"] = symbol_pair
    df_positions = pd.concat([df_positions, P])

    # trader balances
    asset_value = T.data_most_recent_block(size=1, stale=np.inf)["close"].mean() * T.balance_asset
    offset_profits = trader_hist_profits(T, SUMMARY_PROFIT_OFFSETS)
    profit_total = T.value_total - SYMBOLS_CONFIG[symbol_pair]["curr"]
    profit_total_perc = T.value_total / SYMBOLS_CONFIG[symbol_pair]["curr"] - 1
    balance_T = {
      "Currency Budget" : T._format_curr(T.balance_curr),
      "Asset Balance"   : T._format_asset(T.balance_asset),
      "Asset Value"     : T._format_curr(asset_value),
    }
    balance_T.update({
      f"Profit ({v['name']})" : f"{T._format_curr(v['profit'])} ({v['profit_perc']:+.2%})" for v in offset_profits.values()
    })
    balance_T["Profit (total)"] = f"{T._format_curr(profit_total)} ({profit_total_perc:+.2%})"
    balances.loc[symbol_pair] = balance_T

    # account balances
    currencies.update(symbol_pair.split('-'))
    formats.update(zip(symbol_pair.split('-'), [T._format_asset, T._format_curr]))

  # Discard old candles, drop the date column, and titlecase the column names.
  candles = candles.dropna(axis=1)
  candles[candles["date"] != candles["date"].max()] = "---"
  candles = candles.drop(columns="date").rename(columns=lambda x: x.title())

  # Reorder columns, sort positions by date, and drop columns which contain only na
  # values (even if they have been string formatted).
  df_positions = df_positions.sort_values("Position Date") \
      .set_index("Symbol", drop=True) \
      .dropna(axis=1, how="all") \
      .loc[:, ~(df_positions.eq("NaT") | df_positions.eq("nat") | df_positions.eq("NaN") | df_positions.eq("nan")).all()]

  # Get total balances and format them nicely.
  currencies = CB.balances(currencies)
  currencies = {k: map_dict(formats[k], currencies[k]) for k in currencies.keys()}
  currencies = pd.DataFrame(currencies).sort_index(axis=1).rename(index=lambda x: x.title())

  # Do all the printing.
  print_dyn_line(f"-- {datetime_iso()} ", pad='-', end='\n')
  print(f"Model and parameter filenames:\n{str_dict(filenames_model_params, prefix='  ')}\n")
  print(candles, '\n')
  if len(df_positions) != 0:
    print(df_positions.to_string(index_names=False), '\n')
  print(balances, "\n\n", currencies, '\n', sep='')
#----------------------------------------------------------------------------}}}
def traders_update(traders, filenames_model_params, verbose=True):  # {{{
  """
  Check whether any of the deciders for the traders can be updated, and if so then
  update them.
  """
  for symbol_pair in traders.keys():
    filename_new = file_newest(f"{symbol_pair}/models_params/")
    if filenames_model_params[symbol_pair] != filename_new:
      if verbose:
        print(f"{symbol_pair}: Found updated model_params file {filename_new}.")
      bals_default = SYMBOLS_CONFIG[symbol_pair]
      traders[symbol_pair], filenames_model_params[symbol_pair] = trader_load(
        symbol_pair=symbol_pair, balance_curr=bals_default.get("curr", 0),
        balance_asset=bals_default.get("asset", 0), verbose=verbose
      )
#----------------------------------------------------------------------------}}}
def trading_loop(traders, filenames_model_params, trade_on_rec=True, loop_min_time=1):  # {{{
  """
  Loop through each trader in $traders, executing each one, printing a summary, and
  finally trying to update each one. Repeat this forever.
  """
  print(f"Entering trading loop for symbol pairs {str_iter(traders.keys(), sep=', ')}.")
  while True:
    time_sleep = (LOOP_DELAY - time() % 60) % DATA_INTERVAL
    if VERBOSE_SUMMARY:
      print_dyn_line(f"-- Sleeping until {datetime_iso(time() + time_sleep)} ({time_sleep:.0f}s) ", pad='-')
    sleep(time_sleep)
    time_start = time()

    for symbol_pair, T in traders.items():
      if VERBOSE_SUMMARY:
        print_dyn_line(f"-- {datetime_iso()}: running {symbol_pair} trader ", pad='-')
      T(trade_on_rec=SYMBOLS_CONFIG[symbol_pair].get("trade_on_rec", trade_on_rec))
      T.state_export(save=True)
    if VERBOSE_SUMMARY:
      print_dyn_line(f"-- {datetime_iso()}: generating summary ", pad='-')
      traders_print(traders, filenames_model_params)
    traders_update(traders, filenames_model_params)

    sleep(max(loop_min_time - (time() - time_start), 0))
#----------------------------------------------------------------------------}}}

# Load all the traders.
traders = {}
filenames_model_params = {}
for (symbol_pair, bals) in SYMBOLS_CONFIG.items():
  traders[symbol_pair], filenames_model_params[symbol_pair] = trader_load(
    symbol_pair=symbol_pair, balance_curr=bals.get("curr", 0),
    balance_asset=bals.get("asset", 0), verbose=VERBOSE_SUMMARY,
    # load_state=False  # XXX
  )

# Execute the trading loop.
trading_loop(traders, filenames_model_params)

from sys import exit; exit()

traders_print(traders, filenames_model_params)

# NOT EXECUTED {{{1
# Don't trade.
trading_loop(traders, filenames_model_params, trade_on_rec=False)

# Discard trading state.
traders = {}
filenames_model_params = {}
for (symbol_pair, bals) in SYMBOLS_CONFIG.items():
  traders[symbol_pair], filenames_model_params[symbol_pair] = trader_load(
    symbol_pair=symbol_pair, balance_curr=bals.get("curr", 0),
    balance_asset=bals.get("asset", 0), verbose=VERBOSE_SUMMARY,
    load_state=False   # discard previous trading state
  )
trading_loop(traders, filenames_model_params)

# Take profit.
traders["BTC-USD"].balance_curr -= 1000
traders["ETH-USD"].balance_curr -= 0
traders["SOL-USD"].balance_curr -= 3000
traders["BTC-USD"].balance_curr += 1000
traders["ETH-USD"].balance_curr += 1000
traders["SOL-USD"].balance_curr += 500
SYMBOLS_CONFIG["BTC-USD"]["curr"] += 1000
SYMBOLS_CONFIG["ETH-USD"]["curr"] += 1000
SYMBOLS_CONFIG["SOL-USD"]["curr"] += 500

# Reserve some money for human trading.
reserve_curr, reserve_asset = 0, 0
for T in traders.values():
  T.reserve_curr, T.reserve_asset = reserve_curr, reserve_asset
  print(f"{T.symbol_pair=}, {T.reserve_curr=}, {T.reserve_curr=}")

# print positions
for p in traders["ETH-USD"].positions:
  print(str_dict(p))
  print("--")
#----------------------------------------------------------------------------}}}1
