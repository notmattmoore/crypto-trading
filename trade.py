# imports {{{1
# general
from   copy      import deepcopy
from   itertools import chain, repeat
from   time      import time
import os
import re
import myutils as mu

# ML
import numpy  as np
import pandas as pd

# custom modules
from   mylibs.utils import datetime_iso
import feat_engineering
#----------------------------------------------------------------------------}}}1

class RandDecider: # {{{1
  def __init__(self, rng=np.random, data_ticks=10):
    self.rng = rng
    self.data_ticks = data_ticks
  def __call__(self, candle):
    return self.rng.choice(["sell", "hold", "buy"])
#----------------------------------------------------------------------------}}}1

class SeqDecider: # {{{1
  def __init__(self, seq=None, fill="hold", data_ticks=10):
    if seq is None:
      seq = []

    self.data_ticks = data_ticks
    self.seq = chain(deepcopy(seq), repeat(fill))
  def __call__(self, *args, **kwargs):
    return next(self.seq)
#----------------------------------------------------------------------------}}}1

class SimTrader: # {{{1
  """
  A class to simulate a trading strategy on for backtesting. The class has data
  - balance_asset, balance_curr: balance of asset and currency to trade with.
  - borrow: whether to allow negative currency and asset balances (i.e. always trade
    on a recommendation, regardless of whether there is enough currency/asset).
  - min_buy_perc, max_buy_perc: minimum/maximum buy size, as a percentage of
    balance_curr.
  - ratio_buy, ratio_sell: how much of the trading balance to commit to a buy or
    sell recommendation.
  - stop_loss_perc, take_profit_perc: stop loss and take profit percentages.
  - take_profit_ratchet: percentage of slippage allowed in a take profit rachet
    strategy.
  - hold_length: duration (in number of calls) to hold a long position for.
  - history: whether to record history.
  - verbose: (0-2) how much information to print.
  """
  def __init__(  # {{{
    self, *,
    balance_asset=0, balance_curr=1, borrow=False, min_buy_perc=0, max_buy_perc=1,
    ratio_buy=1, ratio_sell=1, stop_loss_perc=np.inf, take_profit_perc=np.inf,
    take_profit_ratchet=0, hold_length=np.inf, fee_taker=0, fee_maker=0,
    decider_data_ticks=None, history=False, verbose=0,
  ):
    self.balance_asset = balance_asset
    self.balance_curr  = balance_curr

    self.borrow = borrow

    self.min_buy = min_buy_perc * self.balance_curr
    self.max_buy = max_buy_perc * self.balance_curr

    self.ratio_buy           = ratio_buy
    self.ratio_sell          = ratio_sell
    self.stop_loss_perc      = stop_loss_perc
    self.take_profit_perc    = take_profit_perc
    self.take_profit_ratchet = take_profit_ratchet
    self.hold_length         = hold_length
    self.fee_taker           = fee_taker
    self.fee_maker           = fee_maker
    self.verbose             = verbose

    # formatting for currency and asset output
    self._format_curr  = lambda x: f"{x:+.2f}"
    self._format_asset = lambda x: f"{x:+.6f}"

    # CoinBase doesn't support delayed sell orders, so that incurs a taker fee. It
    # does support 'bracket' orders, which can be used to place simultaneous
    # stop-loss and take-profit orders and thus incurs a maker fee.
    self.stop_loss_fee    = self.fee_maker
    self.take_profit_fee  = self.fee_maker
    self.delayed_sell_fee = self.fee_taker

    if history:
      # At each call, append a dictionary of the form
      #   { "balance_asset" : asset balance,
      #     "balance_curr"  : currence balance,
      #     "candle"        : candle recieved,
      #     "positions"     : current positions,
      #   }.
      # Also maintain a list of executed orders.
      self.history = []
      self.orders  = []
    else:
      self.history = False

    # Array of long positions we currently hold. Each position is represented by
    # a dictionary of the form
    #   {
    #     "stop_loss"     : None,  # the stop loss order info (a dict)
    #     "take_profit"   : None,  # the take profit order info (a dict)
    #     "delayed_sell"  : None,  # delayed sell order info (a dict)
    #     "init_asset"    : None,  # initial asset size of the position (+/-)
    #     "init_curr"     : None,  # initial currency size of the position (+/-)
    #     "balance_asset" : None,  # balance of the asset for this position, initially $init_asset
    #     "balance_curr"  : None,  # balance of the currency for this position, initially $init_curr
    #     "status"        : None,  # "open" or "closed"
    #   }
    self.positions = []
  #--------------------------------------------------------------------------}}}

  def __call__(self, candle, action="hold"): # {{{
    """
    Issue buy/sell orders based on price and trading action.
    """
    self._print_verbose("--")
    self._print_verbose(pd.DataFrame([candle]).to_string(index=False))

    # process positions.
    self._positions_minder(candle)

    # perform the trading action
    getattr(self, f"{action}_strat", self.fallback_strat)(candle)

    if self.history != False:
      self.history.append({
        "balance_asset" : self.balance_asset,
        "balance_curr"  : self.balance_curr,
        "candle"        : candle,
        "positions"     : deepcopy(self.positions),
      })

    bal_asset = self._format_asset(self.balance_asset)
    bal_curr  = self._format_curr(self.balance_curr)
    self._print_verbose(f"Balances | asset: {bal_asset}, curr: {bal_curr}")
  #--------------------------------------------------------------------------}}}

  def _positions_minder(self, candle):  # {{{
    """
    Execute each position, closing any that need closed. Finally, print the
    status of all the positions (if verbose).
    """
    # Execute local orders for each open position and close (possibly newly)
    # closed positions
    positions_new = []
    for position in self.positions:
      # Execute position
      if position["status"] == "open":
        position = self._position_execute_local(position, candle)

      # Close position
      if position["status"] == "closed":
        self._position_close(position)
      elif position["status"] == "open":
        positions_new.append(position)
      else:
        raise ValueError(f"invalid status for position: {position['status']=}\n{position=}")

    if len(self.positions) != 0 and self.verbose > 0:
      self._print_verbose("Positions:")
      for i, p in enumerate(self.positions, start=1):
        self._print_verbose(self.position_str(p, line_start=f"{i}. "))

    self.positions = positions_new
  #--------------------------------------------------------------------------}}}
  def _position_execute_local(self, position, candle):  # {{{
    """
    Update the status for position, tracking execution status of stop loss, take
    profit, and delayed sell orders and changes in asset and currency. If the
    position needs to be closed (e.g. due to a fully filled stop loss order),
    then mark it as closed.
    """
    if len(candle) == 0:
      self._print_verbose("Not executing _position_execute_local(): could not get a recent candle.")
      return position

    for order_kind in ["stop_loss", "take_profit", "delayed_sell"]:
      order = position[order_kind]
      # skip (possibly newly) closed positions and remote orders
      if position["status"] == "closed" or "id" in order:
        continue
      asset_delta, curr_delta, order_new = getattr(self, f"_{order_kind}_local")(position, candle)
      position["balance_asset"] += asset_delta
      position["balance_curr"]  += curr_delta
      position[order_kind] = order_new

      # If one of the orders has status "done" or the asset balance is close to 0,
      # then mark position as closed.
      if order_new.get("status") == "done":
        self._print_verbose2("Found done order:", order_new)
        position["status"] = "closed"
      elif round(position["balance_asset"], 8) <= 0:
        position["status"] = "closed"

    return position
  #--------------------------------------------------------------------------}}}
  def _position_close(self, position): # {{{
    """
    Close position. This means updating trader balances.
    """
    self.balance_asset += position["balance_asset"] - position["init_asset"]
    self.balance_curr  += position["balance_curr"] - position["init_curr"]
  #--------------------------------------------------------------------------}}}

  def _stop_loss_local(self, position, candle): # {{{
    """
    Local implementation of stop loss orders. Returns the triple
      (change in asset, change in currency, order info).
    """
    order = position["stop_loss"]

    # For a limit order type stop-loss, use the low price as the reference
    # price. For a to-be-placed market order type stop-loss, use the closing
    # price.
    if order.get("type") == "limit":
      price_cmp = order["price"]
      price_ref = candle["low"]
      sell_price = order["price"]
    elif order.get("type") == "stop-limit":
      price_cmp = order["stop_price"]
      price_ref = candle["low"]
      sell_price = order["price"]
    else:
      price_cmp = order["price"]
      price_ref = sell_price = candle["close"]

    if price_cmp < price_ref:  # stop loss not triggered
      return 0, 0, order

    # update the order and execute it
    order.update({"price": sell_price, "size": position["balance_asset"]})
    return self._order_execute(order, desc=f"stop loss ({price_ref=}, {order['price']=})", update_balances=False)
  #--------------------------------------------------------------------------}}}
  def _take_profit_local(self, position, candle): # {{{
    """
    Local implementation of stop loss orders. Returns the triple
      ( change in asset, change in currency, order info ).
    """
    order = position["take_profit"]

    # For a limit order type take-profit, use the high price as the reference
    # price. For a to-be-placed market order type take-profit, use the closing
    # price.
    if order.get("type") == "limit":
      price_ref = candle["high"]
      sell_price = order["price"]
    else:
      price_ref = sell_price = candle["close"]

    if price_ref >= order["price"]:  # take profit triggered
      order.update({"price": sell_price, "size": position["balance_asset"]})
      return self._order_execute(order, desc=f"take profit ({price_ref=}, {order['price']=})", update_balances=False)

    if order["price"] <= (1 + self.take_profit_ratchet) * price_ref:  # ratcheting triggered
      self._take_profit_ratchet_strat(position, price_ref)
      return 0, 0, position["take_profit"]

    # if neither take profit nor ratcheting is triggered, then do nothing and return
    return 0, 0, order
  #--------------------------------------------------------------------------}}}
  def _delayed_sell_local(self, position, candle):  # {{{
    """
    Local implementation of delayed sell orders. Returns the triple
    (change in asset, change in currency, order info ).
    """
    order = position["delayed_sell"]
    order["time"] -= 1

    if order["time"] > 0:  # order should not execute
      return 0, 0, order

    order.update({"price": candle["close"], "size": position["balance_asset"]})
    return self._order_execute(order, desc="delayed sell", update_balances=False)
  #--------------------------------------------------------------------------}}}
  def _take_profit_ratchet_strat(self, position, price):  # {{{
    """
    Implementation of a ratcheting take-profit strategy: once the price is within
    take_profit_ratchet percent of the take profit price, disable delayed selling and
    place a stop-limit order to lock in the profit.
    """
    price_low = (1 - self.take_profit_ratchet) * price
    price_high = (1 + self.take_profit_ratchet) * price

    position["take_profit"]["price"] = price_high
    if price_low >= position["stop_loss"]["price"]:  # update stop-loss if needed
      position["stop_loss"] = {
        "side"       : "sell",
        "type"       : "limit",
        "price"      : price_low,
        "size"       : position["balance_asset"],
        "fee"        : self.stop_loss_fee,
        "desc"       : "take profit ratchet",
      }
    position["delayed_sell"]["time"] = np.inf
  #--------------------------------------------------------------------------}}}
  def _order_execute(self, order, desc='', update_balances=True):  # {{{
    """
    Execute an order, modifying balances if $update_balances=True. $order should
    be a dictionary of the form
      order = {
        "side"  : "buy" or "sell",
        "type"  : "limit", "market", or "ratchet",
        "price" : price in currency units to transact at,
        "size"  : ammount of currency to buy with or asset to sell,
        "fee"   : fee charged for order (as a percentage),
      }
    Returns the triple
      ( change in asset, change in currency, order info ).
    """
    if desc != '':
      desc = ' ' + desc
    self._print_verbose2(f"Executing{desc} order:", order)
    if order["side"] == "buy":
      if not self.borrow and round(self.balance_curr - order["size"], 8) < 0:   # prevent rounding errors...
        print(
          f"ERROR {datetime_iso()}, {mu.func_name()}: unable to place order due to "
          f"insufficient funds: {order['size']=}, {self.balance_curr=}.\n  {order=}."
        )
        return 0, 0, order
      curr_delta  = -1 * order["size"]
      asset_delta = -1 * (1 - order["fee"]) * curr_delta / order["price"]
    elif order["side"] == "sell":
      if not self.borrow and round(self.balance_asset - order["size"], 8) < 0:  # prevent rounding errors...
        print(
          f"ERROR {datetime_iso()}, {mu.func_name()}: unable to place order due to "
          "insufficient funds: {order['size']=}, {self.balance_asset=}.\n  {order=}."
        )
        return 0, 0, order
      asset_delta = -1 * order["size"]
      curr_delta  = -1 * (1 - order["fee"]) * asset_delta * order["price"]
    else:
      raise ValueError(f"invalid value {order['side']=}\n  {order=}")

    if update_balances:
      self.balance_asset += asset_delta
      self.balance_curr  += curr_delta

    if self.history:
      self.orders.append(order)

    return asset_delta, curr_delta, order
  #--------------------------------------------------------------------------}}}

  def buy_strat(self, candle):  # {{{
    """
    Upon receiving a buy recommendation, market buy (if our balance is high
    enough) and place stop loss, take profit, and delayed sell orders.
    """
    self._print_verbose("BUY recommendation.")

    price = candle["close"]

    if self.borrow:
      curr_delta = self.max_buy
    else:
      curr_delta = min(self.ratio_buy * self.balance_curr, self.max_buy)
    if curr_delta < self.min_buy or curr_delta == 0:
      self._print_verbose(f"Insufficient funds to buy: min_buy={self.min_buy}, {curr_delta=}.")
      return

    # place a market buy order
    asset_delta, curr_delta, _ = self._order_execute({
      "side"  : "buy",
      "price" : price,
      "size"  : curr_delta,
      "fee"   : self.fee_taker,   # market order
      "desc"  : "buy rec",
    })

    # stop loss, take profit, and delayed sell orders
    stop_loss_order = {
      "side"  : "sell",
      "type"  : "limit",
      "price" : (1 - self.stop_loss_perc) * price,
      "size"  : asset_delta,
      "fee"   : self.stop_loss_fee,
      "desc"  : "stop loss",
    }
    take_profit_order = {
      "side"  : "sell",
      "type"  : "limit",
      "price" : (1 + self.take_profit_perc) * price,
      "size"  : asset_delta,
      "fee"   : self.take_profit_fee,
      "desc"  : "take profit",
    }
    delayed_sell_order = {
      "side"  : "sell",
      "type"  : "market",
      "size"  : asset_delta,
      "fee"   : self.delayed_sell_fee,
      "time"  : self.hold_length,
      "desc"  : "delayed sell",
    }

    self.positions.append({
      "stop_loss"     : stop_loss_order,
      "take_profit"   : take_profit_order,
      "delayed_sell"  : delayed_sell_order,
      "init_asset"    : asset_delta,
      "init_curr"     : curr_delta,
      "balance_asset" : asset_delta,
      "balance_curr"  : curr_delta,
      "status"        : "open",
    })
    self._print_verbose("New Position:", self.position_str(self.positions[-1], line_start='  '), sep='\n')
  #--------------------------------------------------------------------------}}}
  def sell_strat(self, candle):  # {{{
    """
    We use a one-sided buy only strategy. Sells are managed through stop loss
    orders and delayed sell orders, so we don't need a dedicated sell strategy.
    """
    self._print_verbose("SELL recommendation.")
  #--------------------------------------------------------------------------}}}
  def hold_strat(self, candle):  # {{{
    """
    We use a one-sided buy only strategy where holds are implicit. A hold
    recommendation is therefore never acted on.
    """
    self._print_verbose("HOLD recommendation.")
  #--------------------------------------------------------------------------}}}
  def fallback_strat(self, candle):  # {{{
    """
    Fallback strategy for when an invalid trading recommendation is made.
    """
    self._print_verbose("FALLBACK recommendation.")
  #--------------------------------------------------------------------------}}}

  def position_str(self, p, line_start=''):  # {{{
    """
    Returns a nicely formatted string describing position p. The first line of
    the string starts with $line_start, and subsequent lines are aligned to
    match.

    Example with line_start = "1. ":

    1. Orders   | stop_loss: 19123.51, take_profit: 20983.32, delayed_sell: 12
       Purchase | asset: 0.454312, curr: -10343.23, price: 20123.12, type: long
       Status   | asset: 0.123456, curr: 110.23, price: 23122.34, status: open
    """
    line_align = ' '*len(line_start)

    orders_stop_loss    = self._format_curr(p["stop_loss"]["price"])
    orders_take_profit  = self._format_curr(p["take_profit"]["price"])
    orders_stop_loss_type = f" ({p['stop_loss']['type']})" if "type" in p["stop_loss"] else ''
    orders_take_profit_type = f" ({p['take_profit']['type']})" if "type" in p["take_profit"] else ''
    orders_delayed_sell = p["delayed_sell"]["time"]
    r = (f"{line_start}Orders   | "
         f"stop_loss{orders_stop_loss_type}: {orders_stop_loss}, "
         f"take_profit{orders_take_profit_type}: {orders_take_profit}, "
         f"delayed sell: {orders_delayed_sell:.0f}")

    purch_asset = self._format_asset(p["init_asset"])
    purch_curr  = self._format_curr(p["init_curr"])
    purch_price = self._format_curr(-1 * p["init_curr"] / p["init_asset"])
    if p["init_asset"] > 0:
      purch_type = "long"
    elif p["init_curr"] > 0:
      purch_type = "short"
    else:
      purch_type = "unknown!"
    r += (f"\n{line_align}Purchase | "
          f"asset: {purch_asset}, curr: {purch_curr}, "
          f"price: {purch_price}, type: {purch_type}")

    # if the position has been partially or fully closed, then print status info
    if p["balance_asset"] != p["init_asset"] or p["balance_curr"] != p["init_curr"]:
      stat_asset = self._format_asset(p["balance_asset"])
      stat_curr  = self._format_curr(p["balance_curr"])
      if p["init_asset"] == p["balance_asset"] or p["init_curr"] == p["balance_curr"]:
        stat_price = self._format_curr(np.nan)
      else:
        stat_price = self._format_curr(-1 * (p["balance_curr"] - p["init_curr"]) / (p["balance_asset"] - p["init_asset"]))
      stat_status = p["status"]
      r += (f"\n{line_align}Status   | "
            f"asset: {stat_asset}, curr: {stat_curr}, "
            f"price: {stat_price}, status: {stat_status}")

    return r
  #--------------------------------------------------------------------------}}}
  def _print_verbose(self, *args, sep="\n  ", end='\n', level=1):  # {{{
    """
    Do nothing or print, depending on the value self.verbose and level
    """
    if level <= self.verbose:
      print(*args, sep=sep, end=end)
  #--------------------------------------------------------------------------}}}
  def _print_verbose2(self, *args, sep="\n  ", end='\n'):  # {{{
    self._print_verbose(*args, sep=sep, end=end, level=2)
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1

class TradeDecider: # {{{1
  """
  A class to wrap a trade predictor function. The class has data

  - model: the model. model.predict_proba(X) should return 0 (sell), 1 (hold), or 2
    (buy), where X is a 2D array.
  - pred_thresh: prediction threshold.
  - params_data: the dictionary of data parameters.
  - params_fe: the dictionary of feature engineering parameters.
  - data_ticks: the amount of data needed to make a single prediction after feature
    engineering.
  - symbol_pair: the trading symbol pair, used to prefix verbose output
  - logging: False or a filename.
  - verbose: (0-2) how much information to print.

  Calling TD(price) will return a string "sell", "hold", "buy", or "pass".
  """
  def __init__( # {{{
    self, *,
    model, pred_wrap=lambda x:x, params_data=None, params_fe=None, data_ticks=None,
    symbol_pair='', logging=False, verbose=0
  ):
    if params_data is None:
      params_data = {}
    if params_fe is None:
      params_fe = {}

    self.model       = model
    self.pred_wrap   = pred_wrap
    self.params_data = deepcopy(params_data)
    self.params_fe   = deepcopy(params_fe)

    if data_ticks is None:
      # Find the largest number that occurs amongst the feature names and use
      # data_resample and stride parameters to calculate how much data is needed
      # for a single prediction. Sharpe metrics double this number, which is why
      # we multiply by 2.
      feat_max_n = 2 * max(map(int, chain.from_iterable(iter(re.findall(r"\d+", f) for f in self.model.feature_names_in_))))
      data_ticks = self.params_fe.get("stride_ticks", 1) * feat_max_n
    self.data_ticks = data_ticks

    self.symbol_pair = symbol_pair
    self.verbose = verbose

    self.logging = logging
    if self.logging != False:
      self.logging = True
      self.log_filename = logging

    # Store last decision, probabilities, and features in dec, dec_prob, and X,
    # respectively.
    self.dec = 'uninitialized!'
    if self.params_fe["task"] == "binary":
      self._dec_prob_null = np.zeros(2)
    elif self.params_fe["task"] == "multiclass":
      self._dec_prob_null = np.zeros(3)
    self.dec_prob = self._dec_prob_null
    self.X = []
  #--------------------------------------------------------------------------}}}

  def __call__(self, data): # {{{
    """
    Process data and (if data is big enough) make a prediction. Based on the
    prediction, make a recommendation to "sell", "hold", or "buy". If the data
    is insufficient, then return "pass".
    """
    self.X = feat_engineering.prepare_data_trader(data, self.params_data, self.params_fe)
    if len(self.X) == 0:
      self.dec = "pass (insufficient data)"
      self.dec_prob = self._dec_prob_null
      return self.dec

    # use predict_proba by default, falling back to usual predict if it isn't available
    self._print_verbose2("Predicting on input", self.X.iloc[[-1]])
    pred_proba = getattr(self.model, "predict_proba", self.model.predict)(self.X.iloc[[-1]])
    [pred] = self.pred_wrap(pred_proba)
    self.dec_prob = pred_proba[0]

    if self.params_fe["task"] == "binary":
      self._print_verbose(", ".join(f"{t}: {p:.4g}" for (t,p) in zip(["Hold", "Buy"], self.dec_prob)))
      if pred == 0:
        self.dec = "hold"
      elif pred == 1:
        self.dec = "buy"
      else:
        self.dec = f"pass ({pred=})"
    elif self.params_fe["task"] == "multiclass":
      self._print_verbose(", ".join(f"{t}: {p:.4g}" for (t,p) in zip(["Sell", "Hold", "Buy"], self.dec_prob)))
      if pred == 0:
        self.dec = "sell"
      elif pred == 1:
        self.dec = "hold"
      elif pred == 2:
        self.dec = "buy"
      else:
        self.dec = f"pass ({pred=})"

    self._log()

    return self.dec
  #--------------------------------------------------------------------------}}}

  def _log(self):  # {{{
    """
    Log if logging is enabled.
    """
    if not self.logging:
      return

    if len(self.X) >= 1:
      features_df = self.X.iloc[[-1]]
    else:
      features_df = pd.DataFrame({'':["-- NO UPDATE --"]}, index=[''])

    # form decision dataframe
    if self.params_fe["task"] == "binary":
      dec_keys = ["hold", "buy"]
    elif self.params_fe["task"] == "multiclass":
      dec_keys = ["sell", "hold", "buy"]
    dec_prob = dict(zip(dec_keys, self.dec_prob.reshape(-1,1)))
    dec_df = pd.DataFrame(
      {"log date (local)": [datetime_iso()], **dec_prob, "rec": self.dec},
      index=features_df.index
    )

    # concatenate decision and features, reorder and rename columns
    log_df = pd.concat([features_df, dec_df], axis=1).fillna('')
    log_df = log_df[
      ["log date (local)"] + list(features_df.columns) + dec_keys + ["rec"]
    ].rename(columns={c:f"{c}".title() for c in dec_df.columns})

    log_entry = log_df.to_string()
    if getattr(self, "_log_prev_run", False):  # only print the header on first run
      log_entry = log_df.to_string().split('\n', maxsplit=1)[1]
    with open(self.log_filename, 'a') as f:
      f.write(log_entry + '\n')
    self._log_prev_run = True
  #--------------------------------------------------------------------------}}}
  def _print_verbose(self, *args, sep="\n  ", end='\n', level=1):  # {{{
    """
    Do nothing or print, depending on the value self.verbose and level
    """
    if level <= self.verbose:
      if self.symbol_pair != '' and len(args) > 0:
        args = chain([f"{self.symbol_pair}: {args[0]}"], args[1:])
      print(*args, sep=sep, end=end)
  #--------------------------------------------------------------------------}}}
  def _print_verbose2(self, *args, sep="\n  ", end='\n'):  # {{{
    self._print_verbose(*args, sep=sep, end=end, level=2)
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1

class CBTrader:  # {{{1
  """
  A class to implement a trading strategy on CoinBase. The class has data
  - decider: a TradeDecider() instance to be used for trade recommendations
    given price data.
  - CB: the CoinBase API wrapper.
  - symbol_asset, symbol_curr: the symbols for the asset and the trading
    currency (e.g. BTC and USD).
  - data_granularity, data_interval: the candle precision we sample from
    CoinBase. If neither are provided, default to 1 minute precision.
  - data_stale: how long a candle is considered "current" for. Defaults
    to 1.5*data_interval.
  - data_expire: whether to prune data that is no longer needed by the model.
    Defaults to True.
  - balance_asset, balance_curr: balance of asset and currency to trade with. By
    default the entire CoinBase balance is used.
  - reserve_curr, reserve_asset: reserve of currency or asset to keep in the account.
  - min_buy_perc, max_buy_perc: minimum and maximum buy sizes, as a percentage of the
    initial currency balance.
  - max_volume_buy_perc: average volume which a single trade should not exceed.
  - ratio_buy, ratio_sell: how much of the trading balance to commit to a buy or
    sell recommendation.
  - stop_loss_perc, take_profit_perc: stop loss and take profit percentages.
  - take_profit_ratchet: percentage of slippage allowed in a take profit ratchet
    strategy.
  - hold_length: duration (as multiples of $data_interval) to hold a long position for.
  - halt_config: triple of the form (time_window, pct_thresh, time_wait), interpreted
    as "if over duration time_window seconds asset price has percent change less
    than pct_thresh, then halt trading for time_wait seconds".
  - history: whether to record history.
  - logging: False, 4-tuple of filenames (data, orders, positions, verbose), or a
    base filename. In the case of a base filename, logs will be stored in files
    filename_data.log, filename_orders.log, filename_positions.log, and
    filename_verbose.log.
  - verbose: (0-2) how much information to print.
  """
  def __init__(  # {{{
      self, *,
      decider, CB, symbol_asset, symbol_curr, data_granularity=None,
      data_interval=None, data_stale=None, data_expire=True,
      balance_asset=np.inf, balance_curr=np.inf, reserve_curr=0, reserve_asset=0,
      min_buy_perc=0, max_buy_perc=1, max_volume_buy_perc=np.inf, ratio_buy=1,
      ratio_sell=1, stop_loss_perc=np.inf, take_profit_perc=np.inf,
      take_profit_ratchet=0, hold_length=np.inf, halt_config=None, history=False,
      logging=False, verbose=0,
  ):
    self.decider = decider
    self.CB      = CB

    self.symbol_asset = symbol_asset.upper()
    self.symbol_curr  = symbol_curr.upper()
    self.symbol_pair  = f"{symbol_asset}-{symbol_curr}"

    if data_granularity is None and data_interval is None:
      data_granularity, data_interval = "ONE_MINUTE", 60
    self.data_granularity, self.data_interval = self.CB._granularity_interval(granularity=data_granularity, interval=data_interval)

    if data_stale is None:
      data_stale = self.data_interval * 1.5
    self.data_stale = data_stale

    self.data_expire    = data_expire
    self.balance_asset  = balance_asset
    self.balance_curr   = balance_curr
    self.reserve_curr   = reserve_curr
    self.reserve_asset  = reserve_asset

    self.min_buy = min_buy_perc * self.balance_curr
    self.max_buy = max_buy_perc * self.balance_curr

    self.max_volume_buy_perc = max_volume_buy_perc
    self.ratio_buy           = ratio_buy
    self.ratio_sell          = ratio_sell
    self.stop_loss_perc      = stop_loss_perc
    self.take_profit_perc    = take_profit_perc
    self.take_profit_ratchet = take_profit_ratchet

    self.hold_length = hold_length * self.data_interval

    self.halt_config = halt_config

    if history:
      # At each call, append a dictionary of the form
      #   {
      #     "data_update"   : data_update,
      #     "balance_asset" : asset balance,
      #     "balance_curr"  : currency balance,
      #     "positions"     : current positions,
      #   }.
      self.history = []
    else:
      self.history = False

    if logging != False:
      self.logging = True
      if isinstance(logging, str):
        logging = map(lambda x,y: f"{x}_{y}.log", repeat(logging), ["data", "orders", "positions", "verbose"])
      (self.log_data_filename, self.log_orders_filename, self.log_positions_filename, self.log_verbose_filename) = logging
    else:
      self.logging = False

    self.verbose = verbose

    # formatting for currency and asset output
    self._format_asset, self._format_curr = self.CB.format_base_quote(self.symbol_pair)

    # Array of positions we currently hold.
    self._position_default = {
      "time"            : np.datetime64("NaT"), # the unix time the position was entered
      "stop_loss"       : { "price" : -np.inf,  # the stop loss order info
                            "local" : False },  #   whether to run this order locally
      "take_profit"     : { "price" : np.inf,   # the take profit order info (a dict)
                            "local" : False },
      "delayed_sell"    : { "time": np.inf,     # delayed sell order info (a dict)
                            "local" : False },
      "init_order"      : {},                   # the inital order that the position is based on
      "init_asset"      : np.nan,               # initial asset size of the position (+/-)
      "init_curr"       : np.nan,               # initial currency size of the position (+/-)
      "balance_asset"   : np.nan,               # balance of the asset for this position, initially $init_asset
      "balance_curr"    : np.nan,               # balance of the currency for this position, initially $init_curr
      "balances_update" : True,                 # True or False, whether to update global balances as position executes
      "status"          : "open",               # "open" or "closed"
      "desc"            : {},                   # structured description of the position, used for printing
    }
    self.positions = []

    # list of orders which we were unable to cancel and so will keep retrying
    self._cancel_retry = set()

    # initialize the data
    self.data = pd.DataFrame()
    self.data_update()
  #--------------------------------------------------------------------------}}}

  def __call__(self, trade_on_rec=True):  # {{{
    """
    Update price data, handle trading positions, and if $trade_on_rec is True,
    then get a recommendation from self.decider and act on it.
    """
    self._print_verbose(f"-- {datetime_iso()} --")

    # Update the data, process positions.
    data_update_latest = self.data_update()
    self._positions_minder()

    # Get a trading recommendation, determine whether to trade, and act on it.
    trade_rec = self.decider(self.data_most_recent_block())
    if not trade_on_rec:
      self._print_verbose(f"{trade_on_rec=}, so not trading.")
    elif self._trade_halt():
      print(
        f"WARNING {datetime_iso()}, {mu.func_name()}: {self.symbol_pair} trading "
        f"halted due to percent change of less than {self.halt_config[1]} over "
        f"{self.halt_config[0]} seconds. Will resume at "
        f"{datetime_iso(self._halt_resume)}."
      )
    else:
      getattr(self, f"{trade_rec}_strat", self.fallback_strat)()

    # Try to cancel any orders we previously failed to cancel.
    self._retry_cancels()

    if self.history != False:
      self.history.append({
        "data_update"   : data_update_latest,
        "balance_asset" : self.balance_asset,
        "balance_curr"  : self.balance_curr,
        "positions"     : deepcopy(self.positions),
        "rec"           : trade_rec,
      })

    bal_asset = self._format_asset(self.balance_asset)
    bal_curr  = self._format_curr(self.balance_curr)
    self._print_verbose(f"Balances | {self.symbol_asset}: {bal_asset}, {self.symbol_curr}: {bal_curr}")

    self._log_data(data_update_latest)

    # Expire old data and try to get get missing candles
    self.data_do_expire()
    self.data_get_missing()
  #--------------------------------------------------------------------------}}}

  def data_update(self, direction="future", size=np.inf):  # {{{
    """
    Update candle data. Detect missing data and expire data if self.data is too
    long. Returns the data update. Parameters:
    - direction: if "future", then get candles starting from the most recent
        candle in self.data to the current time. If "past", get $size-many
        candles in the past before the first candle in self.data.
    - size: Obtain min(self.decider.data_ticks, size)-many candles.
    """
    if len(self.data) == 0:  # on initialization
      time_end = int(time())
      time_start = time_end - self.data_interval * min(self.decider.data_ticks, size)
    elif direction == "future":
      time_end = int(time())
      time_start = max(self.data.index.max() + self.data_interval // 2, time_end - self.data_interval * size)
    elif direction == "past":
      time_end = self.data.index.min() - self.data_interval // 2
      time_start = time_end - self.data_interval * min(self.decider.data_ticks, size)

    # Get the new data, index it by timestamp, merge it with existing data, and
    # then reindex it.
    self._data_update = self.CB.candles(
      self.symbol_pair, granularity=self.data_granularity,
      interval=self.data_interval, start=time_start, end=time_end
    )
    self._data_update = self._data_update.set_index("unix", verify_integrity=True).sort_index()
    self.data = pd.concat([self._data_update, self.data])
    index_full = np.arange(
      min(self.data.index, default=0), max(self.data.index, default=-1) + 1,
      step=self.data_interval
    )
    self.data = self.data.reindex(index_full).sort_index(ascending=False)

    # If we didn't get enough data, then recurse and update from the bottom.
    # This mostly happens on initialization, but it can also happen if we have
    # been unable to update for a long time. Use try/except in case we can't get
    # enough.
    if len(self.data) < self.decider.data_ticks:
      try:
        self.data_update(direction="past", size=self.decider.data_ticks - len(self.data))
      except Exception as err:
        print(
          f"ERROR {datetime_iso()}, {mu.func_name()}: failed to get data update due "
          f"to exception\n  {err}."
        )

    self._print_verbose(self._data_update.to_string(index=False))
    return self._data_update
  #--------------------------------------------------------------------------}}}
  def data_most_recent_block(self, size=None, stale=None):  # {{{
    """
    Return the most recent block of size $size (defaults to
    self.decider.data_ticks). If the most recent timestamp is > $stale old
    or there is not enough data to meet the request, then return an empty
    DataFrame.
    """
    if size is None:
      size = self.decider.data_ticks
    if stale is None:
      stale = self.data_stale

    if time() - self.data.index[0] > stale:
      self._print_verbose(f"Data is stale by {time() - self.data.index[0]:.0f}s.")
      return pd.DataFrame(columns=self.data.columns)
    if len(self.data) < size:
      self._print_verbose(f"Insufficient data for recommendation: have {len(self.data)}, but need {size}.")
      return pd.DataFrame(columns=self.data.columns)

    return self.data.iloc[:size]
  #--------------------------------------------------------------------------}}}
  def data_get_missing(self):  # {{{
    """
    Attempt to fetch missing candles from self.data.
    """
    # Get the NA indices. Don't try to get indices which we previously failed to get.
    index_na_prev = getattr(self, "_index_na", [])
    index_na = self._index_na = self.data[self.data.isna().any(axis=1)].index
    index_na = index_na.difference(index_na_prev)
    if len(index_na) == 0:
      if len(index_na_prev) != 0:
        self._print_verbose(f"Missing {len(index_na_prev)} data entries ({len(index_na_prev) / len(self.data):.2%}) (same as last run)")
      return
    self._print_verbose(
      f"Missing {len(index_na)} data entries ({len(index_na) / len(self.data):.2%}):",
      self.data.loc[index_na], sep='\n'
    )

    # Find consecutive blocks of missing indicies and try to get them again. The
    # index is ordered from large to small.
    end = index_na[0]
    for i in range(len(index_na)):
      if i == len(index_na) - 1 or index_na[i] - index_na[i+1] != self.data_interval:
        candles = self.CB.candles(
          self.symbol_pair, granularity=self.data_granularity,
          interval=self.data_interval, start=index_na[i], end=end
        )
        end = index_na[i+1] if i < len(index_na)-1 else None
        if len(candles) != 0:
          candles = candles.set_index("unix", verify_integrity=True)
          self.data.loc[candles.index] = candles
          self._print_verbose(f"Recovered data:\n{candles}")
  #--------------------------------------------------------------------------}}}
  def data_do_expire(self):  # {{{
    """
    Delete rows from self.data so that it is of length self.decider.data_ticks if
    self.data_expire == True.
    """
    if self.data_expire and len(self.data) > self.decider.data_ticks:
      self.data = self.data.iloc[:self.decider.data_ticks]
  #--------------------------------------------------------------------------}}}

  def _positions_minder(self):  # {{{
    """
    Update the status of remote orders in each position, then get a candle and
    and execute each position, closing any that need closed. Finally, print the
    status of all the positions (if verbose).
    """
    # Update the status of remote orders in positions.
    self.positions = list(map(self._position_update, self.positions))

    candle = self.data_most_recent_block(size=1)
    if len(candle) > 0:
      candle = candle.iloc[0]
    else:
      print(
        f"WARNING {datetime_iso()}, {mu.func_name()}: could not get a recent "
        f"candle, so not executing local orders."
      )
      return

    # Execute local orders for each open position, run the ratcheting strategy (if
    # enabled), and close (possibly newly) closed positions.
    positions_new = []
    for position in self.positions:
      if position["status"] == "open":
        position = self._position_execute_local(position, candle)
      if position["status"] == "open" and self.take_profit_ratchet > 0:
        position = self._take_profit_ratchet_strat(position, candle)
      if position["status"] == "closed":
        self._position_close(position)

      position["desc"] = self._position_desc(position)

      if position["status"] == "open":  # if it's still open, then keep it
        positions_new.append(position)

    if len(self.positions) != 0:
      if 1 <= self.verbose or self.logging != False:
        self._print_verbose(
          "Positions:",
          *[self.position_str(p, line_start=f"{i}. ") for i,p in enumerate(self.positions, start=1)],
          sep='\n'
        )
      self._log_positions(self.positions)

    self.positions = positions_new
  #--------------------------------------------------------------------------}}}
  def _position_update(self, position):  # {{{
    """
    Update the status for position, tracking execution status of stop loss, take
    profit, and delayed sell orders and changes in asset and currency. This applies
    only to remote orders since local orders are implemented as market orders and
    performed at the time of execution. If a position has remote status "done", then
    mark it as closed.
    """
    # If the initial order hasn't been filled, then try to update it. If successful,
    # then finish forming the position, otherwise return.
    if position["init_order"].get("status", "open") == "open":
      _, _, order_info = self._order_update(position["init_order"])
      if order_info.get("status", '') == "filled":
        self._print_verbose2("Found filled initial order:", order_info)
        self._log_order(order_info)
        position = self.position_form(order_info, position_init=position)
      else:
        return position

    # We will recalculate the active balance right now, so reset it.
    position["balance_asset"] = position["init_asset"]
    position["balance_curr"]  = position["init_curr"]
    for order_type in ["stop_loss", "take_profit", "delayed_sell"]:
      order_info = position[order_type]
      if order_info.get("local", False) or "order_id" not in order_info.keys():  # skip locally executing orders
        continue
      asset_delta, curr_delta, order_info_new = self._order_update(order_info)
      position["balance_asset"] += asset_delta
      position["balance_curr"]  += curr_delta
      position[order_type].update(order_info_new)

      # If one of the orders has done status (i.e. not "open") then mark position as
      # closed.
      if order_info_new.get("status", "open") != "open":
        position["status"] = "closed"
        self._print_verbose2("Found done order:", order_info_new)
        self._log_order(order_info_new)

    return position
  #--------------------------------------------------------------------------}}}
  def _position_execute_local(self, position, candle):  # {{{
    """
    Execute local orders in the position.
    """
    for order_type in ["stop_loss", "take_profit", "delayed_sell"]:
      order = position[order_type]
      # skip closed positions and remote orders
      if position["status"] == "closed" or not order.get("local", False):
        continue
      asset_delta, curr_delta, order_info_new = getattr(self, f"_{order_type}_local")(position, candle)
      position["balance_asset"] += asset_delta
      position["balance_curr"]  += curr_delta
      position[order_type].update(order_info_new)

      # If one of the orders has done status (i.e. not "open") then mark
      # position as closed.
      if "status" in order_info_new.keys() and order_info_new["status"] != "open":
        self._print_verbose2("Found done order:", order_info_new)
        position["status"] = "closed"

    return position
  #--------------------------------------------------------------------------}}}
  def _position_close(self, position):  # {{{
    """
    Close position. This means updating trader balances and canceling all pending
    remote orders. If we are unable to cancel an order, then add the order to the
    array _cancel_order_retry so that we can keep trying to cancel
    it.
    """
    # Try to cancel any open remote orders.
    _, cancel_results = self._position_cancel_remote(position)

    # Add the asset and currency proceeds of the postion to the overall balances
    # and if any orders didn't cancel add them to the cancel retry list.
    for o_id, res in cancel_results.items():
      if res["active"] == False and position["balances_update"]:  # order was cancelled
        self.balance_asset += res["order_info"].get("base_delta", 0)
        self.balance_curr  += res["order_info"].get("quote_delta", 0)
      else:  # order was not cancelled
        print(
          f"WARNING {datetime_iso()}, {mu.func_name()}: adding order\n  order_id: "
          f"{o_id}\n  {res['order_info']}\nto _cancel_retry."
        )
        self._cancel_retry.add(o_id)
  #--------------------------------------------------------------------------}}}
  def _order_update(self, order_info):  # {{{
    """
    Get a status update for $order_info. Returns the triple
      (change in asset, change in currency, order info).
    On error returns (0, 0, {}).
    """
    order_info_new, _ = self.CB.order_info(order_info["order_id"])
    return order_info_new.get("base_delta", 0), order_info_new.get("quote_delta", 0), order_info_new
  #--------------------------------------------------------------------------}}}

  def _stop_loss_local(self, position, candle):  # {{{
    """
    Local implementation of stop loss orders. Given a candle, calculate a
    reference price for it (to mitigate noise). Using this price, see if
    position["stop loss"] should execute. Returns the triple
      (change in asset, change in currency, order info).
    """
    order = position["stop_loss"]
    price_ref = candle["close"]

    if order["price"] < price_ref:  # stop loss not triggered
      return 0, 0, order

    # Don't bother selling if we are just going to buy again. Instead, cancel all
    # remote orders and re-form the orders from the position.
    if self.decider.dec.lower().startswith("buy"):
      cancel_status, _ = self._position_cancel_remote(position)
      if not cancel_status:  # do nothing if we failed to cancel the remote orders
        return 0, 0, order
      position = self._position_form_orders(position, price=candle["close"])
      return 0, 0, position["stop_loss"]

    return self._position_order_execute_local(position, "stop_loss", f"stop loss (price_ref={self._format_curr(price_ref)})")
  #--------------------------------------------------------------------------}}}
  def _take_profit_local(self, position, candle):  # {{{
    """
    Local implementation of take profit orders. Given a candle, calculate a
    reference price for it (to mitigate noise). Using this price, see if
    position["take profit"] should execute. Returns the triple
      (change in asset, change in currency, order info).
    """
    order = position["take_profit"]
    price_ref = candle["close"]

    if price_ref < order["price"]:  # take profit not triggered
      return 0, 0, order

    return self._position_order_execute_local(position, "take_profit", f"take profit (price_ref={self._format_curr(price_ref)})")
  #--------------------------------------------------------------------------}}}
  def _take_profit_ratchet_strat(self, position, candle):  # {{{
    """
    Implementation of a ratcheting take-profit strategy: when reference price is
    within take_profit_ratchet percent of the take profit price, cancel outstanding
    orders and place a new bracket order to lock in the profits.
    """
    # Calculate new take profit and stop loss prices.
    price_take_profit = position["take_profit"]["price"]
    price_low = max(
      (1 - self.take_profit_ratchet) * price_take_profit,
      position["stop_loss"]["price"]
    )
    price_high = (1 + self.take_profit_ratchet) * price_take_profit
    price_ref = candle["close"]

    # Only ratchet if the percent difference between take_profit_price and price_ref
    # is <= take_profit_ratchet.
    if (1 + self.take_profit_ratchet) * price_ref < price_take_profit:
      return position

    # XXX
    print(f"{datetime_iso()}: ratcheting on position\n{mu.str_dict(position, prefix='  ')}.")
    self._print_verbose2(f"{datetime_iso()}: ratcheting on position\n{mu.str_dict(position, prefix='  ')}.")

    # Try to cancel outstanding remote orders. If any fail, bail out.
    cancel_success, _ = self._position_cancel_remote(position)
    if not cancel_success:
      return position

    # Run the strategy locally if necessary.
    if position["take_profit"].get("local", False):
      position["stop_loss"]["local"] = True
      position["take_profit"]["price"] = price_high
      position["stop_loss"]["price"] = price_low
      return position

    # Form new stop_loss, take_profit, and delayed sell orders.
    position = self._position_form_orders(
      position, price_take_profit=price_high, price_stop_loss=price_low,
      time_delayed_sell=position["delayed_sell"]["time"]
    )
    position["take_profit"]["type"] = "ratchet" # XXX

    return position
  #--------------------------------------------------------------------------}}}
  def _delayed_sell_local(self, position, candle):  # {{{
    """
    Local implementation of delayed sell orders. If the time in
    position["delayed_sell"]["time"] has passed, then execute the order. Returns
    the triple
      (change in asset, change in currency, order info).
    """
    order = position["delayed_sell"]

    late = time() - order["time"]
    if late < 0:  # order should not execute
      return 0, 0, order

    # don't sell if we are going to immediately buy again, instead update the delayed sell time
    if self.decider.dec.lower().startswith("buy"):
      order["time"] = time() + self.hold_length
      return 0, 0, order

    return self._position_order_execute_local(position, "delayed_sell", f"delayed sell ({late=:.6f})")
  #--------------------------------------------------------------------------}}}
  def _position_order_execute_local(self, position, order_key, order_desc):  # {{{
    """
    Execute the order $position[$order_key] by cancelling any remote orders the
    position has and placing a market sell order for the balance of the
    position. This function is meant to be called by the various _local()
    functions. Returns the triple
      (change in asset, change in currency, order info).
    """
    # Try to cancel outstanding remote orders. If any fail, bail out.
    cancel_success, _ = self._position_cancel_remote(position)
    if not cancel_success:
      return 0, 0, position[order_key]

    # Place a market sell order.
    self._print_verbose2(f"Executing {order_desc} order:", position[order_key])
    market_order_info, _ = self.CB.market_order_conf({
      "symbol_pair" : self.symbol_pair,
      "side"        : "sell",
      "type"        : "market",
      "base_size"   : position["balance_asset"],
    })
    self._log_order(market_order_info)
    if len(market_order_info) != 0:
      self._print_verbose2(f"Placed market sell ({order_desc}):", market_order_info)

    return market_order_info.get("base_delta", 0), market_order_info.get("quote_delta", 0), market_order_info
  #--------------------------------------------------------------------------}}}
  def _trade_halt(self):  # {{{
    """
    If $halt_config has been set, then check to see if the asset price has changes
    less than the threshold over the specified time window. If it has, then pause
    trading for the specified wait time.
    """
    if time() <= getattr(self, "_halt_resume", -np.inf):  # not yet time to resume trading
      return True

    if self.halt_config is None:
      return False
    time_window, pct_thresh, time_wait = self.halt_config

    # Data is indexed from new -> old. Calculate percent change over the time window.
    window = self.data.loc[ : time()-time_window]
    if len(window) == 0:  # insufficient data to determine percent change
      return False
    pct_change = window["close"].iloc[[-1,0]].pct_change().iloc[-1]
    if pct_change <= pct_thresh:
      self._halt_resume = time() + time_wait
      return True

    return False
  #--------------------------------------------------------------------------}}}

  def order_cancel(self, order_ids):  # {{{
    """
    Attempt to cancel all orders in $order_ids. Returns the pair
      (status, results),
    where status is False if any failed to cancel and otherwise True, and results
    are the replies from the exchange. $order_ids can be an array of ids or a single
    order id.
    """
    cancel_results = self.CB.order_cancel(order_ids)
    all_cancelled = True
    for o_id, res in cancel_results.items():
      if res["active"]:  # order was not cancelled
        all_cancelled = False
        print(
            f"ERROR {datetime_iso()}, {mu.func_name()}: failed to cancel order\n"
            f"  order_id: {o_id}\n  {res['order_info']}.\nAdding order to "
            "_cancel_retry."
        )
        self._cancel_retry.add(o_id)
        from IPython import embed; embed()  # XXX
    return all_cancelled, cancel_results
  #--------------------------------------------------------------------------}}}
  def _position_cancel_remote(self, position):  # {{{
    """
    Attempt to cancel all remote orders in position. Returns the pair
      (status, results),
    where status is False if any failed to cancel, and otherwise True, and results
    are the replies from the exchange.
    """
    # Collect the remote order IDs.
    order_ids = []
    for order_type in ["stop_loss", "take_profit", "delayed_sell"]:
      order_info = position[order_type]
      if "order_id" in order_info.keys():
        order_ids.append(order_info["order_id"])

    # Try to cancel them. Return the results.
    cancel_success, cancel_replies = self.order_cancel(order_ids)
    if cancel_success:
      self._print_verbose2(
        f"Cancelled remote orders {order_ids} in position\n"
        f"{mu.str_dict(position, prefix='  ')}."
      )
    else:
      print(
        f"WARNING {datetime_iso()}, {mu.func_name()}: could not cancel position\n",
        mu.str_dict(position, sep='\n  '), '.', sep=''
      )
      from IPython import embed; embed()  # XXX
    return cancel_success, cancel_replies
  #--------------------------------------------------------------------------}}}
  def _retry_cancels(self):  # {{{
    """
    For each order in self._cancel_retry, try to cancel it again. If we fail to
    cancel it then keep it on the list, otherwise remove it.
    """
    _, cancel_results = self.order_cancel(list(self._cancel_retry))
    for o_id, res in cancel_results.items():
      if not res["active"]:  # order was cancelled
        self._cancel_retry.remove(o_id)
  #--------------------------------------------------------------------------}}}

  def buy_strat(self):  # {{{
    """
    Upon receiving a buy recommendation, market buy (if our balance is high enough)
    and place stop loss, take profit, and delayed sell orders.

    CoinBase doesn't implement OCO (one cancels other) orders, so we can't place all
    these orders on the exchange. Instead, we place the take profit order (with
    expiration date according to self.hold_length) (unless ratchet trading is being
    used) and then run stop loss and delayed sell orders locally.
    """
    self._print_verbose("BUY recommendation.")

    # Calculate how much to buy.
    balance_curr_total = self.CB.balances([self.symbol_curr])[self.symbol_curr]["available"]
    volume_curr_mean = self.data_most_recent_block()[["close", "volume"]].mean().product(min_count=1)
    if pd.isna(volume_curr_mean):
      volume_curr_mean = np.inf
    curr_delta = min(
      self.ratio_buy * (balance_curr_total - self.reserve_curr),
      self.ratio_buy * self.balance_curr,
      self.max_volume_buy_perc * volume_curr_mean, self.max_buy
    )
    if curr_delta < self.min_buy or curr_delta <= 0:
      self._print_verbose(
        f"Insufficient funds to buy: min_buy={self._format_curr(self.min_buy)}, "
        f"curr_delta={self._format_curr(curr_delta)}, "
        f"balance_curr_total={self._format_curr(balance_curr_total)}, "
        f"reserve_curr={self._format_curr(self.reserve_curr)}."
      )
      return

    # Place the order.
    order_id, r = self.CB.order_place({
      "symbol_pair" : self.symbol_pair,
      "side"        : "buy",
      "type"        : "market",
      "quote_size"  : curr_delta,
    })
    if order_id is None: # Failed to place the market order.
      print(
        f"ERROR {datetime_iso()}, {mu.func_name()}: failed to place market order. "
        f"Received reply\n  {r}."
      )
      from IPython import embed; embed()  # XXX
      return

    # Get confirmation of the order.
    order_info, _ = self.CB.order_info(order_id)
    if order_info.get("status") == "filled":
      self._print_verbose2("Placed market buy:", order_info)
      self._log_order(order_info)
    else:  # The order hasn't been filled yet, so we'll try to update it later.
      self._print_verbose2(
        f"WARNING {datetime_iso()}, {mu.func_name()}: failed to get confirmation of "
        f"market buy: {order_id}."
      )
      order_info = { "order_id": order_id, "status": "open" }

    # Establish the new position.
    position = self.position_form(order_info)
    self.positions.append(position)
    self._print_verbose("Added position", self.position_str(position, line_start="  "), sep='\n')
  #--------------------------------------------------------------------------}}}
  def sell_strat(self):  # {{{
    """
    We use a one-sided buy only strategy. Sells are managed through stop loss
    orders and delayed sell orders, so we don't need a dedicated sell strategy.
    """
    self._print_verbose("SELL recommendation.")
  #--------------------------------------------------------------------------}}}
  def hold_strat(self):  # {{{
    """
    We use a one-sided buy only strategy, so a hold recommendation is never
    received.
    """
    self._print_verbose("HOLD recommendation.")
  #--------------------------------------------------------------------------}}}
  def fallback_strat(self):  # {{{
    """
    Fallback strategy for when an invalid trading recommendation is made.
    """
    self._print_verbose("FALLBACK recommendation.")
  #--------------------------------------------------------------------------}}}

  def position_form(self, init_order, position_init=None):  # {{{
    """
    Given an initial order and partially filled position dictionary, complete all
    other position fields. This includes the various other orders, possibly
    submitting them to the exchange. Returns the fully completed position. If the
    initial order is not settled, then most fields will be the default.
    """
    if position_init is None:
      position_init = {}

    # Merge together the default position and the initial position.
    position = deepcopy(self._position_default)
    for (k,v) in position_init.items():
      if isinstance(v, dict):
        position[k].update(v)
      else:
        position[k] = v

    if np.isnan(position.get("time", np.datetime64("NaT"))):
      position["time"] = time()
    position["init_order"] = init_order

    # If the initial order is settled then calculate asset and currency deltas and
    # price, update the balances, and form the take profit, stop loss, and delayed
    # sell orders.
    if init_order.get("status", '') == "filled":
      asset_delta, curr_delta = init_order["base_delta"], init_order["quote_delta"]
      self.balance_asset += asset_delta
      self.balance_curr  += curr_delta
      position["init_asset"] = position["balance_asset"] = asset_delta
      position["init_curr"]  = position["balance_curr"]  = curr_delta
      if asset_delta != 0:
        price = -1 * curr_delta / asset_delta   # one of curr_delta or asset_delta is negative, but price should be positive
      else:
        price = np.nan

      # Form the stop loss, take profit, and delayed sell orders.
      position = self._position_form_orders(position, price=price)

    position["desc"] = self._position_desc(position)

    return position
  #--------------------------------------------------------------------------}}}
  def _position_form_orders(  # {{{
    self, position,
    *, price=None, price_take_profit=None, price_stop_loss=None,
    time_delayed_sell=None
  ):
    """
    Form the take profit, stop loss, and delayed sell orders for the position.
    Returns the new position. position must have the key "balance_asset" set.
    """
    # Default values and sanity checking.
    if price is None:
      price = np.nan
    elif price_take_profit is not None:
      print(
        f"WARNING {datetime_iso()}, {mu.func_name()}: both price and "
        f"price_take_profit specified: {price=}, {price_take_profit=}. Will ignore "
        "price."
      )
    elif price_stop_loss is not None:
      print(
        f"WARNING {datetime_iso()}, {mu.func_name()}: both price and "
        f"price_stop_loss specified: {price=}, {price_stop_loss=}. Will ignore "
        "price."
      )
    if price_take_profit is None:
      price_take_profit = (1 + self.take_profit_perc) * price
    if price_stop_loss is None:
      price_stop_loss = (1 - self.stop_loss_perc) * price
    if time_delayed_sell is None:
      time_delayed_sell = time() + self.hold_length

    position = deepcopy(position)

    # begin forming orders
    take_profit  = {"price": price_take_profit}
    stop_loss    = {"price": price_stop_loss}
    delayed_sell = {"time": time_delayed_sell, "local": True}

    # Fill out and submit a bracket order. This fulfills the role of both the
    # take-profit and stop-loss orders.
    if np.isfinite(price_take_profit) and np.isfinite(price_stop_loss):
      take_profit.update({
        "symbol_pair" : self.symbol_pair,
        "side"        : "sell",
        "type"        : "bracket",
        "stop_price"  : price_stop_loss,
        "base_size"   : position["balance_asset"],
      })
      take_profit_id, r = self.CB.order_place(take_profit)
      if take_profit_id is not None:  # Successfully placed order.
        take_profit["order_id"] = take_profit_id
        take_profit_info, _ = self.CB.order_info(take_profit_id)
        take_profit.update(take_profit_info)
        stop_loss["local"] = False
        self._print_verbose2("Placed bracket sell:", take_profit_info)
        self._log_order(take_profit_info)
      else:  # We failed to place a remote order, so run locally instead.
        print(
          f"ERROR {datetime_iso()}, {mu.func_name()}: failed to place bracket sell "
          f"order\n  {take_profit}.\nRecieved reply\n  {r}. \nWill run locally "
          "instead."
        )
        take_profit["local"] = stop_loss["local"] = True
        from IPython import embed; embed()  # XXX

    # Update the position and return.
    position.update({
      "stop_loss"     : stop_loss,
      "take_profit"   : take_profit,
      "delayed_sell"  : delayed_sell,
      })
    return position
  #--------------------------------------------------------------------------}}}

  def _position_desc(self, p):  # {{{
    """
    Form a dictionary of strings describing various things about the position.
    """
    desc = {
      "asset" : self._format_asset(p["balance_asset"]),
      "curr"  : self._format_curr(p["balance_curr"]),
      "price" : self._format_curr(-1 * p["init_curr"] / p["init_asset"])
    }

    if not np.isnan(p["time"]):
      desc["date"] = datetime_iso(p["time"])
    else:
      desc["date"] = str(np.datetime64("NaT"))

    for order in ["stop_loss", "take_profit"]:
      desc[order] = self._format_curr(p[order]['price'])
      if "type" in p[order].keys():  # use an acronym of what the order type is
        desc[order] += " (" + ''.join(r[0] for r in re.split("[\s\-_]", p[order]["type"])) + ')'

    if np.isfinite(p["delayed_sell"]["time"]):
      desc["delayed_sell"] = datetime_iso(p["delayed_sell"]["time"])
    else:
      desc["delayed_sell"] = str(np.datetime64("NaT"))

    return desc
  #--------------------------------------------------------------------------}}}
  def positions_desc_df(self, show_profit=False):  # {{{
    """
    Return a DataFrame containing descriptions of all the positions. Suitable for
    printing a more concise description than position_str().
    """
    # Use a mapping to translate the description keys into the column headings.
    mapping = {
      "date"         : "Position Date",
      "asset"        : "Asset",
      "curr"         : "Curr",
      "price"        : "Price",
      "stop_loss"    : "Stop Loss",
      "take_profit"  : "Take Profit",
      "delayed_sell" : "Expiration Date",
    }
    P = pd.DataFrame({k: [p["desc"].get(k, '') for p in self.positions] for k in mapping.keys()}) \
             .rename(columns=mapping)

    # Calculate profit (absolute and percentage). Depending on whether the position
    # is long or short, either the asset or currency balance in the position will be
    # negative. We therefore must take the *sum* of the position value and the
    # balance to calculate the profit.
    if show_profit:
      asset_price = self.data_most_recent_block(size=1, stale=np.inf)["close"].mean()
      P_bals = pd.DataFrame({k: [p[k] for p in self.positions] for k in ["balance_asset", "balance_curr"]})
      P_profit = asset_price * P_bals["balance_asset"] + P_bals["balance_curr"]
      P_profit_perc = P_profit / P_bals["balance_curr"].abs()
      P["Profit"] = P_profit.map(self._format_curr) + ' ' + P_profit_perc.map(lambda s: f"({s:.2%})")

    return P
  #--------------------------------------------------------------------------}}}
  def position_str(self, p, line_start=''):  # {{{
    """
    Returns a nicely formatted string describing position p. The first line of
    the string starts with $line_start, and subsequent lines are aligned to
    match.

    Example with line_start = "1. ":

    1. Orders   | stop_loss: 19123.51, take_profit: 20983.32, delayed_sell: 12
       Purchase | BTC: 0.454312, USD: -10343.23, price: 20123.12, type: long
       Status   | BTC: 0.123456, USD: 110.23, price: 23122.34, status: open
    """
    line_align = ' ' * len(line_start)

    # Information about the various orders.
    r = f"{line_start}Orders   | "
    r += ", ".join(f"{order.replace('_', ' ')}: {p['desc'][order]}" for order in ["stop_loss", "take_profit", "delayed_sell"])

    # Information about the purchase.
    purch_asset = self._format_asset(p["init_asset"])
    purch_curr  = self._format_curr(p["init_curr"])
    if p["init_asset"] > 0:
      purch_type = "long"
    elif p["init_curr"] > 0:
      purch_type = "short"
    else:
      purch_type = "unknown!"
    r += (f"\n{line_align}Purchase | "
          f"{self.symbol_asset}: {purch_asset}, {self.symbol_curr}: {purch_curr}, "
          f"price: {p['desc']['price']}, type: {purch_type}, time: {p['desc']['date']}")

    # If the position is not open, then print status info.
    if p["status"].lower() != "open":
      if p["init_asset"] != p["balance_asset"]:
        stat_price = self._format_curr(-1 * (p["balance_curr"] - p["init_curr"]) / (p["balance_asset"] - p["init_asset"]))
      else:
        stat_price = self._format_curr(np.nan)
      stat_status = p["status"]
      r += (f"\n{line_align}Status   | "
            f"{self.symbol_asset}: {p['desc']['asset']}, {self.symbol_curr}: {p['desc']['curr']}, "
            f"price: {stat_price}, status: {stat_status}")

    return r
  #--------------------------------------------------------------------------}}}

  def _print_verbose(self, *args, sep="\n  ", end='\n', level=1):  # {{{
    """
    Do nothing or print or log, depending on the value self.verbose level, and
    self.logging.
    """
    if level <= self.verbose:
      if len(args) > 0:
        args = chain([f"{self.symbol_pair}: {args[0]}"], args[1:])
      print(*args, sep=sep, end=end)

    if self.logging != False:
      with open(self.log_verbose_filename, 'a') as f:
        f.write(sep.join(map(str, args)) + end)
  #--------------------------------------------------------------------------}}}
  def _print_verbose2(self, *args, sep="\n  ", end='\n'):  # {{{
    self._print_verbose(*args, sep=sep, end=end, level=2)
  #--------------------------------------------------------------------------}}}
  def _log_data(self, data_update):  # {{{
    """
    Log the candle data if logging is enabled.
    """
    if not self.logging:
      return

    if len(data_update) == 0:
      data_update = pd.DataFrame(["-- NO UPDATE --"])

    log_entry = data_update.rename(columns=lambda x:f"{x}".title()).to_string(header=not os.path.isfile(self.log_data_filename), index=False)

    with open(self.log_data_filename, 'a') as f:
      f.write(log_entry + '\n')
  #--------------------------------------------------------------------------}}}
  def _log_order(self, order):  # {{{
    """
    Log an order if logging is enabled.
    """
    if not self.logging:
      return

    with open(self.log_orders_filename, 'a') as f:
      f.write(f"{datetime_iso()}: {order}\n")
  #--------------------------------------------------------------------------}}}
  def _log_positions(self, positions):  # {{{
    """
    Log the positions if logging is enabled. Don't log the positions if nothing has
    changed.
    """
    if not self.logging:
      return

    str_positions = '\n'.join([self.position_str(p, line_start=f"{i}. ") for i,p in enumerate(positions, start=1)])
    bal_asset = self._format_asset(self.balance_asset)
    bal_curr  = self._format_curr(self.balance_curr)
    str_balances = f"Balances | asset: {bal_asset}, curr: {bal_curr}"
    log_entry = str_positions + '\n' + str_balances

    # Just return if what we would write a duplicate log entry.
    if log_entry == getattr(self, "_log_positions_last", None):
      return
    self._log_positions_last = log_entry

    with open(self.log_positions_filename, 'a') as f:
      f.write(f"-- {datetime_iso()} --\n")
      f.write(log_entry + '\n')
  #--------------------------------------------------------------------------}}}

  def state_export(self, save=True):  # {{{
    """
    Return a dictionary describing the state of the trader. If $save=True, then save
    the state to
      ./$symbol_pair/trading-bot_state.pickle,
    if $save is a string, then use that as the filename, and if $save=False, then
    don't save. Can be used with state_import() to resume a trader after
    reinitialization.
    """
    if save == True:
      filename = f"./{self.symbol_pair}/trading-bot_state.pickle"
    elif save != False:
      filename = save

    params_state = [
      "symbol_asset",
      "symbol_curr",
      "symbol_pair",
      "balance_asset",
      "balance_curr",
      "history",
      "positions",
      "_cancel_retry",
      "_halt_resume",
    ]
    state = {param: getattr(self, param) for param in params_state if hasattr(self, param)}

    if save != False:
      mu.save(state, filename)
    return state
  #--------------------------------------------------------------------------}}}
  def state_import(self, *, state=None, load=True):  # {{{
    """
    Given a dictionary describing the state of a trader, modify self so that it
    matches. If $load=True, then load the state from
      ./$symbol_pair/trading-bot_state.pickle,
    if $load is a string, then use that as the filename, and if $load=False, then
    use $state instead. $state and $load are mutually exclusive options. Sanity check
    a little.
    """
    if state is not None and load == True:
      load = False
    elif state is not None and load != False:
      raise ValueError(f"both state and load specified in state_import(): {state=} {load=}")
    elif state is None and load == True:
      filename = f"./{self.symbol_pair}/trading-bot_state.pickle"
    elif state is None and load != False:
      filename = load

    if load != False:
      state = mu.load(filename)

    if state["symbol_pair"] != self.symbol_pair:
      raise ValueError(f"cannot import state. Symbol pair differs: {state['symbol_pair']=}, {self.symbol_pair=}")

    for param, value in state.items():
      setattr(self, param, value)

    return state
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1
