# Cryptocurrency Trading Bot

This project implements an end-to-end automated trading system for
cryptocurrency markets using the CoinBase API. It features real-time trading,
dynamic model updates, technical indicator–based feature engineering, and
strategy simulation.

*Models are not included.*

All files are heavily commented and contain more specific documentation.

Requires [coinbase](https://github.com/notmattmoore/coinbase) and 
[python-mylibs](https://github.com/notmattmoore/python-mylibs).


### Modules
#### `trade.py`
Framework for algorithmic trading. The two main classes are `TradeDecider`,
which wraps the model and determines whether to buy, hold, or sell, and
`CBTrader`, which takes a `TradeDecider` instance and actually implements
trading policy and tracks trades.


#### `trading_bot.py`
An implementation of a trading bot using `trade.py`. Requires prediction models
in order to function.


#### `feat_engineering.py`
A module for feature engineering candle (OHLCV) data. Features:
- performs data cleanup and interpolation of missing data
- supports and implements basic de-autocorrelating of data
- computes many historical and real-time technical indicators
- supports data resampling, striding, smoothing, and history stacking


#### `training_tools.py`
This mostly consists of useful wrappers around `sklearn` metrics and
`matplotlib`. It also contains three classes for splitting time-series data into
training/validation pairs aimed at reducing biases.
