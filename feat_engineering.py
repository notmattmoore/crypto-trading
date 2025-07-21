# imports {{{1
# general
from copy      import deepcopy
from functools import reduce
from itertools import islice, product
import datetime as dt
import joblib
import numbers

# ML
import numpy  as np
import pandas as pd

# custom modules
from mylibs.utils   import datetime_iso, func_name, map_dict, filter_recurse, is_iter
from training_tools import df_astype_numpy
#----------------------------------------------------------------------------}}}1

memory = joblib.Memory(location="./cache/", bytes_limit=16*1024**3, verbose=0)
memory.reduce_size()

def identity(D):
  return D
def identity_delta(D, t=1):
  return D.diff(t)
def log_signed(D):
  return np.sign(D) * np.log(np.abs(D[D!=0]))
def log_signed_delta(D, t=1):
  return log_signed(D).diff(t)
TRANSF_IDENTITY   = {"price": identity,   "volume": identity,   "price_delta": identity_delta,   "volume_delta": identity_delta}
TRANSF_LOG_SIGNED = {"price": log_signed, "volume": log_signed, "price_delta": log_signed_delta, "volume_delta": log_signed_delta}

NOT_FEATS = (
  "include_cols",
  "task",
  "stride_ticks",
  "stride_smooth",
  "history_tensor",
  "transf",
  "target",
  "target_lookahead_ticks",
  "target_low",
  "target_high",
  "target_hist",
  "target_transf",
  "features_final",
)
FEAT_OPTION_SUFFIXES = (
  "_transf",
  "_on",
  "_bins",
  "_hist",
  "_kwargs",
)

def stride_data(D, stride, smooth): # {{{
  """
  Stride/aggregate/resample the data. Based on the starting index in
  [0, stride), there are different ways to do this. We do *all* of them. Returns a
  dictionary of strides.
  """
  # find the data suffixes
  suffixes = set()
  for col, label in product(D.columns, ["date", "open", "high", "low", "close", "volume"]):
    if col.startswith(label):
      suffixes.add(col.split(label, 1)[1])

  # stride each data suffix
  to_concat = {}
  for suff in suffixes:
    col_date   = f"date{suff}"
    col_open   = f"open{suff}"
    col_high   = f"high{suff}"
    col_low    = f"low{suff}"
    col_close  = f"close{suff}"
    col_volume = f"volume{suff}"

    if col_date in D.columns:
      to_concat[col_date] = D[col_date]
    if col_volume in D.columns:
      D_vol = D[col_volume].rolling(stride).sum()
      to_concat[col_volume] = D_vol

    # If smoothing, then use a weighted average (based on volume) to calculate the
    # open, high, low, and close prices for each stride. Otherwise, just calculate
    # them normally.
    if smooth:
      (weights, weights_sum) = (D[col_volume], D_vol) if col_volume in D.columns else (1, stride)
      for label in [col_open, col_high, col_low, col_close]:
        if label in D.columns:
          D_wt_mean = (D[label] * weights).rolling(stride).sum() / weights_sum
          D_wt_mean = D_wt_mean.rename(label)
          to_concat[label] = D_wt_mean
    else:
      if col_open in D.columns:
        to_concat[col_open] = D[col_open].shift(stride-1)
      if col_high in D.columns:
        to_concat[col_high] = D[col_high].rolling(stride).max()
      if col_low in D.columns:
        to_concat[col_low] = D[col_low].rolling(stride).min()
      if col_close in D.columns:
        to_concat[col_close] = D[col_close]

  # Order the columns of D_stride like D, then segment them into a dictionary
  # indexed modulo the stride.
  D_stride = pd.concat([to_concat[k] for k in D.columns if k in to_concat], axis=1)
  D_stride = {m: D_stride.iloc[m::stride].dropna() for m in range(stride)}

  return D_stride
#----------------------------------------------------------------------------}}}
def stride_data_max_stride(D, stride, smooth): # {{{
  """
  See stride_data(). Returns the stride with maximal maximal index.
  """
  D_stride = stride_data(D, stride, smooth)
  idx_stride = max((max(d.index, default=0), k) for (k, d) in D_stride.items())[1]
  D_stride = D_stride[idx_stride]
  return D_stride
#----------------------------------------------------------------------------}}}
def classify(D, bins, outlier_thresh=0.005): # {{{
  """
  Classify each columns of the dataframe into classes. If bins is an integer,
  then compute the bins uniformly based on the min/max of each columns,
  excluding outliers. If bins is iterable, then use those bins as-is.
  """
  D = pd.DataFrame(D)
  R = {}
  for col in D.columns:
    if is_iter(bins):
      bins_local = bins
    else:
      low, high = D[col].quantile([outlier_thresh, 1-outlier_thresh])
      bins_local = np.linspace(low, high, num=bins+1)
      bins_local[0], bins_local[-1] = -np.inf, np.inf

    # print(f"{col} bins: {bins_local}.")

    R[col] = np.digitize(D[col], bins_local) - 1.0
    R[col][R[col] == len(bins_local)-1] = np.nan  # nan values are given this value --- restore them to nan

  return pd.DataFrame(R, index=D.index)
#----------------------------------------------------------------------------}}}
def history(D, hist_length):  # {{{
  """
  Return a row(D)x((hist_length+1)*col(D)) DataFrame consisting of a copy of D
  followed by copies of D shifted down by 1 tick. If D is indexed by time with
  row i coming before i+k then this corresponds to including past D-entries
  along each row.
  """
  return iter(pd.DataFrame(D).shift(h).rename(columns=lambda s, h=h: f"{s}-{h}") for h in range(1,hist_length+1))
#----------------------------------------------------------------------------}}}
def transf_type_selector(label, delta=False):  # {{{
  if delta:
    delta = "_delta"
  else:
    delta = ''

  if label.startswith("volume"):
    return "volume" + delta
  else:
    for prefix in ["price", "open", "high", "low", "close"]:
      if label.startswith(prefix):
        return "price" + delta
  print(
    f"WARNING {datetime_iso()}, {func_name()}: could not determine transformation "
    f"type for {label=}."
  )
  return ''
#----------------------------------------------------------------------------}}}

def target(D, low, high, horiz, transf, task, name="target"):  # {{{
  """
  Form the target series according to task. For binary and multiclass tasks, we look
  for and classify the maximimum transformed delta price values over $horiz indicies
  in the future which do not occur after a greater minimum decrease (the so-called
  "triple barrier" method, cf Advances in Financial Machine Learning by de Prado). If
  high/low are not provided for classification tasks, then return the max change
  above the lower barriers and the min change when below. For "regression" task just
  give the transformed delta future values.
  """
  P = D["price"]

  if task != "regression":  # binary or multiclass tasks
    # Determine all the profit-making opportunities by calculating the sum of moving
    # max and mins of the transformed data, then using the max values where it is
    # possible to make a profit and the min values where it is not possible.
    P_transf = transf["price_delta"](P, 1).shift(horiz - 1)
    Op = pd.DataFrame({"max": P_transf, "min": P_transf, "trade": 0})
    for t in range(2, horiz+1):
      P_transf = transf["price_delta"](P, t).shift(horiz - t)
      Op["max"] = pd.concat([Op["max"], P_transf], axis=1).max(axis=1)
      Op["min"] = pd.concat([Op["min"], P_transf], axis=1).min(axis=1)
      S = Op["max"] + Op["min"]
      Op["trade"] = pd.concat([
        Op[["trade", "max"]][S > 0].max(axis=1),
        Op[["trade", "min"]][(S < 0) & (Op["trade"] <= 0)].min(axis=1),
        Op["trade"][(S < 0) & (Op["trade"] > 0)],
        Op["trade"][(S == 0) | (S.isna())]
      ])
    R = Op["trade"]

    # Form bins and classify if high/low are provided.
    if task == "binary" and low != None:
      R = classify(R, [-np.inf, low, np.inf])
    elif task == "binary" and high != None:
      R = classify(R, [-np.inf, high, np.inf])
    elif task == "multiclass" and low != None != high:
      R = classify(R, [-np.inf, low, high, np.inf])
  else:  # regression task
    R = transf["price_delta"](P, horiz)

  # squeeze into a series, shift future values, and rename
  R = R.squeeze().shift(-horiz).rename(name)

  return pd.DataFrame(R)
#----------------------------------------------------------------------------}}}

def transf_apply(C, period, *, transf, data, delta=False): # {{{
  """
  Return a DataFrame consisting of changes of C (either difference or percent
  differences), possibly classified.
  """
  if delta:
    delta = "_delta"
  else:
    delta = ''

  R = transf[transf_type_selector(C.name) + delta](C)
  return R.rename(f"transf{delta}({R.name})")
#----------------------------------------------------------------------------}}}
def transf_delta_apply(C, period, *, transf, data): # {{{
  return transf_apply(C, period, transf=transf, data=data, delta=True)
#----------------------------------------------------------------------------}}}
def high_feat(C, period, *, transf, data): # {{{
  """
  Calculate the high prices over the given period.
  """
  cols = [c for c in data.columns if c.startswith("high")]
  R = data[cols].rolling(period).max()
  R = transf["price_delta"](R)
  return R.rename(columns=lambda x: f"{x}{period}")
#----------------------------------------------------------------------------}}}
def low_feat(C, period, *, transf, data): # {{{
  """
  Calculate the low prices over the given period.
  """
  cols = [c for c in data.columns if c.startswith("low")]
  R = data[cols].rolling(period).min()
  T = transf["price_delta"](R)
  return R.rename(columns=lambda x: f"{x}{period}")
#----------------------------------------------------------------------------}}}

def MA(C, period, *, transf, data):  # {{{
  """
  Compute the moving average of C over period, possibly normalizing and
  including standard deviation.
  """
  R = pd.DataFrame(C).rolling(period).mean().rename(columns=lambda x: f"MA{period}({x})")
  R = transf[transf_type_selector(C.name, delta=True)](R)
  return R
#----------------------------------------------------------------------------}}}
def EMA(C, period, *, transf, data):  # {{{
  """
  Compute the exponentially weighted moving average of C over period, possibly
  normalizing and including standard deviation.
  """
  R = pd.DataFrame(C).ewm(span=period, min_periods=period).mean().rename(columns=lambda x: f"EMA{period}({x})")
  R = transf[transf_type_selector(C.name, delta=True)](R)
  return R
#----------------------------------------------------------------------------}}}
def MA_delta(C, period, *, transf, data, breakout=False):  # {{{
  """
  Compute differences of moving averages of C. If breakout=True, then also return a
  column with 1 when MAp1 < MAp2 for the first time, -1 when MAp1 > MAp2 for the
  first time, and 0 otherwise.
  """
  transf_type = transf_type_selector(C.name)
  (p1, p2) = period
  C_MA1 = C.rolling(p1).mean()
  C_MA2 = C.rolling(p2).mean()

  R = transf[transf_type](C_MA2) - transf[transf_type](C_MA1)
  if breakout:
    S = np.sign(R)
    R = np.sign(S - S.shift(1)).rename(f"MABO({p1},{p2})({C.name})")
  else:
    R = R.rename(f"(MA{p1}-MA{p2})({C.name})")

  return R
#----------------------------------------------------------------------------}}}
def MA_delta_breakout(C, period, *, transf, data):  # {{{
  """
  Return a series with 1 when MAp1 < MAp2 for the first time, -1 when MAp1 > MAp2 for
  the first time, and 0 otherwise. This is just a wrapper to run MA_delta with
  breakout=True.
  """
  return MA_delta(C, period, transf=transf, data=data, breakout=True)
#----------------------------------------------------------------------------}}}
def EMA_delta(C, period, *, transf, data, breakout=False):  # {{{
  """
  Compute differences of exponentially weighted moving averages of C. If
  breakout=True, then also return a column with 1 when EMAp1 < EMAp2 for the first
  time, -1 when EMAp1 > EMAp2 for the first time, and 0 otherwise.
  """
  transf_type = transf_type_selector(C.name)
  (p1, p2) = period
  C_EMA1 = C.ewm(span=p1, min_periods=p1).mean()
  C_EMA2 = C.ewm(span=p2, min_periods=p2).mean()

  R = transf[transf_type](C_EMA2) - transf[transf_type](C_EMA1)
  if breakout:
    S = np.sign(R)
    R = np.sign(S - S.shift(1)).rename(f"EMABO({p1},{p2})({C.name})")
  else:
    R = R.rename(f"(EMA{p1}-EMA{p2})({C.name})")

  return R
#----------------------------------------------------------------------------}}}
def EMA_delta_breakout(C, period, *, transf, data):  # {{{
  """
  Return a series with 1 when EMAp1 < EMAp2 for the first time, -1 when EMAp1 > EMAp2
  for the first time, and 0 otherwise. This is just a wrapper to run EMA_delta with
  breakout=True.
  """
  return EMA_delta(C, period, transf=transf, data=data, breakout=True)
#----------------------------------------------------------------------------}}}

def AD(C, period, *, transf, data): # {{{
  """
  Calculate the accumulation/distribution indicator for the given period.
  """
  to_concat = []
  suffixes = iter(c.split("close", 1)[1] for c in data.columns if c.startswith("close"))
  for suff in suffixes:
    C = data["close" + suff]
    H = data["high" + suff]
    L = data["low" + suff]
    V = data["volume" + suff]

    H_p = H.rolling(period).max()
    L_p = L.rolling(period).min()
    MFV = V * C * ((C - L_p) - (H_p - C)) / (H_p - L_p)
    AD = MFV.fillna(0).cumsum()
    AD = transf["price_delta"](AD)
    to_concat.append(AD.rename(f"AD{period}{suff}"))

  return pd.concat(to_concat, axis=1)
#----------------------------------------------------------------------------}}}
def Aroon(C, period, *, transf, data): # {{{
  """
  Calculate the difference between the Aroon Up and Aroon Down lines.
  """
  to_concat = []
  suffixes = iter(c.split("high", 1)[1] for c in data.columns if c.startswith("high"))
  for suff in suffixes:
    H = data["high" + suff]
    L = data["low" + suff]

    A_U = 1 - H.rolling(period+1).apply(lambda x: x.argmax()) / period
    A_D = 1 - L.rolling(period+1).apply(lambda x: x.argmin()) / period
    R = A_U - A_D
    to_concat.append(R.rename(f"Aroon{period}{suff}"))

  return pd.concat(to_concat, axis=1)
#----------------------------------------------------------------------------}}}
def MACD(C, period, *, transf, data): # {{{
  """
  Compute the MACD indicators for C, possibly normalizing. For period=(a,b,c),
  - the MACD is EMA(a) - EMA(b) (we assume a<b),
  - compute EMA(c) of the MACD,
  - a 'buy' signal is when the MACD > EMA_MACD.
  """
  transf_type = transf_type_selector(C.name)
  (a,b,c) = period

  C_EMAa = C.ewm(span=a, min_periods=a).mean()
  C_EMAb = C.ewm(span=b, min_periods=b).mean()
  C_MACD = C_EMAa - C_EMAb
  C_MACD_EMA = C_MACD.ewm(span=c, min_periods=c).mean()

  MACD = transf[transf_type](C_MACD) - transf[transf_type](C_MACD_EMA)
  return MACD.rename(f"MACD({a},{b},{c})({C.name})")
#----------------------------------------------------------------------------}}}
def OBV(C, period, *, transf, data):  # {{{
  """
  Calculate the on-balance volume, where C is the price data.
  """
  suffix = ''.join(C.name.partition('_')[1:])
  R = (np.sign(C.diff()) * data[f"volume{suffix}"] * C).cumsum()
  R = transf["volume_delta"](R)
  return R.rename(f"OBV({C.name})")
#----------------------------------------------------------------------------}}}
def RSI(C, period, *, transf, data): # {{{
  """
  Calculate RSIs of C over the period.
  """
  diff = C.diff()
  U = diff.clip(lower=0)
  D = -1*diff.clip(upper=0)

  U_EMA_p = U.ewm(span=period, min_periods=period).mean()
  D_EMA_p = D.ewm(span=period, min_periods=period).mean()
  RS_p = U_EMA_p / D_EMA_p
  C_RSI = 1 - 1/(1 + RS_p)

  return C_RSI.rename(f"RSI{period}({C.name})")
#----------------------------------------------------------------------------}}}
def Sharpe(C, period, *, transf, data): # {{{
  """
  Calculate the volume-weighted sharpe ratio for the given period.
  """
  to_concat = []
  suffixes = iter(c.split("low", 1)[1] for c in data.columns if c.startswith("low"))
  for suff in suffixes:
    price_label = "price" if suff == '' else "close"
    P = data[price_label + suff]
    L = data["low" + suff]
    H = data["high" + suff]
    V = data["volume" + suff]

    V_sum = V.rolling(period).sum()
    mean = ((P + L + H) * V).rolling(period).sum() / (3 * V_sum)
    st_dev = np.sqrt((((P - mean)**2 + (L - mean)**2 + (H - mean)**2) * V).rolling(period).sum() / (3 * V_sum))
    sharpe = (P - P.shift(period - 1)) / st_dev
    to_concat.append(sharpe.rename(f"Sharpe{period}{suff}"))

  return pd.concat(to_concat, axis=1)
#----------------------------------------------------------------------------}}}

@memory.cache
def preproc_data(params_data): # {{{
  """
  Read the data from disk and prepare it for feature engineering:
  - reverse the data (it's backwards on disk),
  - identify missing data and impute it
  - give the date column the right type,
  - clip the data if needed,
  - resample the data.
  """
  # the main data source is stored with key None
  data_filenames = {None: params_data["data_filename"], **params_data.get("data_filenames_addl", {})}

  # load and preprocess the data
  data_addl_preproc = {}
  for (k, f) in data_filenames.items():
    # read data and reverse it (it is backwards on disk)
    data_preproc = pd.read_csv(f)
    data_preproc = data_preproc.loc[::-1].reset_index(drop=True)

    # identify and impute missing data
    data_preproc = data_preproc.set_index("unix", verify_integrity=True).sort_index()
    index_interval = int(data_preproc.index.to_series().diff().min())
    index_full = np.arange(data_preproc.index.min(), data_preproc.index.max()+1, step=index_interval)
    data_preproc = data_preproc.reindex(index_full)
    data_preproc = data_preproc.interpolate(method="linear").interpolate(method="pad")

    # set up the date column and use it as the index
    data_preproc["date"] = pd.to_datetime(data_preproc.index, unit="s")
    data_preproc = data_preproc.reset_index().rename(columns={"index": "unix"})
    data_preproc = data_preproc.set_index("date")

    # clip the data if needed
    data_preproc = data_preproc[params_data.get("data_clip", data_preproc.index[0]):]

    data_addl_preproc[k] = data_preproc

  # Reindex based on main data source so that indicies line up, rename the columns
  # for the additional data sources, concatenate it all together, and finally stride
  # the result.
  data_addl_preproc = map_dict(
    lambda D: D.reindex(data_addl_preproc[None].index), data_addl_preproc
  )
  for k in filter(lambda x:x != None, data_addl_preproc.keys()):
    data_addl_preproc[k].rename(columns=lambda x, k=k: f"{x}_{k}", inplace=True)
  data_addl_preproc = pd.concat(data_addl_preproc.values(), axis=1).sort_index()
  data_addl_preproc = stride_data_max_stride(data_addl_preproc, params_data.get("data_resample", 1), False)

  return data_addl_preproc
#----------------------------------------------------------------------------}}}
def target_feats(D, params_data, params_fe):  # {{{
  """
  Calculate the target feature.

  ** We assume that data is indexed by time with row i coming before i+k. **
  """
  # Make a working copy of D and form the price column.
  D_work = D.copy()
  D_work["price"] = D_work[params_data["price"]]

  # Get the target parameters for given the task.
  target_task = params_fe["task"]
  if target_task == "binary":
    target_low = None
    target_high = params_fe.get("target_high")
  elif target_task == "multiclass":
    target_low = params_fe.get("target_low")
    target_high = params_fe.get("target_high")
  elif target_task == "regression":
    target_low = target_high = None

  # Get the transformation to apply to the price data.
  transf_get = params_fe.get("transf")
  if transf_get == None:
    transf = TRANSF_IDENTITY
  elif isinstance(transf_get, str):
    transf = globals()[f"TRANSF_{transf_get}".upper()]
  else:
    transf = transf_get
  target_transf_get = params_fe.get("target_transf", True)
  if target_transf_get == True:
    target_transf = transf
  elif target_transf_get == False:
    target_transf = TRANSF_IDENTITY
  elif isinstance(target_transf_get, dict):
    target_transf = target_transf_get

  # Calculate the target feature.
  lookahead_rows = params_fe["target_lookahead_ticks"] // params_data.get("data_resample", 1)
  D_target = target(
    D_work, target_low, target_high, lookahead_rows, target_transf, target_task,
    name=params_data["labels_y"][0],
  )

  # Only include the target feature if specified (don't include for production)
  if params_fe.get("target", True):
    to_concat = [D_target]
  else:
    to_concat = []

  # Include historical target values.
  if "target_hist" in params_fe.keys():
    D_target_hists = history(D_target, lookahead_rows + params_fe["target_hist"] - 1)
    D_target_hists = pd.concat(islice(D_target_hists, lookahead_rows - 1, None), axis=1)
    D_target_hists = D_target_hists.rename(columns={c:f"target-{n}" for (n, c) in enumerate(sorted(D_target_hists.columns), start=1)})
    if params_fe.get("features_final") != None:  # Keep only the specified historical target features.
      feats = [f for f in params_fe["features_final"] if f.startswith(tuple(params_data["labels_y"]))]
      D_target_hists = D_target_hists[feats]
    to_concat.append(D_target_hists)

  return pd.concat(to_concat, axis=1)
#----------------------------------------------------------------------------}}}
def add_feature(feat_name, params_feat, D):  # {{{
  """
  Boilerplate to add a feature given the name of the function, options for the
  functions, and the data (D). Returns the feature columns and history as a list.
  """
  func       = globals()[feat_name]
  periods    = params_feat[feat_name]
  transf     = params_feat["transf"]
  stats_on   = params_feat["stats_on"]
  class_bins = params_feat["class_bins"]
  hist       = params_feat["hist"]
  kwargs     = params_feat["kwargs"]

  # Usually this is a list of periods, but sometimes it's just set (usually to
  # True). This logic makes the loop below iterate once over that parameter.
  if periods == False:  # do nothing if set to False
    return []
  if not is_iter(periods):
    periods = [periods]

  cols_stats = [col for col in D.columns if col.startswith(tuple(stats_on))]
  feats = []
  for p, col in product(periods, cols_stats):
    feat_loop = func(D[col], p, transf=transf, data=D, **kwargs)
    if class_bins != None:
      feat_loop = classify(feat_loop, class_bins)
    feats.append(feat_loop)
    feats.extend(history(feat_loop, hist))
  return feats
#----------------------------------------------------------------------------}}}
def features_filter(F, feature_names):  # {{{
  """
  Given a DataFrame, Series, or array of DataFrames and Series, return only those
  with names in feature_names. If feature_names == None then do nothing.
  """
  if feature_names == None:
    return F

  # if F is not a Series or DataFrame then recurse
  if not isinstance(F, (pd.Series, pd.DataFrame)):
    return map(lambda x, feature_names=feature_names: features_filter(x, feature_names), F)

  if isinstance(F, pd.Series) and F.name in feature_names:
    return F

  if isinstance(F, pd.DataFrame):
    return F[[f for f in F.columns if f in feature_names]]

  return pd.DataFrame()
#----------------------------------------------------------------------------}}}
def feature_engineer(D, params_data, params_fe):  # {{{
  """
  Feature engineer the data.

  ** We assume that data is indexed by time with row i coming before i+k. **
  """
  transf_get = params_fe.get("transf")
  if transf_get == None:
    transf = TRANSF_IDENTITY
  elif isinstance(transf_get, str):
    transf = globals()[f"TRANSF_{transf_get}".upper()]
  else:
    transf = transf_get

  stats_on = params_fe.get("stats_on", ["price"])

  features_final = params_fe.get("features_final", None)
  if features_final != None:
    features_final = [f for f in features_final if not f.startswith(tuple(params_data["labels_y"]))]

  # Make a working copy of D and form the price column.
  D_work = D.copy()
  D_work["price"] = D_work[params_data["price"]]

  to_concat = []

  # Copy these cols from D / D_work. This is mostly for debugging.
  for col in params_fe.get("include_cols", []):
    if col in D_work.columns:
      to_concat.append(D_work[col])
    elif col in D.columns:
      to_concat.append(D[col])
    else:
      print(f"WARNING {datetime_iso()}, {func_name()}: could not find column {col} to include.")

  # Do all the feature engineering.
  for feat_name in params_fe.keys():
    if feat_name in NOT_FEATS or feat_name.endswith(FEAT_OPTION_SUFFIXES):
      continue
    if feat_name not in globals():
      print(f"WARNING {datetime_iso()}, {func_name()}: skipping unrecognized feature {feat_name}.")
      continue
    feat_transf_get = params_fe.get(f"{feat_name}_transf", True)
    if feat_transf_get == True:
      feat_transf = transf
    elif feat_transf_get == False:
      feat_transf = TRANSF_IDENTITY
    elif isinstance(feat_transf_get, dict):
      feat_transf = feat_transf_get
    feat_options = {
      feat_name    : params_fe[feat_name],
      "transf"     : feat_transf,
      "stats_on"   : params_fe.get(f"{feat_name}_on", stats_on),
      "class_bins" : params_fe.get(f"{feat_name}_bins", None),
      "hist"       : params_fe.get(f"{feat_name}_hist", 0),
      "kwargs"     : params_fe.get(f"{feat_name}_kwargs", {}),
    }
    # Compute the features, discard those not in features_final, and add the result
    # to list of featues of concatenate.
    to_concat.extend(features_filter(
      add_feature(feat_name, feat_options, D_work), features_final)
    )

  # Concatenate features.
  D_ret = pd.concat(to_concat, axis=1)

  return D_ret
#----------------------------------------------------------------------------}}}
@memory.cache
def proc_data(data_preproc, params_data, params_fe, parallel=True): # {{{
  """
  Process the data:
  - stride the data, smoothing the strides,
  - calculate the target features and perform feature engineering,
  - drop NA values, sort the index and columns, and make sure everything has the
    right type.
  """
  # Stride/aggregate the data
  stride_rows = params_fe["stride_ticks"] // params_data.get("data_resample", 1)
  if stride_rows <= 0:
    raise ValueError(
      f"stride_rows := stride_ticks // data_resample must be strictly positive, "
      f"but stride_ticks={params_fe['stride_ticks']} and "
      f"data_resample={params_data.get('data_resample', 1)}"
    )
  data_proc = stride_data(data_preproc, stride_rows, params_fe.get("stride_smooth", True))

  # Calculate the target features, feature engineer the data, concatenate them
  # together, drop NA values, and convert types. This is done in parallel, which has
  # the potential to use a *lot* of memory.
  data_target = target_feats(pd.concat(data_proc.values()).sort_index(), params_data, params_fe)
  if parallel:
    data_proc = pd.concat(joblib.Parallel(n_jobs=-1)(
      joblib.delayed(feature_engineer)(v, params_data, params_fe) for v in data_proc.values()
    ))
  else:
    data_proc = pd.concat(map(
      lambda v: feature_engineer(v, params_data, params_fe), data_proc.values()
    ))
  data_proc = pd.concat([data_target, data_proc], axis=1)
  data_proc = data_proc.dropna().sort_index().sort_index(axis=1).convert_dtypes()
  data_proc = df_astype_numpy(data_proc)

  return data_proc
#----------------------------------------------------------------------------}}}
def prepare_data(params_data, params_fe, parallel=True): # {{{
  """
  Read options and prepare the data.
  """
  return proc_data(preproc_data(params_data), params_data, params_fe, parallel=parallel)
#----------------------------------------------------------------------------}}}
def prepare_data_backtest(params_data, params_fe, backtest_t1=None, backtest_t2=None, parallel=True): # {{{
  """
  Read options and prepare the data. If $backtest_t1 or $backtest_t2 is set, then
  return data suitable for backtesting in the time interval
  [$backtest_t1, $backtest_t2], where None is interpreted as unbounded.
  """
  # Use data_backtest_resample for resampling.
  params_data = deepcopy(params_data)
  params_data["data_resample"] = params_data.get("data_backtest_resample", 1)

  # Preprocess the data.
  data_preproc = preproc_data(params_data)

  # If backtest_t1 or backtest_t2 are unset, then use the extreme of the index.
  if backtest_t1 is None:
    backtest_t1 = data_preproc.index.min()
  if backtest_t2 is None:
    backtest_t2 = data_preproc.index.max()

  # Trim the data to the backtesting interval. Many features will be NA at the start
  # of the interval, and the target will be NA at the end of the interval, so we
  # include some ticks before and after backtest_t1 and backtest_t2.
  feat_periods = [v for v in params_fe.values() if is_iter(v)]
  feat_period_max = 2 * max(filter_recurse(lambda x: isinstance(x, numbers.Number), feat_periods))
  time_granularity = abs(data_preproc.index[1] - data_preproc.index[0])
  stride_rows = params_fe["stride_ticks"] // params_data.get("data_resample", 1)
  backtest_t1_pad = backtest_t1 - time_granularity * stride_rows * feat_period_max
  backtest_t2_pad = backtest_t2 + time_granularity * params_fe["target_lookahead_ticks"]
  data_preproc = data_preproc.loc[backtest_t1_pad : backtest_t2_pad]

  # Process the data and clip it to the backtest range
  data_proc = proc_data(data_preproc, params_data, params_fe, parallel=parallel)
  data_proc = data_proc.loc[backtest_t1 : backtest_t2]

  # Include candle information from the preprocessed data computed above.
  data_back = data_preproc[["open", "high", "low", "close", "volume"]].loc[data_proc.index]
  data_back["date"] = data_back.index
  data_proc = pd.concat([data_proc, data_back], axis=1)

  return data_proc
#----------------------------------------------------------------------------}}}
def prepare_data_trader(D, params_data, params_fe): # {{{
  """
  Process DataFrame D in preparation for making a trading prediction. D should
  consist of candles completely indexed (i.e. no missing indices) by epoch time or
  date.
  """
  D_proc = D.copy().sort_index()

  # Index by date.
  D_proc["date"] = pd.to_datetime(D_proc.index, unit="s")
  D_proc = D_proc.set_index("date", verify_integrity=True)

  # Resample and stride the data.
  D_proc = stride_data_max_stride(D_proc, params_data.get("data_resample", 1), False)
  stride_rows = params_fe["stride_ticks"] // params_data.get("data_resample", 1)
  D_proc = stride_data(D_proc, stride_rows, params_fe.get("stride_smooth", True))

  # Target and feature engineering. We only need to feature engineer the most recent
  # stride, so use D_proc[0] there.
  params_fe = {**params_fe, "target": False}  # no need to calculate target variable
  D_target = target_feats(pd.concat(D_proc.values()).sort_index(), params_data, params_fe)
  D_feats = feature_engineer(D_proc[0], params_data, params_fe)
  D_proc = pd.concat([D_target, D_feats], axis=1)

  # Clean up engineered data.
  D_proc = D_proc.dropna().sort_index().sort_index(axis=1).convert_dtypes()
  D_proc = df_astype_numpy(D_proc)

  return D_proc
#----------------------------------------------------------------------------}}}


# Volume Ticks {{{1
#def read_data(params_data):  # {{{
#  """
#  Read the data, index by date, and clip it if needed.
#  """
#  data_read = pd.read_csv(params_data["data_filename"]).dropna()
#  data_read["date"] = pd.to_datetime(data_read["unix"], unit="s")
#  data_read = data_read.set_index("date", verify_integrity=True).sort_index()
#  data_read = data_read[params_data.get("data_clip", data_read.index[0]):]
#  return data_read
##----------------------------------------------------------------------------}}}
#def data_vol_tick_resolution(params_data, D=None):  # {{{
#  """
#  Return params_data["data_vol_tick"] if set, otherwise calculate
#  params_data["data_vol_ticks_percentile"] percentile volume tick size (defaults to
#  0.9 if unset). In this case, D must be specified.
#  """
#  vol_tick = params_data.get("data_vol_tick")
#  if vol_tick is None:
#    if D is None:
#      D = read_data(params_data)
#    vol_tick = D["volume"].quantile(params_data.get("data_vol_ticks_percentile", 0.9))
#  elif params_data.get("data_vol_ticks_percentile") is not None:
#    print(
#      "WARNING: in data_vol_tick_resolution() both data_vol_tick and "
#      "data_vol_ticks_percentile are set:\n"
#      f"  data_vol_tick={params_data['data_vol_tick']},\n"
#      f"  data_vol_ticks_percentile={params_data['data_vol_ticks_percentile']}.\n"
#      "Using data_vol_tick."
#    )
#  return vol_tick
##----------------------------------------------------------------------------}}}
#def preproc_data_vol_ticks(params_data): # {{{
#  """
#  Read the data from disk and prepare it for feature engineering:
#  - index by volume ticks, making the index uniform and imputing data,
#  - reindex by date.
#  """
#  data_preproc = read_data(params_data)

#  # Index by volume ticks. Volume ticks are extremely non-uniform, so we re-index
#  # using a linearly spaced index capturing the $data_vol_ticks_percentile percentile
#  # of volume tick deltas.
#  vol_tick = data_vol_tick_resolution(params_data, D=data_preproc)
#  data_preproc["volume_ticks"] = data_preproc["volume"].cumsum()
#  data_preproc = data_preproc.set_index("volume_ticks", verify_integrity=True)
#  ticks_num = int((data_preproc.index[-1] - data_preproc.index[0]) / vol_tick)
#  ticks_index = np.linspace(data_preproc.index[0], data_preproc.index[-1], num=ticks_num)
#  ticks_index_ext = set(list(ticks_index) + data_preproc.index.to_list())
#  data_preproc = data_preproc.reindex(ticks_index_ext).sort_index().interpolate(method="linear").interpolate(method="pad").loc[ticks_index]

#  # Index by date.
#  data_preproc["date"] = pd.to_datetime(data_preproc["unix"], unit="s")
#  data_preproc = data_preproc.set_index("date", verify_integrity=True)

#  # drop extra columns and resample
#  data_preproc = data_preproc.drop(columns=["symbol", "unix"])
#  data_preproc = stride_data_max_stride(data_preproc, params_data.get("data_resample", 1), False)

#  return data_preproc
##----------------------------------------------------------------------------}}}
#def prepare_data_vol_ticks(params_data, params_fe, parallel=True): # {{{
#  """
#  Read options and prepare the data using volume ticks.
#  """
#  return proc_data(preproc_data_vol_ticks(params_data), params_data, params_fe, parallel=parallel)
##----------------------------------------------------------------------------}}}
#def data_ts_resolution(params_data):  # {{{
#  """
#  Return the time series resolution of the data specified by params_data (i.e. tick
#  duration).
#  """
#  data = read_data(params_data)
#  return data.index.to_series().diff().median()
##----------------------------------------------------------------------------}}}
#def prepare_data_backtest_vol_ticks(params_data, params_fe, backtest_t1=None, backtest_t2=None, parallel=True): # {{{
#  """
#  Read options and prepare the data using volume ticks. If $backtest_t1 or
#  $backtest_t2 is set, then return data suitable for backtesting in the time interval
#  [$backtest_t1, $backtest_t2], where None is interpreted as unbounded.
#  """
#  # Use data_backtest_resample for resampling.
#  params_data = deepcopy(params_data)
#  params_data["data_resample"] = params_data.get("data_backtest_resample", 1)

#  # Preprocess the data.
#  data_preproc = preproc_data_vol_ticks(params_data)

#  # If backtest_t1 or backtest_t2 are unset, then use the extreme of the index.
#  if backtest_t1 is None:
#    backtest_t1 = data_preproc.index.min()
#  if backtest_t2 is None:
#    backtest_t2 = data_preproc.index.max()

#  # Process the data, resample to original time series resolution (i.e. discard
#  # interpolated data), and clip it to the backtest range
#  data_proc = proc_data(data_preproc, params_data, params_fe, parallel=parallel)
#  data_proc = data_proc.loc[backtest_t1 : backtest_t2]

#  # Include candle information from the preprocessed data computed above.
#  data_back = data_preproc[["open", "high", "low", "close", "volume"]].loc[data_proc.index]
#  data_proc = pd.concat([data_proc, data_back], axis=1)
#  data_proc = data_proc.resample(data_ts_resolution(params_data)).first().dropna()
#  data_proc["date"] = data_proc.index

#  return data_proc
##----------------------------------------------------------------------------}}}
#def prepare_data_trader_vol_ticks(D, params_data, params_fe): # {{{
#  """
#  Process DataFrame D in preparation for making a trading prediction. D should
#  consist of candles completely indexed (i.e. no missing indices) by epoch time or
#  date.
#  """
#  D_proc = D.copy().sort_index()

#  # Convert existing index to epoch time and save it, then reindex by volume ticks.
#  D_proc["unix"] = pd.to_datetime(D_proc.index, unit="s")
#  D_proc["unix"] = (D_proc["unix"] - dt.datetime(1970,1,1)).dt.total_seconds()
#  D_proc["volume_ticks"] = D_proc["volume"].cumsum()
#  D_proc = D_proc.set_index("volume_ticks", verify_integrity=True)
#  ticks_num = int((D_proc.index[-1] - D_proc.index[0]) / params_data.get("data_vol_tick"))
#  ticks_index = np.linspace(D_proc.index[0], D_proc.index[-1], num=ticks_num)
#  ticks_index_ext = set(list(ticks_index) + D_proc.index.to_list())
#  D_proc = D_proc.reindex(ticks_index_ext).sort_index().interpolate(method="linear").interpolate(method="pad").loc[ticks_index]

#  # Index by date.
#  D_proc["date"] = pd.to_datetime(D_proc["unix"], unit="s")
#  D_proc = D_proc.set_index("date", verify_integrity=True).drop(columns="unix")

#  # Resample and stride the data. We only care about the most recent resample (i.e.
#  # the max one), but we need all the properly strided ones to calculate the target
#  # features.
#  D_proc = stride_data_max_stride(D_proc, params_data.get("data_resample", 1), False)
#  stride_rows = params_fe["stride_ticks"] // params_data.get("data_resample", 1)
#  D_proc = stride_data(D_proc, stride_rows, params_fe.get("stride_smooth", True))

#  # Target and feature engineering. We only need to feature engineer the most recent
#  # stride, which is at idx_max_stride.
#  params_fe = {**params_fe, "target": False}  # no need to calculate target variable
#  D_target = target_feats(pd.concat(D_proc.values()).sort_index(), params_data, params_fe)
#  idx_max_stride = max((max(d.index), k) for (k, d) in D_proc.items())[1]
#  D_feats = feature_engineer(D_proc[idx_max_stride], params_data, params_fe)
#  D_proc = pd.concat([D_target, D_feats], axis=1)

#  # Clean up engineered data.
#  D_proc = D_proc.dropna().sort_index().sort_index(axis=1).convert_dtypes()
#  D_proc = df_astype_numpy(D_proc)

#  return D_proc
##----------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1
