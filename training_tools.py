# imports {{{1
from copy      import deepcopy
from itertools import chain, combinations, product, repeat
from time      import time
import mylibs.utils as mu

# general (optional)  # {{{
from collections.abc import Iterable
import datetime as dt
#----------------------------------------------------------------------------}}}

# ML data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ML processing
# sklearn {{{
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
#----------------------------------------------------------------------------}}}

# ML metrics
from sklearn.metrics import auc
# sklearn classification {{{
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
#----------------------------------------------------------------------------}}}
# sklearn regression {{{
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
#----------------------------------------------------------------------------}}}

# custom modules
from mylibs.utils import datetime_iso, print_dyn_line, str_dict_round
#----------------------------------------------------------------------------}}}1


# General utilities
is_datetime = lambda v: isinstance(v, (int, np.datetime64))
is_float    = lambda v: isinstance(v, (float, np.floating))
is_int      = lambda v: isinstance(v, (int, np.integer))
def time_to_row(*, t=None, td=None, X=pd.DataFrame()): # {{{
  # Convert a time or timedelta to a row index. Used for timeseries splits.
  if td is not None:
    t = X.index[0] + td
  r = X.index.get_indexer([t], method="nearest")[0]
  if isinstance(r, slice):
    r = r.start
  return r
#----------------------------------------------------------------------------}}}


# Plotting
def plot_multi(  # {{{
  data, *,
  titles=None, dim_x=3, dim_y=2, share_x=False, share_y=False, title_prefix=None,
  save=False, show=True
):
  """
  Plot multiple datasets, possibly sharing x or y axes.
  Arguments:
  - data: an iterable of either arrays of dictionaries of arrays to plot. If
          data is a dictionary, then use the keys for the plot titles.
  - titles: an iterable of the same length as data of the plot titles.
  - dim_x, dim_y: number of columns and rows of subplots, respectively.
  - share_x, share_y: whether x and y axes should be shared.
  - title_prefix: prefix for plot titles.
  - save: whether to save the plot.
  """
  if titles == None:
    titles = range(len(data))
  if title_prefix == None:
    plt_title_prefix = filename_prefix = ''
  if title_prefix != None:
    plt_title_prefix = title_prefix + ': '
    filename_prefix = title_prefix + '_'

  if isinstance(data, dict):
    data_titles = [(data[k], str(k)) for k in data]
  else:
    data_titles = list(zip(data, titles))

  # plt.subplots returns a 1D array of dim_x == 1 or dim_y == 1
  axes_indexer = lambda i, A: A if dim_x == dim_y == 1 else A[i % max(dim_x, dim_y)] if dim_x == 1 or dim_y == 1 else A[i//dim_x][i%dim_x]

  n_subplots = dim_x * dim_y
  for i in range(int(np.ceil(len(data) / n_subplots))):
    _, axes = plt.subplots(dim_y, dim_x, sharex=share_x, sharey=share_y)
    for j, (d, t) in enumerate(data_titles[i*n_subplots:min((i+1)*n_subplots, len(data))]):
      axes_loop = axes_indexer(j, axes)
      axes_loop.set_title(f"{plt_title_prefix}{t}")
      if isinstance(d, dict):
        for k in d:
          axes_loop.plot(d[k], label=k)
      else:
        axes_loop.plot(d)
    axes_indexer(0, axes).legend()
    if save:
      plt.savefig(f"{filename_prefix}plot-multi_{i}.pdf")
    if show:
      plt.show()
      plt.close()
#----------------------------------------------------------------------------}}}
def plot_precision_recall_theshold(y_true, y_scores, label="", save=None, show=True): # {{{
  if label != "":
    label += " "

  precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
  plt.plot(thresholds, precisions[:-1], label=f"{label}Precision")
  plt.plot(thresholds, recalls[:-1], label=f"{label}Recall")
  plt.xlabel("Threshold")
  plt.legend()
  plt.grid(True)
  plt.axis([min(thresholds), max(thresholds), 0, 1])

  if save != None:
    plt.savefig(save)
  if show:
    plt.show()
    plt.close()
#----------------------------------------------------------------------------}}}
def plot_precision_recall(y_true, y_scores, label=None, save=None, show=True): # {{{
  precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
  plt.plot(recalls, precisions, label=label)
  plt.plot([0,1], [0,1], '.')
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  if label:
    plt.legend()
  plt.grid(True)
  plt.axis([0, 1, 0, 1])

  if save != None:
    plt.savefig(save)
  if show:
    plt.show()
    plt.close()
#----------------------------------------------------------------------------}}}
def plot_roc_curve(y_true, y_scores, label=None, save=None, show=True): # {{{
  fpr, tpr, _ = roc_curve(y_true, y_scores)
  plt.plot(fpr, tpr, label=label)
  plt.plot([0,1], [0,1], '.')
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  if label:
    plt.legend()
  plt.grid(True)
  plt.axis([0, 1, 0, 1])

  if save != None:
    plt.savefig(save)
  if show:
    plt.show()
    plt.close()
#----------------------------------------------------------------------------}}}


# Stats and scoring
def precision_at_recall(y_true, y_pred_prob, recall_min): # {{{
  """
  Return the triple (precision, recall, threshold) such that precisions is
  maximal for recall >= recall_min.
  """
  precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
  thresholds = [0] + list(thresholds)  # for some reason the min threshold is omitted
  pt = np.array(list(zip(precisions, recalls, thresholds)))

  return max(pt[recalls >= recall_min], key=lambda X:X[0])
#----------------------------------------------------------------------------}}}
def precision_recall_auc(y_true, y_pred_prob):  # {{{
  precisions, recalls, _ = precision_recall_curve(y_true, y_pred_prob)
  return auc(recalls, precisions)
#----------------------------------------------------------------------------}}}
def confusion_matrix_prob(y_true, y_score): # {{{
  """
  Compute the probabilistic confusion matrix.
  """
  y_true = np.array(y_true)
  y_score = np.array(y_score)

  y_true_by_class = [ (y_true == k).astype(int) for k in np.sort(np.unique(y_true)) ]
  if len(y_score.shape) == 1:   # binary
    y_score_by_class = [ 1-y_score, y_score ]
  else:   # multiclass
    y_score_by_class = [ y_score[:,k] for k in range(y_score.shape[-1]) ]

  C = np.array([ [np.dot(row, col) for col in y_score_by_class] for row in y_true_by_class ])
  return C
#----------------------------------------------------------------------------}}}
def precision_recall_fscore_prob(y_true, y_score, average="binary"): # {{{
  """
  Compute the probabilistic precision, recall, and F1 scores.
  """
  prec        = lambda tp,fp,fn: tp / (tp + fp)
  rec         = lambda tp,fp,fn: tp / (tp + fn)
  f1          = lambda tp,fp,fn: 2 * prec(tp,fp,fn) * rec(tp,fp,fn) / (prec(tp,fp,fn) + rec(tp,fp,fn))
  prec_rec_f1 = lambda tp,fp,fn: (prec(tp,fp,fn), rec(tp,fp,fn), f1(tp,fp,fn))

  C   = confusion_matrix_prob(y_true, y_score)
  TPs = [ C[i][i] for i in range(len(C)) ]
  FPs = [ sum(C[r][c] for r in range(len(C)) if r != c) for c in range(len(C)) ]
  FNs = [ sum(C[r][c] for c in range(len(C)) if c != r) for r in range(len(C)) ]

  if average == "binary":
    if C.shape != (2,2):
      print(f"Error: confusion matrix has wrong shape {C.shape} for binary classification.")
      return (0, 0, 0)
    return prec_rec_f1(TPs[1], FPs[1], FNs[1])
  elif average == "micro":
    return prec_rec_f1(sum(TPs), sum(FPs), sum(FNs))
  elif average == "macro":
    M = list(map(prec_rec_f1, TPs, FPs, FNs))
    return tuple(np.mean(M, axis=0))
  elif average == "weighted":
    M = list(map(prec_rec_f1, TPs, FPs, FNs))
    support = [ (np.array(y_true) == k).sum() for k in range(len(C)) ]
    return tuple(sum(support[k] * M[k][i] for k in range(len(M))) / len(y_true) for i in range(len(M[0])))
  return (0, 0, 0)
#----------------------------------------------------------------------------}}}
def precision_prob(y_true, y_score, average="binary"):  # {{{
  return precision_recall_fscore_prob(y_true, y_score, average=average)[0]
#----------------------------------------------------------------------------}}}
def recall_prob(y_true, y_score, average="binary"):  # {{{
  return precision_recall_fscore_prob(y_true, y_score, average=average)[1]
#----------------------------------------------------------------------------}}}
def f1_prob(y_true, y_score, average="binary"):  # {{{
  return precision_recall_fscore_prob(y_true, y_score, average=average)[2]
#----------------------------------------------------------------------------}}}
def accuracy_prob(y_true, y_score): # {{{
  """
  Compute the probabilistic accuracy.
  """
  C = confusion_matrix_prob(y_true, y_score)
  correct = sum( C[i][i] for i in range(len(C)) )
  return correct / len(y_true)
#----------------------------------------------------------------------------}}}
def balanced_accuracy_prob(y_true, y_score): # {{{
  """
  Compute the probabilistic balanced accuracy.
  """
  C = confusion_matrix_prob(y_true, y_score)
  correct_by_class  = [ C[i][i] for i in range(len(C)) ]
  support_by_class  = [ (np.array(y_true) == k).sum() for k in range(len(C)) ]
  accuracy_by_class = [ correct_by_class[k] / support_by_class[k] for k in range(len(C)) ]
  return np.mean(accuracy_by_class)
#----------------------------------------------------------------------------}}}


# Printing
def print_cv_stats(results, score_prefix="test_", other_keys=None, sort=True):  # {{{
  if other_keys is None:
    other_keys = {}

  scores=[ k[len(score_prefix):] for k in results.keys() if k[:len(score_prefix)] == score_prefix ]
  I = [results[k] for k in other_keys] + [results[score_prefix+s] for s in scores]
  R = list(zip(*I))

  if other_keys:
    other_widths = [ max(len(str(r[i])) for r in R) for i in range(len(other_keys)) ]
  if sort:  # sort by the first score
    R.sort(key=lambda z: z[len(other_keys)])

  header = "CV Results: ("
  if other_keys:
    header += ", ".join(other_keys) + " | "
  header += ", ".join(scores) + ")"
  print(header)

  for r in R:
    s = ", ".join( f"{r[i+len(other_keys)]:<.8f}" for i in range(len(scores)) )
    if other_keys:
      other_s = ", ".join( f"{r[i]!s:{other_widths[i]}}" for i in range(len(other_keys)) )
      s = f"{other_s} | {s}"
    print("  " + s)

  means = [ sum(r[i+len(other_keys)] for r in R)/len(R) for i in range(len(scores)) ]
  print("Means: " + ", ".join(f"{m:<.8f}" for m in means))
  return R
#----------------------------------------------------------------------------}}}
class StatusPrinter:   # {{{1
  """
  A status printer implementing ETA and max/min detection. Initialization
  parameters:
  - calls_start: what number to start counting calls at. Default is 1.
  - calls_total: total number of calls, optional.
  - show_minmax: whether to indicate min/max values. Default is True.
  - format_calls: string formatting for call display. Defaults just str().
  - format_score: string formatting for score. Defaults to 4 decimal places.
  - format_input: string formatting for inputs. Defaults to
    str_dict_round().
  """
  def __init__(   # {{{
    self, *,
    calls_start=1, calls_total=-1, show_minmax=True, format_calls=None,
    format_score=None, format_input=None
  ):
    self._call_count = calls_start
    self.calls_total = calls_total
    self.show_minmax = show_minmax

    self.format_calls = str                  if format_calls == None else format_calls
    self.format_input = str_dict_round       if format_input == None else format_input
    self.format_score = lambda s: f"{s:.4f}" if format_score == None else format_score

    self._time_last_call = time()
    self.time_elapsed = self.time_elapsed_mean = -1
    self.score_min, self.score_max = np.inf, -np.inf
    self.input_min = self.input_max = None
    self._len_score = self._len_input = -1
  #--------------------------------------------------------------------------}}}
  def __call__(self, inp, score):  # {{{
    self.time_elapsed = time() - self._time_last_call

    # Build the status line, showing the score and parameters.
    line_lead = "  "
    if score > self.score_max:
      self.score_max = score
      self.input_max = inp
      line_lead = "++"
    if score < self.score_min:
      self.score_min = score
      self.input_min = inp
      line_lead = "--"
    if self.score_min == self.score_max:
      self.score_min = self.score_max = score
      self.input_min = self.input_max = inp
      line_lead = "+-"
    if not self.show_minmax:
      line_lead = "  "
    str_score = self.format_score(score)
    str_inp = self.format_input(inp)
    self._len_input = max(self._len_input, len(str_inp))
    self._len_score = max(self._len_score, len(str_score))
    print_dyn_line(f"{line_lead}{str_score:>{self._len_score}} | {str_inp:{self._len_input}} | {dt.timedelta(seconds=int(self.time_elapsed))}", end='\n')

    # Build the eta line, showing which call we are on, eta for that call, and
    # eta for the entire process.
    self._call_count += 1
    if self.calls_total < 0 or self._call_count <= self.calls_total:  # don't print eta on the last iteration
      self.time_elapsed_mean = (self.time_elapsed_mean * (self._call_count - 2) + self.time_elapsed) / (self._call_count - 1)
      eta_call = datetime_iso(time() + self.time_elapsed)
      eta = datetime_iso(time() + self.time_elapsed_mean * (self.calls_total - self._call_count + 1))
      str_eta = f" | ETA Next: {eta_call}"
      if self.calls_total > 0:
        str_total_calls = f"/{self.calls_total}"
        str_eta_total = f" | ETA: {eta}"
      else:
        str_total_calls = str_eta_total = ''
      str_calls = self.format_calls(f"{self._call_count}{str_total_calls}")
      print_dyn_line(f"{str_calls}{str_eta}{str_eta_total}")
    elif self._call_count > self.calls_total + 1:   # somehow we've called more than calls_total...
      self.calls_total = -1

    self._time_last_call = time()
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1


# Processing
def df_astype_numpy(D): # {{{
  """
  Convert the dtypes in DataFrame D to all be numpy dtypes instead of pandas
  dtypes. SKLearn is very picky about dtypes, so this is sometimes necessary.
  """
  to_concat = []
  for col in D.columns:
    if D[col].apply(is_float).any():
      to_concat.append(D[col].astype(np.float32))
    elif D[col].apply(is_int).all():
      to_concat.append(D[col].astype(np.int32))
    elif D[col].apply(is_datetime).all():
      to_concat.append(D[col].astype(np.datetime64))
    else:
      to_concat.append(D[col])
  return pd.concat(to_concat, axis=1)
#----------------------------------------------------------------------------}}}
def feats_num_cat(D): # {{{
  """
  Returns (feats_num, feats_cat), where feats_num are the numeric columns of D
  and feats_cat are the categorical ones.
  """

  feats_num = [ c for c in D.columns if D[c].apply(is_float).any() ]
  feats_cat = [ c for c in D.columns if c not in feats_num ]

  return feats_num, feats_cat
#----------------------------------------------------------------------------}}}
def feats_vocab(D): # {{{
  """
  Returns (feats_num, feats_cat), where feats_num are the numeric columns of D
  and feats_cat are the categorical ones.
  """

  feats_num = [ c for c in D.columns if D[c].apply(is_float).any() ]
  feats_cat = [ c for c in D.columns if c not in feats_num ]

  return feats_num, feats_cat
#----------------------------------------------------------------------------}}}
def scale_feats(D, scaler=None, scale_cols=None):  # {{{
  """
  Scale features which appear to be non-categorical. Returns (X, scaler).
  """
  R = D.copy()

  if scale_cols == None:
    scale_cols, _ = feats_num_cat(R)
  if len(scale_cols) == 0:
    return R, scaler

  if scaler == None:
    # scaler = StandardScaler()
    scaler = RobustScaler()
    # scaler = QuantileTransformer(random_state=random_state)
    scaler.fit(R[scale_cols])
  R[scale_cols] = scaler.transform(R[scale_cols])

  return R, scaler
#----------------------------------------------------------------------------}}}
def train_test(D, test_size=0.1):  # {{{
  train_index = int((1-test_size)*len(D))

  train_set = D[:train_index]
  test_set  = D[train_index:]

  return train_set, test_set
#----------------------------------------------------------------------------}}}
def train_valid_test(D, valid_size=0.1, test_size=0.1):  # {{{
  train_index = int((1-valid_size-test_size)*len(D))
  valid_index = int((1-valid_size)*len(D))

  train_set = D[:train_index]
  valid_set = D[train_index:valid_index]
  test_set  = D[valid_index:]

  return train_set, valid_set, test_set
#----------------------------------------------------------------------------}}}
def features_targets(D, y_labels):  # {{{
  X = D[[l for l in D.columns if l not in y_labels]]
  y = D[y_labels]
  return X, y
#----------------------------------------------------------------------------}}}


# Feature Selection
def discard_low_corr_feats(D, corr_thresh, corr_feats, method): # {{{
  """
  Discard the columns of D which are not correlated > corr_thresh with
  features in corr_feat.
  """
  feats = [ f for f in D.columns if f not in corr_feats ]
  feats_high_corr = set(corr_feats)
  C = D.corr(method=method).abs()
  for corr_feat in corr_feats:
    C_loop = C[corr_feat][feats]
    F = C_loop[C_loop > corr_thresh].dropna().index
    if len(F) == 0:   # if we don't get any features, then just take the best one(s)
      print(f"Warning: {corr_thresh=} would discard all columns. Maximum correlation is {C_loop.max()}.")
      F = C_loop[C_loop == C_loop.max()].dropna().index
    feats_high_corr.update(F)

  feats_discard = list(filter(lambda x:x not in feats_high_corr, D.columns))
  print(f"Discarding low correlation (<{corr_thresh}) features: {feats_discard}.")

  feats_high_corr = filter(lambda x:x in feats_high_corr, D.columns)  # maintain the order of columns
  return D[feats_high_corr]
#----------------------------------------------------------------------------}}}
def parameter_grid(params_dict, constraints=None, params_wrap=lambda x:x): # {{{
  """
  params_dict should be a dictionary of the form {key:array}. Returns all
  elements in the cartesian product of the arrays (indexed by respective keys).
  If an element has "skip" as a coordinate, then we omit that coordinate.
  constraints can be given on the return coordinates, as well as a function
  params_wrap to apply before returning each element.
  """
  for p in product(*params_dict.values()):
    p_dict = { k:p[i] for i,k in enumerate(params_dict.keys()) if p[i] != "skip" }
    if constraints == None or constraints(p_dict):
      yield params_wrap(p_dict)
#----------------------------------------------------------------------------}}}
def product_dict_random(  # {{{
  d, num, *, constraints=None, product_wrap=lambda x:x, rng=np.random.default_rng(),
):
  """
  d should be a dictionary of the form {key:array}. Yields a random collection of
  $num many objects in the cartesian product of the arrays (indexed by respective
  keys). Constraints can be given on the return dictionaries, as well as a function
  product_wrap to apply before returning each dictionary.
  """
  # Re-form d so that all entries are lists.
  d = deepcopy(d)
  for (k,v) in d.items():
    # make non-iterables, dicts, and strs into singleton lists
    if not mu.is_iter(v) or isinstance(v, (dict, str)):
      d[k] = [v]

  # Form a random element of the cartesian product by randomly choosing values in
  # each factor.
  count = 0
  while count < num:
    p = {}
    for (k, v) in d.items():
      p[k] = rng.choice(v)
    if constraints == None or constraints(p):
      count += 1
      yield product_wrap(p)
#----------------------------------------------------------------------------}}}


# CV splits
class TimeSeriesMovingSplit:  # {{{
  def __init__(self, test_size=0.1, div_size=1/3, overlap=1/2):
    self.test_size = test_size
    self.div_size  = div_size
    self.overlap   = overlap
  def split(self, X, y=None, groups=None):
    n_splits = (1 - self.div_size) / (self.div_size * self.overlap) + 1
    div_len = self.div_size * len(X)

    I = np.arange(len(X))
    for i in range(int(n_splits)-1):
      start = i*self.div_size*self.overlap*len(X)
      end   = start + div_len
      split = end - self.test_size*(end - start)
      yield I[int(start):int(split)], I[int(split):int(end)]

    start = (1 - self.div_size) * len(X)
    end = len(X)
    split = end - self.test_size*(end - start)
    yield I[int(start):int(split)], I[int(split):int(end)]
  def __str__(self):
    return f"TimeSeriesMovingSplit({self.test_size}, {self.div_size}, {self.overlap})"
#----------------------------------------------------------------------------}}}
class TimeSeriesWindows:  # {{{
  """
  Split a dataset sequentially based on timedeltas. Inputs to split() *must* be
  indexed by time with older times preceeding newer ones.

  Parameters:
  - test_time: time duration of test set.
  - train_time: time duration of training set.
  - n_splits: number of splits.
  - split_sep_time: time between splits.
  - random: True/False/rng. Whether to randomize the splits.
  - repeat_random: How many times to repeat the splits (for randomization). Total
    splits will be repeat_random * n_splits.
  """
  def __init__( # {{{
    self, *, train_time=None, test_time=dt.timedelta(days=100), n_splits=5,
    split_sep_time=dt.timedelta(weeks=1), random=False, repeat_random=1
  ):
    self.train_time     = train_time
    self.test_time      = test_time
    self.n_splits       = n_splits
    self.split_sep_time = split_sep_time

    if random == True:
      self.random = np.random.default_rng()
    else:
      self.random = random
    self.repeat_random = repeat_random
  #--------------------------------------------------------------------------}}}
  def split(self, X, y=None, groups=None, repeat=None):  # {{{
    # handle defaults
    if repeat == None:
      repeat = self.repeat_random
    X_time = X.index[-1] - X.index[0]
    test_total_time = self.test_time + (self.n_splits - 1) * self.split_sep_time
    if self.train_time != None:
      train_time = self.train_time
    else:
      train_time = X_time - test_total_time

    # if there is not enough data then relax train_time
    if train_time + test_total_time > X_time:
      train_time = X_time - test_total_time

    start_time = X.index[-1] - train_time - test_total_time

    # convert times to seconds
    start_sec = start_time.to_pydatetime().timestamp()
    end_sec   = X.index[-1].to_pydatetime().timestamp()
    train_sec = train_time.total_seconds()
    test_sec  = self.test_time.total_seconds()

    # calculate split start times in seconds
    split_starts_sec = np.linspace(start_sec, end_sec-train_sec-test_sec, num=self.n_splits)
    if self.random != False:
      split_starts_sec += self._offsets_rand(split_starts_sec)

    # yield the splits, converting seconds to row indicies
    I = np.arange(len(X))
    for start in split_starts_sec:
      start_idx = time_to_row(t=dt.datetime.fromtimestamp(start), X=X)
      split_idx = time_to_row(t=dt.datetime.fromtimestamp(start + train_sec), X=X)
      end_idx = time_to_row(t=dt.datetime.fromtimestamp(start + train_sec + test_sec), X=X)
      yield I[start_idx : split_idx], I[split_idx : end_idx]

    # recurse if needed
    if self.random != False and repeat > 1:
      yield from self.split(X, y=y, groups=groups, repeat=repeat-1)
  #--------------------------------------------------------------------------}}}
  def _offsets_rand(self, L):  # {{{
    """
    Randomize by choosing a random offset to add to each starting index. We begin by
    choosing a positive offset such that the first split cannot equal the second. For
    the second offset, we choose a positive or negative offset such that the second
    split cannot be the first and cannot be the third, and continue in this manner.
    The last split can only have a negative offset since otherwise the split will
    extend beyond the bounds of X.
    """
    offsets = []
    offset_low = 0
    for k in range(len(L)):
      if k < len(L) - 1:
        offset_high = L[k+1] - L[k]
      else:  # the last offset cannot be positive (see remarks above)
        offset_high = 1
      offsets.append(self.random.integers(offset_low, offset_high))
      offset_low = offsets[-1] - offset_high + 1
    return offsets
  #--------------------------------------------------------------------------}}}

  def __str__(self):  # {{{
    s = type(self).__name__ + '('
    s += ", ".join(f"{a}={str(getattr(self, a))}" for a in vars(self))
    s += ')'
    return s
  #--------------------------------------------------------------------------}}}
  def __copy__(self):  # {{{
    return self.__class__(**vars(self))
  #--------------------------------------------------------------------------}}}
  def __deepcopy__(self, memo):  # {{{
    return self.__copy__()
  #--------------------------------------------------------------------------}}}
  def __eq__(self, other):  # {{{
    attribs_special = ["random"]
    for a in vars(self):
      if a in attribs_special:
        continue
      if getattr(self, a) != getattr(other, a):
        return False
    if self.random == False != other.random or self.random != False == other.random:
      return False
    return True
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}
class TimeSeriesGrowingSplits:  # {{{
  def __init__(self, test_size=0.2, n_splits=5, min_size=0):
    self.test_size = test_size
    self.n_splits  = n_splits
    self.min_size  = min_size
  def split(self, X, y=None, groups=None):
    first_end = int(max(len(X)/self.n_splits, self.min_size*len(X)))
    idx_ends = np.linspace(first_end, len(X), num=self.n_splits, dtype=int)
    I = np.arange(len(X))
    for end in idx_ends:
      split = int((1-self.test_size)*end)
      yield I[:split], I[split:end]
  def __str__(self):
    return f"TimeSeriesGrowingSplits({self.test_size}, {self.n_splits}, {self.min_size})"
#----------------------------------------------------------------------------}}}
