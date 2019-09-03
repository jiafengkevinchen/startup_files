import datetime
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython import get_ipython
from numpy import exp, log, nan
from pandas import DataFrame, Series
from seaborn import regplot
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools import add_constant
from tqdm.auto import tqdm

import janitor

tqdm.pandas()

# pip install git+https://github.com/jiafengkevinchen/pandas_tools
try:
    import pandas_tools.latex as tex
except ImportError:
    print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")


try:
    from janitor.utils import skiperror, skipna
except ImportError:
    try:
        from pandas_tools.latex import skiperror, skipna
    except ImportError:
        print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")


try:
    get_ipython().run_line_magic("matplotlib", "inline")
    get_ipython().run_line_magic("load_ext", "rpy2.ipython")
except:
    pass

sns.set_style("white")
np.set_printoptions(suppress=True)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["figure.dpi"] = 150
