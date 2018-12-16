from datetime import datetime
from time import sleep

import numpy as np
from numpy import log, exp, nan
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
from seaborn import regplot
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from tqdm import tqdm_notebook as tqdm
from IPython import get_ipython


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
except:
    pass

sns.set_style('white')
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi']= 150

