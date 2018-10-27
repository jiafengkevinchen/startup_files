import numpy as np
from statsmodels.iolib.summary2 import summary_col
import re

def skipna(f):
    def _wrapped(x):
        if type(x) is float and np.isnan(x):
            return np.nan
        else:
            return None
    return _wrapped

def skiperror(f):
    def _wrapped(x):
        try: return f(x)
        except: return None
    return _wrapped

def regression_table(regs, notes=None, stars=True, **kwargs):
    """
    Create a pandas.DataFrame object summarizing a series of regressions
    Inputs
    -----
    regs: a list of statsmodels.regression.linear_model.RegressionResults
        objects, one for each column of the regression table

    notes: optional, a dict of additional rows to the table. Each key (string) is
        the name of a row, and each associated value (list of string) is the content

    Returns:
    -----
    t : a pandas.DataFrame of a regression table
    """
    d = {'$N$' : lambda x : '{0:d}'.format(int(x.nobs)),
         '$R^2$' : lambda x : '{:.2f}'.format(x.rsquared)}
    t = summary_col(regs, stars=stars, info_dict=d, **kwargs).tables[0].copy()
    if notes:
        for k, v in notes.items():
            t.loc[k] = v
    return t

def as_latex_regtable(table, table_opt='tb',
                      column_names=None, caption=None,
                      label=None, covariate_names=None, notes='', superscript=True,
                      filename=None, transpose=False, col_format=None):
    """
    Convert a suitably formatted pandas.DataFrame to LaTeX. Requires booktabs,
    threeparttable packages in LaTeX.

    Inputs:
    -----
    table: a pandas.DataFrame
    table_opt: string, optional arguments passed to \begin{table} in LaTeX
    column_names: optional string list, change the name of the columns
    caption: string, optional argument passed to \caption in LaTeX
    label: string, optional argument passed to \label in LaTeX, if caption is
        specified and label is not, then label is caption joined by underscores
    covariate_names: a (string, string) dict where keys are covariate names in table
        and values are their proper string representations
    notes: additional notes to appear under the table
    filename: output .tex file directory; does not output file if unspecified. Will
        _overwrite_ existing file

    Returns:
    -----
    output: string
    """

    table = table.copy()
    if col_format is None:
        if not transpose:
            col_format = 'l{}'.format('c' * (len(table.columns)))
        else:
            col_format = 'l{}'.format('c' * (len(table.index)))
    def formatter(x):
        if superscript:
            x = re.sub('\*+', lambda s: '\\textsuperscript{{{}}}'.format(s[0]), x)
        if '$' not in x:
            x = re.sub(r'[-+]?[0-9]*\.?[0-9]+', lambda s: '${}$'.format(s[0]), x)
        return re.sub('_', ' ', x)
    if column_names:
        table.columns = column_names
    else:
        table.columns = map(formatter, (table.columns))

    if covariate_names:
        table.index = [covariate_names[s] if s in covariate_names
                       else s for s in table.index]
    else:
        table.index = map(formatter, (table.index))
    if not transpose:
        string = table.to_latex(column_format=col_format, escape=False,
                          formatters=[formatter] * len(table.columns))
        row = ''.join(['& ({})'.format(i) for i in range(1, len(table.columns) + 1)]) \
              + '\\\\\\' + '\n\\midrule'
        string = re.sub(r'\\midrule', row, string)
    else:
        table = table.T.copy()
        string = table.to_latex(column_format=col_format, escape=False,
                          formatters=[formatter] * len(table.columns))




    if not caption:
        caption = 'caption here'
    if not label:
        label = '_'.join(map(lambda s: re.sub(r'\W+','',s),
                             caption.lower().split()))

    output = r'''
\begin{table}[%s]
\caption{%s}
\label{tab:%s}
\centering
\vspace{1em}
\begin{threeparttable}
%s
\begin{tablenotes}
\footnotesize
\item \textsuperscript{*}$p<.1$,
\textsuperscript{**}$p<.05$,
\textsuperscript{***}$p<.01$. %s
\end{tablenotes}
\end{threeparttable}

\end{table}
''' % (table_opt, caption, label, string, notes)
    if filename:
        with open(filename, 'w') as f:
            f.write(output)

    return output

def save_string(filename, s):
    with open(filename, 'w') as f:
        f.write(s)
