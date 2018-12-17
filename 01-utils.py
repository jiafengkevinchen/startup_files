import numpy as np
from statsmodels.iolib.summary2 import summary_col
import re
import pandas as pd


def skipna(f):
    def _wrapped(x):
        if type(x) is float and np.isnan(x):
            return np.nan
        else:
            return f(x)
    return _wrapped


def skiperror(f, return_val=None, return_x=False):
    def _wrapped(x):
        try:
            return f(x)
        except:
            if return_x:
                return x
            return return_val
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
    d = {'$N$': lambda x: '{0:d}'.format(int(x.nobs)),
         '$R^2$': lambda x: '{:.2f}'.format(x.rsquared)}
    t = summary_col(regs, stars=stars, info_dict=d, **kwargs).tables[0].copy()
    if notes:
        for k, v in notes.items():
            t.loc[k] = v
    return t


def mathify(s, num_digit=3, large=9, phantom_space=False,
            max_stars=0, na_rep="---",
            auto_scientific=True, se=False):
    stars = ''
    n_stars = 0
    if (type(s) is float and np.isnan(s)) or s == "NaN":
        return na_rep
    if type(s) is str and s[-1] == "*":
        stars = re.findall(r'\*+', s)[0]
        n_stars = len(stars)
        s = float(re.sub(r'\*+', '', s))
    phantom = max(max_stars - n_stars, 0)

    if type(s) is str and s[0] == "(" and s[-1] == ")":
        s = f"({mathify(float(s[1:-1]), num_digit=num_digit, large=large, max_stars=0, auto_scientific=auto_scientific)})"
    else:
        if type(s) is str:
            s = float(s)
        if (abs(s) / (10**large) > 1 or abs(s) < 10 ** -num_digit) and s != 0 and auto_scientific:
            s = ((f'%.{num_digit}E') % s)
            lst = s.split("E")
            s = f"{lst[0]} \\times 10^{{{int(lst[-1])}}}"
        elif abs(s - round(s)) < 1e-10:
            s = "{:,}".format(int(round(s)))

        else:
            s = ('{:.' + str(num_digit) + 'f}').format(s)
        s = f'${s}$'
    if phantom_space:
        if phantom == 0:
            phantom = ""
        else:
            phantom = '\\phantom{{{}}}'.format('*' * phantom)
    else:
        phantom = ""

    return f"{s}\\textsuperscript{{{stars + phantom}}}"


def mathify_column(x, **kwargs):
    max_stars = x.apply(
        skiperror(lambda s: len(re.findall(r'\*+', s)[0]),
                  return_val=0)).max()
    return x.apply(skiperror(lambda y: mathify(y, max_stars=max_stars, **kwargs), return_x=True))


def mathify_table(df, tt_index=False, tt_cols=False, **kwargs):
    d = df.copy()
    d = d.apply(skiperror(lambda x: mathify_column(x, **kwargs), return_x=True))
    if tt_index:
        d.index = ["\\texttt{{{}}}".format(s.replace("_", "\\_"))
                   if (len(s) > 0 and s[0] != "$" and s[-1] != "$")
                   else s for s in d.index.astype(str)]
    if tt_cols:
        d.columns = ["\\texttt{{{}}}".format(s.replace("_", "\\_"))
                           if (len(s) > 0 and s[0] != "$" and s[-1] != "$")
                           else s for s in np.array(d.columns).astype(str)]
    return d

def interleave(l1, l2):
    return [val for pair in zip(l1, l2) for val in pair]


def consolidate_se(df, coef_cols, se_cols, add_stars=False):
    from scipy.stats import norm
    rest = list(filter(lambda x : x not in coef_cols and x not in se_cols, df.columns))
    return_df = []
    for coef_col, se_col in zip(coef_cols, se_cols):
        p_vals = norm.cdf(-np.abs(df[coef_col]/df[se_col]).values) * 2
        se = df[se_col].apply(lambda x : f"({x})")
        coef = df[coef_col].copy()
        if add_stars:
            for i, p in enumerate(p_vals):
                if p < .01:
                    coef.iloc[i] = str(coef.values[i]) + '***'
                elif p < .05:
                    coef.iloc[i] = str(coef.values[i]) + '**'
                elif p < .1:
                    coef.iloc[i] = str(coef.values[i]) + '*'
                else:
                    coef.iloc[i] = str(coef.values[i])

        v = interleave(coef.values, se.values)
        return_df.append(pd.Series(v, index=interleave(coef.index,
            [''] * len(se)), name=coef_col))

    for coef_col in rest:
        coef = df[coef_col].copy()
        v = interleave(coef.values, [""] * len(coef))
        return_df.append(pd.Series(v, index=interleave(coef.index,
            [''] * len(coef)), name=coef_col))
    col_order = list(filter(lambda x : x in coef_cols or x in rest, df.columns))
    return pd.concat(return_df, axis=1, sort=False)[col_order].copy()


def to_table(df, caption,
             mathify_first=True,
             label=None,
             filename=None,
             insert_column_number=True,
             notes=None,
             include_star_notes=True,
             mathify_args=dict(),
             to_latex_args=dict()):
    if not label:
        label = '_'.join(map(lambda s: re.sub(
            r'\W+', '', s), caption.lower().split()))
    if mathify_first:
        d = df.copy()
        t = mathify_table(d, **mathify_args)

    else:
        t = df.copy()

    if insert_column_number:
        t = t.T.set_index(np.array([f"\hypertarget{{tabcol:{label + str(r)}}}{{({r})}}"
                for r in range(1, t.shape[-1]+1)]), append=True).T.copy()

    if "column_format" not in to_latex_args:
        to_latex_args['column_format'] = "l" + "c" * df.shape[-1]

    opt_val = pd.get_option('max_colwidth')
    pd.set_option('max_colwidth', 10000)
    table_str = t.to_latex(escape=False, **to_latex_args)
    pd.set_option('max_colwidth', opt_val)

    # No threeparttable
    if notes is None:
        s = f"""\\begin{{table}}[tbh]
        \\caption{{{caption}}}
        \\label{{tab:{label}}}
        \\centering
        \\vspace{{1em}}
        {table_str}
        \\end{{table}}"""

    # Threeparttable
    else:
        if include_star_notes:
            notes = f"""\\textsuperscript{{*}}$p<.1$,
                    \\textsuperscript{{**}}$p<.05$,
                    \\textsuperscript{{***}}$p<.01$. {notes}"""
        s = f'''
        \\begin{{table}}[tbh]
        \\caption{{{caption}}}
        \\label{{tab:{label}}}
        \\centering
        \\vspace{{1em}}
        \\begin{{threeparttable}}
        {table_str}
        \\begin{{tablenotes}}
        \\footnotesize
        \item {notes}
        \end{{tablenotes}}
        \end{{threeparttable}}

        \end{{table}}
        '''

    if filename:
        save_string(filename, s)
    return s




def as_latex_regtable(table, table_opt='tb',
                      column_names=None, caption=None,
                      label=None, covariate_names=None, notes='', superscript=True,
                      filename=None, transpose=False, col_format=None):
    """
    DEPRECATED
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
            x = re.sub(
                '\*+', lambda s: '\\textsuperscript{{{}}}'.format(s[0]), x)
        if '$' not in x:
            x = re.sub(r'[-+]?[0-9]*\.?[0-9]+',
                       lambda s: '${}$'.format(s[0]), x)
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
        label = '_'.join(map(lambda s: re.sub(r'\W+', '', s),
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
