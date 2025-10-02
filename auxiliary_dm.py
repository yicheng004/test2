import numpy as np
import pandas as pd


def _tick_loss(errors: np.ndarray, q: float) -> np.ndarray:
    """Tick loss for quantile q in [0,1]."""
    return np.where(errors >= 0, q * errors, (q - 1.0) * errors)


def dm_test(y_true,
            yhat1,
            yhat2,
            h: int = 1,
            power=2,
            alternative: str = 'two-sided',
            use_hln: bool = True):
    """
    Diebold–Mariano (DM) test for equal predictive accuracy between two models.

    Parameters
    - y_true: array-like of shape (T,) – observations
    - yhat1: array-like of shape (T,) – predictions from model 1
    - yhat2: array-like of shape (T,) – predictions from model 2
    - h: int – forecast horizon (e.g., 48 for day-ahead with 30-min steps)
    - power: int|float or ('QL', q)
        * int/float: use |e|**power loss (power=2 => squared error, power=1 => absolute error)
        * ('QL', q): use quantile tick loss at level q in (0,1)
    - alternative: {'two-sided','greater','less'} – hypothesis direction for E[d_t]
      with d_t = L(e1_t) - L(e2_t)
    - use_hln: bool – apply Harvey–Leybourne–Newbold small-sample adjustment for h>1

    Returns
    - DM_stat: float – test statistic
    - p_value: float – p-value
    - T: int – effective sample size

    Notes
    - If DM_stat > 0 (with d_t = L1 - L2), model 2 tends to have lower loss.
    - The Newey–West (HAC) variance is used with bandwidth M = h-1.
    """
    y_true = np.asarray(y_true, dtype=float)
    yhat1 = np.asarray(yhat1, dtype=float)
    yhat2 = np.asarray(yhat2, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(yhat1) & np.isfinite(yhat2)
    y_true, yhat1, yhat2 = y_true[mask], yhat1[mask], yhat2[mask]

    e1 = y_true - yhat1
    e2 = y_true - yhat2

    if isinstance(power, (int, float)):
        L1 = np.abs(e1) ** float(power)
        L2 = np.abs(e2) ** float(power)
    elif isinstance(power, (tuple, list)) and len(power) == 2 and str(power[0]).upper() == 'QL':
        q = float(power[1])
        if not (0.0 < q < 1.0):
            raise ValueError("Quantile q must be in (0,1)")
        L1 = _tick_loss(e1, q)
        L2 = _tick_loss(e2, q)
    else:
        raise ValueError("power must be int/float or ('QL', q)")

    d = L1 - L2
    T = d.size
    if T < 2:
        return np.nan, np.nan, T

    dbar = d.mean()

    # Newey–West HAC variance with bandwidth M = h-1
    M = max(int(h) - 1, 0)
    d_centered = d - dbar
    gamma0 = np.mean(d_centered * d_centered)
    S = gamma0
    if M > 0:
        for k in range(1, min(M, T - 1) + 1):
            gk = np.mean(d_centered[k:] * d_centered[:-k])
            w = 1.0 - k / (M + 1.0)
            S += 2.0 * w * gk

    var_dbar = S / T
    if not np.isfinite(var_dbar) or var_dbar <= 0:
        return np.nan, np.nan, T

    DM = dbar / np.sqrt(var_dbar)

    # Harvey–Leybourne–Newbold small-sample adjustment
    if use_hln and h > 1:
        adj = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        if np.isfinite(adj) and adj > 0:
            DM *= adj

    # p-value
    p_value = np.nan
    try:
        from scipy.stats import t as student_t
        df = max(T - 1, 1)
        if alternative == 'two-sided':
            p_value = 2.0 * (1.0 - student_t.cdf(abs(DM), df=df))
        elif alternative == 'greater':
            p_value = 1.0 - student_t.cdf(DM, df=df)
        elif alternative == 'less':
            p_value = student_t.cdf(DM, df=df)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    except Exception:
        # normal approximation fallback
        from math import erf, sqrt

        def norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))

        if alternative == 'two-sided':
            p_value = 2.0 * (1.0 - norm_cdf(abs(DM)))
        elif alternative == 'greater':
            p_value = 1.0 - norm_cdf(DM)
        elif alternative == 'less':
            p_value = norm_cdf(DM)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return float(DM), float(p_value), int(T)


def dm_test_from_forecasts(df_model1: pd.DataFrame,
                           df_model2: pd.DataFrame,
                           truth_col: str = 'Truth',
                           pred_col1: str = 'Forecasts',
                           pred_col2: str = 'Forecasts',
                           h: int = 1,
                           power=2,
                           alternative: str = 'two-sided',
                           use_hln: bool = True,
                           date_lower=None,
                           date_upper=None):
    """
    Convenience wrapper to run DM test by aligning two forecast DataFrames on index.

    df_model1 and df_model2 must each contain columns [truth_col, pred_colX] and be
    indexed by timestamp (DatetimeIndex).

    Optional date_lower/date_upper (inclusive bounds) allow restricting the window.
    """
    if not isinstance(df_model1.index, pd.DatetimeIndex):
        raise ValueError("df_model1 must be indexed by DatetimeIndex")
    if not isinstance(df_model2.index, pd.DatetimeIndex):
        raise ValueError("df_model2 must be indexed by DatetimeIndex")

    df1 = df_model1[[truth_col, pred_col1]].rename(columns={pred_col1: 'pred1'})
    df2 = df_model2[[truth_col, pred_col2]].rename(columns={truth_col: 'Truth', pred_col2: 'pred2'})

    df = df1.join(df2[['pred2']], how='inner')

    # Robust date filtering that ignores timezone differences by comparing dates
    if date_lower is not None or date_upper is not None:
        dates = df.index.date
        mask = np.ones(len(df), dtype=bool)
        if date_lower is not None:
            dlow = pd.to_datetime(date_lower).date()
            mask &= dates >= dlow
        if date_upper is not None:
            dhigh = pd.to_datetime(date_upper).date()
            mask &= dates <= dhigh
        df = df[mask]

    df = df.dropna()
    DM, p, T = dm_test(df[truth_col].to_numpy(), df['pred1'].to_numpy(), df['pred2'].to_numpy(),
                       h=h, power=power, alternative=alternative, use_hln=use_hln)
    return DM, p, T


def dm_summary(DM: float, p_value: float, alpha: float = 0.05) -> str:
    """Return a human-readable summary string for the DM test result."""
    if not np.isfinite(DM) or not np.isfinite(p_value):
        return "DM test could not be computed (insufficient or invalid data)."
    signif = 'significant' if p_value < alpha else 'not significant'
    direction = 'Model 2 better (lower loss)' if DM > 0 else ('Model 1 better (lower loss)' if DM < 0 else 'No difference')
    return f"DM={DM:.4f}, p={p_value:.4f} ({signif}); direction: {direction}"


# ---------------- Convenience helpers for this project -----------------
def _daily_error_stats_from_forecasts(fore_truth: pd.DataFrame,
                                      truth_col: str = 'Truth',
                                      pred_col: str = 'Forecasts') -> pd.DataFrame:
    """
    Compute per-day RMSE/MAE/SMAPE from a forecast dataframe that contains
    columns `Truth` and `Forecasts` at 30-minute resolution and a DatetimeIndex.

    Returns a dataframe indexed by date with columns ['rmse','mae','smape'].
    """
    df = fore_truth.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Forecast dataframe must be indexed by DatetimeIndex')
    if truth_col not in df.columns or pred_col not in df.columns:
        raise ValueError('Forecast dataframe must have Truth and Forecasts columns')

    tmp = df[[truth_col, pred_col]].dropna().copy()
    tmp['date'] = tmp.index.date
    err = pd.DataFrame(index=pd.unique(tmp['date']), columns=['rmse', 'mae', 'smape'], dtype=float)
    # vectorized within groups
    for d, block in tmp.groupby('date'):
        e = block[truth_col] - block[pred_col]
        rmse = float(np.sqrt(np.mean(np.square(e)))) if len(block) else np.nan
        mae = float(np.mean(np.abs(e))) if len(block) else np.nan
        denom = (block[truth_col].abs() + block[pred_col].abs()) / 2.0
        smape = float(np.mean(np.where(denom != 0, np.abs(e) / denom, 0.0))) if len(block) else np.nan
        err.loc[d, ['rmse', 'mae', 'smape']] = [rmse, mae, smape]
    return err


def pick_best_by_rmse(forecasts_map: dict,
                      truth_col: str = 'Truth',
                      pred_col: str = 'Forecasts',
                      date_lower=None,
                      date_upper=None) -> tuple:
    """
    Given a mapping name -> forecast dataframe, compute mean daily RMSE and
    return (best_name, best_df, best_rmse).

    Optional date_lower/date_upper (inclusive) to restrict comparison window.
    """
    best_name, best_df, best_rmse = None, None, np.inf
    for name, df in forecasts_map.items():
        try:
            df2 = df
            if date_lower is not None or date_upper is not None:
                dates = df2.index.date
                mask = np.ones(len(df2), dtype=bool)
                if date_lower is not None:
                    dlow = pd.to_datetime(date_lower).date()
                    mask &= dates >= dlow
                if date_upper is not None:
                    dhigh = pd.to_datetime(date_upper).date()
                    mask &= dates <= dhigh
                df2 = df2[mask]
            stats = _daily_error_stats_from_forecasts(df2, truth_col, pred_col)
            rmse_mean = float(stats['rmse'].mean())
            if np.isfinite(rmse_mean) and rmse_mean < best_rmse:
                best_name, best_df, best_rmse = name, df, rmse_mean
        except Exception:
            continue
    return best_name, best_df, best_rmse


def run_dm_between_best(baseline_forecasts_map: dict,
                        nlp_forecasts_map: dict,
                        h: int = 48,
                        power=2,
                        alternative: str = 'two-sided',
                        use_hln: bool = True,
                        date_lower=None,
                        date_upper=None) -> dict:
    """
    Select the best baseline and best NLP models (lowest mean daily RMSE)
    from the provided forecasts maps and run DM test between them.

    Returns a dict with selected names and DM test results.
    """
    base_name, base_df, base_rmse = pick_best_by_rmse(baseline_forecasts_map, date_lower=date_lower, date_upper=date_upper)
    nlp_name, nlp_df, nlp_rmse = pick_best_by_rmse(nlp_forecasts_map, date_lower=date_lower, date_upper=date_upper)
    if base_df is None or nlp_df is None:
        return {
            'baseline_best': base_name,
            'nlp_best': nlp_name,
            'DM': np.nan,
            'p_value': np.nan,
            'T': 0,
            'summary': 'Insufficient forecasts to run DM.'
        }

    DM, p, T = dm_test_from_forecasts(base_df, nlp_df,
                                       truth_col='Truth',
                                       pred_col1='Forecasts',
                                       pred_col2='Forecasts',
                                       h=h, power=power, alternative=alternative,
                                       use_hln=use_hln,
                                       date_lower=date_lower,
                                       date_upper=date_upper)
    return {
        'baseline_best': base_name,
        'nlp_best': nlp_name,
        'DM': DM,
        'p_value': p,
        'T': T,
        'summary': dm_summary(DM, p)
    }
