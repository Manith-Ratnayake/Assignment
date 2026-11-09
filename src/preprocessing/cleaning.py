def compute_iqr_bounds(values, multiplier=1.5):
    """Return the (lower, upper) IQR fences of a series.

    Pass only the training rows, so validation and test values never influence
    the bounds. The lower fence is floored at 0 because energy use can't be negative.
    """
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = max(0, q1 - multiplier * iqr)
    upper = q3 + multiplier * iqr
    return lower, upper


def clip_outliers(df, column, lower, upper):
    """Return a copy of df with one column clipped to [lower, upper].

    Clipping keeps every row, so the 10-minute time steps stay continuous
    for the lag and rolling features built afterwards.
    """
    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df
