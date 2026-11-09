from sklearn.model_selection import train_test_split


def split_data(df, train_ratio, val_ratio):
    """Cut df into consecutive train, val and test blocks. shuffle=False keeps the time order."""
    test_ratio = 1 - train_ratio - val_ratio
    train_val, test = train_test_split(df, test_size=test_ratio, shuffle=False)

    # val_ratio is a share of the whole table, so convert it to a share of train_val
    val_share = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_share, shuffle=False)

    return {"train": train, "val": val, "test": test}