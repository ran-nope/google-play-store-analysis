import numpy as np
import pandas as pd


def parse_size(value):
    value = str(value).strip()
    try:
        if value.endswith('M'):
            return float(value[:-1])
        if value.endswith('k'):
            return float(value[:-1]) / 1024
    except ValueError:
        pass
    return np.nan


def clean(raw):
    df = raw.copy()
    rating = pd.to_numeric(df.Rating, errors='coerce')
    installs = pd.to_numeric(df.Installs.astype(str).str.replace(r'[,+]', '', regex=True), errors='coerce')
    price = pd.to_numeric(df.Price.astype(str).str.replace('$', '', regex=False), errors='coerce')
    reviews = pd.to_numeric(df.Reviews, errors='coerce')
    valid = (df.Rating.isna() | rating.between(1, 5)) & installs.ge(0) & price.ge(0) & reviews.ge(0)
    quarantine = raw.loc[~valid].copy()
    df = df.loc[valid].drop_duplicates().copy()
    df['Rating'] = rating.loc[df.index]
    df['Installs'] = installs.loc[df.index]
    df['Price'] = price.loc[df.index]
    df['Reviews'] = reviews.loc[df.index]
    df['Size'] = df.Size.map(parse_size)
    for column in ['Rating', 'Size']:
        median = df[column].median()
        if pd.isna(median):
            raise ValueError(f'No valid {column} values to impute.')
        df[column] = df[column].fillna(median)
    modes = df.Type.mode()
    if modes.empty:
        raise ValueError('No valid app Type to impute.')
    df['Type'] = df.Type.fillna(modes.iloc[0])
    return df, quarantine
