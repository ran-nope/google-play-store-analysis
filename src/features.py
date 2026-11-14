import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def engineer(cleaned):
    df = cleaned.copy()
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce', format='mixed')
    age = (df['Last Updated'].max() - df['Last Updated']).dt.days
    if age.notna().sum() == 0:
        raise ValueError('At least one valid update date is required.')
    df['days_since_update'] = age.fillna(age.median())
    columns = ['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'days_since_update']
    skew = df[columns].skew().round(2)
    transformed = [c for c in columns if c != 'Rating' and abs(skew[c]) > 1]
    for col in transformed:
        df[f'log_{col}'] = np.log1p(df[col])
    scale = ['Rating'] + [f'log_{c}' for c in transformed] + ['days_since_update']
    for suffix, scaler in [('std', StandardScaler()), ('mm', MinMaxScaler())]:
        values = scaler.fit_transform(df[scale])
        for i, col in enumerate(scale):
            df[f'{col}_{suffix}'] = values[:, i]
    df['Type_enc'] = LabelEncoder().fit_transform(df.Type)
    df = pd.get_dummies(df, columns=['Content Rating'], prefix='CR', dtype=int)
    top = df.Category.value_counts().head(10).index
    df['Category_grouped'] = np.where(df.Category.isin(top), df.Category, 'OTHER')
    return pd.get_dummies(df, columns=['Category_grouped'], prefix='Cat', dtype=int)
