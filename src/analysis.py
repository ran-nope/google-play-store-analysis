import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def reduce_features(df, threshold=0.85):
    candidates = ['Rating', 'days_since_update', 'Type_enc'] + [c for c in df if c.startswith('log_') and not c.endswith(('_std', '_mm'))]
    correlation = df[candidates].corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    selected = [c for c in candidates if not upper[c].gt(threshold).any()]
    values = StandardScaler().fit_transform(df[selected])
    model = PCA(n_components=0.95, svd_solver='full').fit(values)
    projected = pd.DataFrame(model.transform(values), index=df.index, columns=[f'PC{i+1}' for i in range(model.n_components_)])
    return projected, {'selected_features': selected, 'components': int(model.n_components_), 'variance_retained': float(model.explained_variance_ratio_.sum())}
