# Google Play Store analysis

A reproducible descriptive analysis of a historical Play Store sample.

- **Free apps make up 92.61% of cleaned records.** This sample is not a current census, and free pricing does not establish a freemium business model.
- **Reviews and install buckets are strongly associated (Spearman 0.971).** Installs are bucket lower bounds; association does not show that reviews cause growth.
- **The cleaned median rating is 4.3/5.** Missing ratings are median-imputed, so this is not an unbiased estimate of observed user satisfaction.

Five PCA components retain **95.08%** of selected-feature variance. This is descriptive variance retained, not predictive accuracy or evidence of launch success.

## Reproduce

Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python pipeline.py --input googleplaystore.csv --output outputs
```

The pipeline writes cleaned/quarantined records, engineered features, PCA coordinates and `summary.json`. The supplied data produces 10,357 clean records from 10,841 raw rows, after removing exact duplicates and quarantining one malformed record. Repeated app names remain distinct observations.

`src/cleaning.py`, `src/features.py`, and `src/analysis.py` are the single implementation. `DSA_PROJECT.ipynb` imports the pipeline. Legacy weekly filenames are compatibility entrypoints that run the same complete pipeline and accept the same explicit path arguments. They no longer contain independent analysis copies.

Scaling, imputation and PCA describe this full sample. A future predictive experiment must fit preprocessing on its training split only. Update age is measured relative to the latest date in the sample.

## Data and license

Source: [Google Play Store Apps by lava18](https://www.kaggle.com/datasets/lava18/google-play-store-apps). Dataset rights remain with their original licensors; the MIT license applies to original project code only.

Planned repository rename: `DSV_PROJECT` → `google-play-store-analysis`.
