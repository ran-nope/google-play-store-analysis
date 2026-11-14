from pathlib import Path
import pandas as pd
import pytest
from src.cleaning import clean, parse_size
from src.features import engineer
from pipeline import run

ROOT = Path(__file__).parents[1]


def test_supplied_dataset_and_row_preservation(tmp_path):
    result = run(ROOT / 'googleplaystore.csv', tmp_path)
    assert result['clean_rows'] == 10357
    assert result['quarantine_rows'] == 1
    assert len(pd.read_csv(tmp_path / 'engineered.csv')) == len(pd.read_csv(tmp_path / 'pca.csv')) == 10357
    assert result['variance_retained'] >= .95


def test_parsing_quarantine_imputation():
    raw = pd.read_csv(ROOT / 'googleplaystore.csv').head(5)
    raw.loc[0, 'Rating'] = 19
    raw.loc[1, 'Rating'] = float('nan')
    raw.loc[2, 'Size'] = 'Varies with device'
    raw.loc[3, 'Last Updated'] = 'not a date'
    cleaned, rejected = clean(raw)
    assert len(rejected) == 1 and len(cleaned) == 4
    assert cleaned[['Rating', 'Size']].notna().all().all()
    assert engineer(cleaned).days_since_update.notna().all()
    assert parse_size('1024k') == 1
    assert parse_size('19M') == 19
    with pytest.raises(ValueError):
        engineer(cleaned.assign(**{'Last Updated': 'invalid'}))
