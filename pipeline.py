import argparse
import json
from pathlib import Path
import pandas as pd
from src.cleaning import clean
from src.features import engineer
from src.analysis import reduce_features


def run(input_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    cleaned, quarantine = clean(raw)
    features = engineer(cleaned)
    reduced, report = reduce_features(features)
    for name, frame in [('cleaned', cleaned), ('quarantine', quarantine), ('engineered', features), ('pca', reduced)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    report.update(raw_rows=len(raw), clean_rows=len(cleaned), quarantine_rows=len(quarantine),
                  free_share=float(cleaned.Type.eq('Free').mean()),
                  reviews_installs_spearman=float(cleaned[['Reviews','Installs']].corr(method='spearman').iloc[0,1]),
                  median_rating=float(cleaned.Rating.median()))
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'googleplaystore.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
