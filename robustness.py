"""How robust is the decoded word to counting errors?

Monte Carlo: perturb the survey counts with errors resampled from the held-out
(out-of-fold) errors on the annotated training photos, scaled x1, x2, x3, x5,
re-threshold (Otsu) and re-read the word. Also additive Gaussian noise.
python robustness.py counts.csv runs data/survey_plots.csv
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import decode

# train_107.jpg is annotated with 0 sheep but clearly shows ~90 (label error, see README);
# it is a ground-truth error, not a model error, so it is left out of the error model.
LABEL_ERRORS = {'train_107.jpg'}


def main(counts_csv, runs, plots_csv, trials=2000):
    df = pd.read_csv(plots_csv).merge(pd.read_csv(counts_csv), on='image')
    col, row, ncol, nrow, _ = decode.grid_indices(df)
    base = df['count'].values.astype(float)
    M0 = np.zeros((nrow, ncol)); M0[row, col] = base
    w0, _ = decode.read_word(M0)
    oof = pd.concat([pd.read_csv(f) for f in glob.glob(f'{runs}/oof_fold*.csv')])
    oof = oof[~oof.image.isin(LABEL_ERRORS)]
    rel = ((oof.pred - oof['count']) / np.maximum(oof['count'], 5)).values
    mae = np.abs(oof.pred - oof['count']).mean()
    print(f'base word {w0}; held-out MAE {mae:.2f} on {len(oof)} photos; '
          f'relative error sd {rel.std():.3f}')
    rng = np.random.default_rng(0)
    res = []
    for kind in ('resampled-relative', 'gaussian-abs'):
        for k in (1, 2, 3, 5):
            ok = 0; words = {}
            for _ in range(trials):
                if kind == 'resampled-relative':
                    e = rng.choice(rel, size=len(base)) * k
                    c = base * (1 + e)
                else:
                    c = base + rng.normal(0, k * mae * 1.25, size=len(base))  # sd of |e| ~ 1.25*MAE
                M = np.zeros((nrow, ncol)); M[row, col] = c
                w, _ = decode.read_word(M)
                words[w] = words.get(w, 0) + 1; ok += (w == w0)
            top = sorted(words.items(), key=lambda t: -t[1])[:4]
            print(f'{kind:20s} x{k}: {ok / trials:6.1%} decode {w0}; most common: {top}')
            res.append((kind, k, ok / trials, top))
    return res


if __name__ == '__main__':
    main(*sys.argv[1:4])
