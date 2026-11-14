"""Lay the survey counts out on the plot grid and render the census map.

python decode.py counts.csv data/survey_plots.csv figure.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # own modules (run with python -I)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font5x7 import FONT


def grid_indices(plots):
    P = plots[['x_m', 'y_m']].values
    c = P - P.mean(0)
    # principal axis of the plot cloud = flight-line direction
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ax = vt[0] if vt[0, 0] > 0 else -vt[0]           # long axis, pointing east-ish
    perp = np.array([-ax[1], ax[0]])                  # 90 deg anticlockwise = "up"
    u, v = c @ ax, c @ perp

    def snap(t):
        # estimate spacing from the gaps between sorted coordinates
        s = np.sort(t); d = np.diff(s); gaps = d[d > 0.3 * d.max()]
        n = len(gaps) + 1
        L = (t.max() - t.min()) / (n - 1)
        return np.round((t - t.min()) / L).astype(int), n

    col, ncol = snap(u)
    row, nrow = snap(v)
    return col, nrow - 1 - row, ncol, nrow, np.degrees(np.arctan2(ax[1], ax[0]))


def otsu(v):
    v = np.sort(np.asarray(v, float)); best = (0, np.median(v))
    for t in np.unique(v)[1:]:
        a, b = v[v < t], v[v >= t]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best[0]:
            best = (w, t)
    return best[1]


def read_word(M, thr=None):
    """Split the 9 x 37 bitmap into 5x7 cells (1-plot margin, 1-plot gap) and match
    each against the 5x7 font. Returns word, per-letter (best, hamming, 2nd, hamming)."""
    if thr is None:
        thr = otsu(M.ravel())
    B = M >= thr
    nrow, ncol = B.shape
    n = (ncol - 1) // 6
    word, info = '', []
    for k in range(n):
        blk = B[1:8, 1 + 6 * k: 6 + 6 * k]
        d = sorted((int((blk != t).sum()), ch) for ch, t in FONT.items())
        word += d[0][1]; info.append((d[0][1], d[0][0], d[1][1], d[1][0]))
    return word, info


def main(counts_csv, plots_csv, out_png):
    plots = pd.read_csv(plots_csv)
    cnt = pd.read_csv(counts_csv)
    df = plots.merge(cnt, on='image')
    col, row, ncol, nrow, ang = grid_indices(df)
    M = np.full((nrow, ncol), np.nan)
    M[row, col] = df['count'].values
    assert np.isfinite(M).all(), 'grid not fully populated'
    thr = otsu(df['count'].values)
    print(f'grid {nrow} rows x {ncol} cols, flight-line angle {ang:.1f} deg, Otsu threshold {thr:.1f}')
    for r in range(nrow):
        print(''.join('#' if x >= thr else '.' for x in M[r]))
    word, info = read_word(M, thr)
    for ch, d1, ch2, d2 in info:
        print(f'  letter {ch}: {d1} of 35 dots differ from font; runner-up {ch2} ({d2} differ)')
    print('WORD:', word, '-> flag SIGMOID_' + word)
    fig, axs = plt.subplots(2, 1, figsize=(16, 7.5), gridspec_kw={'height_ratios': [1, 1]})
    im = axs[0].imshow(M, cmap='viridis', vmax=np.percentile(M, 99))  # one 150+ plot would wash out the scale
    for r in range(nrow):
        for c in range(ncol):
            axs[0].text(c, r, f'{M[r, c]:.0f}', ha='center', va='center', fontsize=6,
                        color='w' if M[r, c] < thr else 'k')
    axs[0].set_title('Sheep count per survey plot (plot grid rotated to the flight lines)')
    plt.colorbar(im, ax=axs[0], fraction=0.015)
    axs[1].imshow(M >= thr, cmap='Greys')
    axs[1].set_title(f'Plots with count >= {thr:.0f} (Otsu threshold)  ->  reads "{word}"')
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(out_png, dpi=130)
    return M, thr, word


if __name__ == '__main__':
    main(*sys.argv[1:4])
