"""MONGOLIA (The Baa-rilliant task) end-to-end solution.

1. Counting: density-map regression (counter.py: ImageNet ConvNeXt-Tiny + small FPN
   head, trained on the train_points.csv dots), 5-fold CV on the 280 annotated photos.
   If runs/fold*.pt are missing they are trained first (CUDA GPU, ~8 min per fold).
2. Survey counts: mean of the 5 fold models x 8 dihedral TTA -> counts.csv.
3. Decoding (decode.py): the 333 plots sit on a 9 x 37 grid rotated 27 deg (the flight
   lines). Thresholding the counts on that grid gives a 5x7 dot-matrix bitmap; each
   5x7 cell is matched to a dot-matrix font -> the word. figure.png is the census map.
4. robustness.py: Monte Carlo re-decoding under held-out-sized counting errors.

python -I solve.py [--data data] [--runs runs] [--recount]
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # own modules (run with python -I)
import numpy as np, pandas as pd, torch
import counter, decode, robustness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--runs', default='runs')
    ap.add_argument('--recount', action='store_true', help='re-run survey inference even if cached')
    a = ap.parse_args()
    torch.set_num_threads(4)
    w = sorted(glob.glob(f'{a.runs}/fold*.pt'))
    if not w:
        sys.argv = ['counter.py', '--data', a.data, '--out', a.runs]
        counter.main()
        w = sorted(glob.glob(f'{a.runs}/fold*.pt'))
    sv = pd.read_csv(f'{a.data}/survey_plots.csv')
    dev = counter.get_device()
    per_fold = []
    for p in w:
        k = os.path.basename(p)[4:-3]
        cache = f'{a.runs}/survey_fold{k}.csv'
        if os.path.exists(cache) and not a.recount:
            per_fold.append(pd.read_csv(cache).set_index('image').loc[sv.image, 'pred'].values)
            continue
        m = counter.Net(pretrained=False)
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m = m.to(dev).eval()
        ps = [counter.predict([m], counter.load_img(f'{a.data}/survey/{n}'), dev) for n in sv.image]
        pd.DataFrame({'image': sv.image, 'pred': ps}).to_csv(cache, index=False)
        per_fold.append(np.array(ps))
    pred = np.mean(per_fold, 0)
    pd.DataFrame({'image': sv.image, 'pred': pred}).to_csv(f'{a.runs}/survey_pred_mean.csv', index=False)
    pd.DataFrame({'image': sv.image, 'count': np.round(pred).astype(int)}).to_csv('counts.csv', index=False)
    print(f'counts.csv written: {len(sv)} survey photos, {len(per_fold)} fold models; '
          f'total sheep {int(np.round(pred).sum())}')
    oof = glob.glob(f'{a.runs}/oof_fold*.csv')
    if oof:
        o = pd.concat([pd.read_csv(f) for f in oof])
        print(f'held-out MAE on train ({len(o)} photos, {len(oof)} folds): {np.abs(o.pred - o["count"]).mean():.2f}')
    _, _, word = decode.main('counts.csv', f'{a.data}/survey_plots.csv', 'figure.png')
    robustness.main('counts.csv', a.runs, f'{a.data}/survey_plots.csv', trials=500)
    print(f'\nFLAG: SIGMOID_{word}')


if __name__ == '__main__':
    main()
