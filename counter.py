"""Sheep counter: density-map regression with a pretrained ConvNeXt-Tiny backbone.

Usage (GPU machine):
  python counter.py --data data --out runs --folds 5 --epochs 60
Writes runs/oof.csv (held-out predictions for train), runs/survey_pred.csv
(mean of fold models on the survey photos) and runs/fold{k}.pt.
"""
import argparse, os, random, math
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import timm

STRIDE = 8
DSCALE = 100.0  # density targets are multiplied by this for well-scaled regression
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def as_float(im):
    return im.astype(np.float32) / 255.0 if im.dtype == np.uint8 else im


def load_img(p):
    with Image.open(p) as im:
        return np.asarray(im.convert('RGB'), np.float32) / 255.0


def density(pts, h, w, sigma=4.0):
    d = np.zeros((h, w), np.float32)
    for x, y in pts:
        xi, yi = min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)
        d[yi, xi] += 1
    return gaussian_filter(d, sigma, mode='constant')


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.bb = timm.create_model('convnext_tiny', pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        try:
            self.bb.set_grad_checkpointing(True)  # keep VRAM ~1 GB on the shared GPU
        except Exception:
            pass
        ch = self.bb.feature_info.channels()  # strides 8,16,32
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch])
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(64, 1, 1))

    def forward(self, x):
        f = self.bb(x)
        y = self.lat[0](f[0])
        for i in (1, 2):
            y = y + F.interpolate(self.lat[i](f[i]), size=y.shape[-2:], mode='bilinear', align_corners=False)
        return F.softplus(self.head(y))  # density per stride-8 cell


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, dens, train, crop=384):
        self.imgs, self.dens, self.train, self.crop = imgs, dens, train, crop

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im, d = as_float(self.imgs[i]), self.dens[i].astype(np.float32)
        if self.train:
            s = random.uniform(0.75, 1.25)
            h, w = im.shape[:2]
            nh, nw = int(h * s) // STRIDE * STRIDE, int(w * s) // STRIDE * STRIDE
            imt = torch.from_numpy(im).permute(2, 0, 1)[None]
            dt = torch.from_numpy(d)[None, None]
            tot = dt.sum()
            imt = F.interpolate(imt, size=(nh, nw), mode='bilinear', align_corners=False)[0]
            dt = F.interpolate(dt, size=(nh, nw), mode='bilinear', align_corners=False)[0]
            if dt.sum() > 0:
                dt = dt * (tot / dt.sum())
            c = min(self.crop, nh, nw)
            y0, x0 = random.randint(0, nh - c), random.randint(0, nw - c)
            y0, x0 = y0 // STRIDE * STRIDE, x0 // STRIDE * STRIDE
            imt, dt = imt[:, y0:y0 + c, x0:x0 + c], dt[:, y0:y0 + c, x0:x0 + c]
            if random.random() < 0.5:
                imt, dt = imt.flip(-1), dt.flip(-1)
            if random.random() < 0.5:
                imt, dt = imt.flip(-2), dt.flip(-2)
            k = random.randint(0, 3)
            imt, dt = torch.rot90(imt, k, (1, 2)), torch.rot90(dt, k, (1, 2))
            # colour jitter
            imt = imt * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)
            g = imt.mean(0, keepdim=True)
            imt = (g + (imt - g) * random.uniform(0.7, 1.3)).clamp(0, 1)
            if c < self.crop:  # pad to fixed size
                p = self.crop - c
                imt, dt = F.pad(imt, (0, p, 0, p)), F.pad(dt, (0, p, 0, p))
        else:
            imt, dt = torch.from_numpy(im).permute(2, 0, 1), torch.from_numpy(d)[None]
        imt = (imt - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        dt = F.avg_pool2d(dt[None], STRIDE)[0] * STRIDE * STRIDE
        return imt.float(), dt.float()


@torch.no_grad()
def predict(models, im, dev):
    x = torch.from_numpy(as_float(im)).permute(2, 0, 1)
    x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None].to(dev)
    outs = []
    for m in models:
        m.eval()
        for k in range(4):
            for fl in (False, True):
                xx = torch.rot90(x, k, (2, 3))
                if fl:
                    xx = xx.flip(-1)
                with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                    outs.append(m(xx).float().sum().item() / DSCALE)
    return float(np.mean(outs))


def train_one(tr_imgs, tr_dens, dev, epochs, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = Net().to(dev)
    dl = torch.utils.data.DataLoader(DS(tr_imgs, tr_dens, True), batch_size=2, shuffle=True,
                                     num_workers=2, drop_last=True, persistent_workers=True)
    m.train()
    opt = torch.optim.AdamW([{'params': m.bb.parameters(), 'lr': 1e-4},
                             {'params': list(m.lat.parameters()) + list(m.head.parameters()), 'lr': 1e-3}],
                            weight_decay=1e-4)
    steps = epochs * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
    cuda = dev == 'cuda'
    scaler = torch.amp.GradScaler(enabled=cuda)
    for ep in range(epochs):
        m.train(); tl = 0
        for x, d in dl:
            x, d = x.to(dev), d.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
                p = m(x)
            p = p.float(); d = d * DSCALE
            loss = F.mse_loss(p, d) + 0.01 * (p.sum((1, 2, 3)) - d.sum((1, 2, 3))).abs().mean() / DSCALE * 10
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tl += loss.item()
        if ep % 10 == 9 or ep == 0:
            print(f'  ep {ep+1} loss {tl/len(dl):.4f}', flush=True)
    return m


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='runs')
    ap.add_argument('--folds', type=int, default=5); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--only', type=int, nargs='*', default=None, help='train only these fold ids')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = get_device()
    torch.set_num_threads(2 if dev == 'cuda' else 4)
    cnt = pd.read_csv(f'{a.data}/train_counts.csv')
    pts = pd.read_csv(f'{a.data}/train_points.csv')
    imgs, dens = [], []
    for n in cnt.image:
        im = load_img(f'{a.data}/train/{n}')
        pp = pts[pts.image == n][['x', 'y']].values
        # compact storage (uint8 / float16) keeps RAM low with DataLoader workers
        imgs.append((im * 255).round().astype(np.uint8)); dens.append(density(pp, *im.shape[:2]).astype(np.float16))
    y = cnt['count'].values
    rng = np.random.RandomState(0); perm = rng.permutation(len(y)); fold = np.zeros(len(y), int)
    for k in range(a.folds):
        fold[perm[k::a.folds]] = k
    sv = pd.read_csv(f'{a.data}/survey_plots.csv')
    for k in (a.only if a.only is not None else range(a.folds)):
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        m = train_one([imgs[i] for i in tr], [dens[i] for i in tr], dev, a.epochs, seed=k)
        torch.save(m.state_dict(), f'{a.out}/fold{k}.pt')
        pv = np.array([predict([m], imgs[i], dev) for i in va])
        print(f'fold {k} held-out MAE {np.abs(pv - y[va]).mean():.2f}', flush=True)
        pd.DataFrame({'image': cnt.image.values[va], 'count': y[va], 'pred': pv, 'fold': k}).to_csv(
            f'{a.out}/oof_fold{k}.csv', index=False)
        ps = [predict([m], load_img(f'{a.data}/survey/{n}'), dev) for n in sv.image]
        pd.DataFrame({'image': sv.image, 'pred': ps}).to_csv(f'{a.out}/survey_fold{k}.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
