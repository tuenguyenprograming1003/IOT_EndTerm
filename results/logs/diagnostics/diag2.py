import sys; sys.path.insert(0, '/Users/ductue/IOTCK/C3_LogMel_Compression/src')
import numpy as np, pandas as pd, torch
from torch import nn
from audio import load_fixed
from features import mel_power, fit_normalization, normalize
from classifier import DigitCNN, predict
from utils import ROOT, SPLITS_DIR, set_seed
variant = sys.argv[1]; AUG = 'aug' in variant; CMVN = 'cmvn' in variant; base = variant.replace('_aug','').replace('_cmvn',''); torch.set_num_threads(2)
m = pd.read_csv(ROOT/'data/manifest.csv')
def feat(y):
    if base == 'peak_log1p':
        y = y / max(np.abs(y).max(), 1e-8)
    P = mel_power(y)
    if base in ('spec_log1p', 'peak_log1p'): return np.log1p(P)
    if base == 'log_eps': return np.log(P + 1e-6)
    if base == 'peak_log_eps': return np.log(mel_power(y / max(np.abs(y).max(),1e-8)) + 1e-6)
S = np.stack([feat(load_fixed(ROOT/p)) for p in m.path]).astype(np.float32); idx = {f:i for i,f in enumerate(m.filename)}
if CMVN: S = (S - S.mean(axis=(1,2), keepdims=True)) / (S.std(axis=(1,2), keepdims=True)+1e-6)
accs=[]
for fold in range(1,7):
    set_seed(0)
    sp = {k: pd.read_csv(SPLITS_DIR/f'fold_{fold}_{k}.csv') for k in ('train','val','test')}
    X = {k: S[[idx[f] for f in v.filename]] for k,v in sp.items()}; Y = {k: v.label.to_numpy() for k,v in sp.items()}
    mu, sg = fit_normalization(X['train']); X = {k: normalize(v, mu, sg) for k,v in X.items()}
    model = DigitCNN(); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(X['train']).unsqueeze(1); yt = torch.from_numpy(Y['train'])
    best=(-1,None)
    for ep in range(40):
        model.train(); perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 64):
            b = perm[i:i+64]; xb = xt[b]
            if AUG: xb = torch.stack([torch.roll(x, int(torch.randint(-6,7,(1,))), dims=-1) for x in xb]) + 0.1*torch.randn(len(b),1,1,1)
            loss = nn.functional.cross_entropy(model(xb), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); va = (predict(model, X['val']).numpy()==Y['val']).mean()
        if va > best[0]: best = (va, {k:v.clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); te = (predict(model, X['test']).numpy()==Y['test']).mean(); accs.append(te)
print(variant, 'test acc per fold', np.round(accs,3), 'mean', round(float(np.mean(accs)),4), flush=True)
