import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

n = 2**18
seed = 20261018
train = 128
samples = 2048
alpha = .01
delta = .08
folder = Path(__file__).resolve().parent


def orthonormalize(p):
    f = np.empty((len(p), 4, 4))
    for m in range(4):
        v = np.broadcast_to(np.arange(4, dtype=float)**m, (len(p), 4)).copy()
        for j in range(m):
            v -= np.sum(p * v * f[:, j], axis=1)[:, None] * f[:, j]
        f[:, m] = v / np.sqrt(np.sum(p * v*v, axis=1))[:, None]
    return f


def prepare(p):
    cdf = np.cumsum(p, axis=1)[:, :3]
    order = np.argsort(cdf.ravel())
    widths = np.diff(np.r_[0., cdf.ravel()[order], 1.])
    return cdf, order, widths


def q2(x, p, cache):
    cdf, order, widths = cache
    w = 1 / p[np.arange(len(p))[None, :], x]
    start = np.sum(w * (x == 0), axis=1) - len(p)
    jumps = np.empty((*x.shape, 3))
    for a in range(3):
        jumps[:, :, a] = w * ((x == a+1).astype(float) - (x == a))
    jumps = jumps.reshape(len(x), -1)[:, order]
    np.cumsum(jumps, axis=1, out=jumps)
    jumps += start[:, None]
    integral = widths[0] * start**2
    integral += np.einsum('bi,i,bi->b', jumps, widths[1:], jumps)
    return .5 * (integral - np.sum(w-1, axis=1))


def sample(cdf, t, rng, batch):
    n = len(cdf)
    if t is None:
        u = rng.random((batch, n))
    else:
        leaves = 1 << int(t)
        labels = rng.integers(leaves, size=(batch, n))
        if leaves <= 4*n:
            u = np.take_along_axis(rng.random((batch, leaves)), labels, axis=1)
        else:
            u = np.empty((batch, n))
            for row in range(batch):
                occupied, inverse = np.unique(labels[row], return_inverse=True)
                u[row] = rng.random(len(occupied))[inverse]
    x = np.zeros((batch, n), dtype=np.uint8)
    for a in range(3):
        x += u >= cdf[None, :, a]
    return x


def scores(p, cache, t, rng, count):
    out = np.empty(count)
    for a in range(0, count, 8):
        b = min(a+8, count)
        out[a:b] = q2(sample(cache[0], t, rng, b-a), p, cache)
    return out


def threshold(x, y):
    z = np.sort(np.r_[x, y])
    gap = np.searchsorted(np.sort(y), z, side='right') / len(y)
    gap -= np.searchsorted(np.sort(x), z, side='right') / len(x)
    return float(z[np.argmax(gap)])


def upper(n, t):
    s = n * 2.**(-t)
    return min(1., 3*s, 1-.5*np.exp(-6*s) if s >= np.log(2)/6 else 1.)


def main():
    path = folder / 'marginals.npy'
    if path.exists():
        p = np.load(path)
        if p.shape != (n, 4):
            raise ValueError('marginals.npy has a different n')
    else:
        p = delta + (1-4*delta)*np.random.default_rng(seed).dirichlet(np.ones(4), size=n) # q2 uses the completeness identity sum_{m>=1} f_m(s)f_m(s') = delta_{s,s'}/p(s) - 1
        np.save(path, p)
    f = orthonormalize(p)
    np.testing.assert_allclose(np.einsum('ims,ins,is->imn', f, f, p),
                               np.broadcast_to(np.eye(4), (n, 4, 4)), atol=1e-11)
    cache = prepare(p)
    times = range(int(np.log2(n))+4)
    seeds = np.random.SeedSequence(seed+1).spawn(2*len(times)+3)
    draw = lambda t, s, m: scores(p, cache, t, np.random.default_rng(s), m)
    pi_train = draw(None, seeds[0], train)
    cut = [threshold(draw(t, seeds[t+1], train), pi_train) for t in times]
    print('Thresholds fixed. Running validation.')
    pi_valid = draw(None, seeds[len(times)+1], samples)
    rows = []
    tail = alpha / (2*len(times))
    for t, h in zip(times, cut):
        x = draw(t, seeds[len(times)+2+t], samples)
        a, b = int(np.sum(x > h)), int(np.sum(pi_valid > h))
        lo = beta.ppf(tail, a, samples-a+1) if a else 0.
        hi = beta.ppf(1-tail, b+1, samples-b) if b < samples else 1.
        lower = float(max(0., lo-hi))
        rows.append([t, lower, upper(n, t), h, a, b])
        print(f'{t:2d}  {lower:.4f}  {upper(n, t):.4f}')
    result = dict(n=n, seed=seed, train=train, samples=samples, alpha=alpha,
                  left=max(0, int(np.log2(n))-8), rows=rows)
    page = folder / 'index.html'
    html = page.read_text()
    start = '<script id="data" type="application/json">'
    before, rest = html.split(start, 1)
    _, after = rest.split('</script>', 1)
    payload = json.dumps({k: v for k, v in result.items() if k != 'rows'})[:-1]
    payload += ', "rows": [\n' + ',\n'.join(json.dumps(r) for r in rows) + '\n]}'
    page.write_text(before + start + '\n' + payload + '\n</script>' + after)


if __name__ == '__main__':
    main()
