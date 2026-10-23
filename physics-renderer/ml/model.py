"""Small NumPy conditional density estimator; predictor has no label argument."""
import numpy as np
from scipy.special import expit, logit
from scipy.linalg import eigh

ROWS, COLS = np.tril_indices(4)
DIAG = ROWS == COLS
LOGIT_SCALE = np.pi / np.sqrt(3.)


class Support:
    def __init__(self, config):
        names = ('em0_pc_cm6', 'te_k', 'major_fwhm_arcsec', 'axis_ratio')
        ranges = config['sampling']['ranges']
        self.lower = np.array([ranges[n][0] for n in names], float)
        self.upper = np.array([ranges[n][1] for n in names], float)
        self.lower[[0, 2]] = np.log10(self.lower[[0, 2]])
        self.upper[[0, 2]] = np.log10(self.upper[[0, 2]])

    def encode(self, original):
        x = np.array(original, float, copy=True)
        x[..., [0, 2]] = np.log10(x[..., [0, 2]])
        u = (x-self.lower)/(self.upper-self.lower)
        if np.any((u <= 0) | (u >= 1)):
            raise ValueError('Targets must lie strictly inside declared base prior support')
        return logit(u)/LOGIT_SCALE

    def decode(self, z):
        x = self.lower + (self.upper-self.lower)*expit(np.asarray(z)*LOGIT_SCALE)
        x[..., [0, 2]] = 10.**x[..., [0, 2]]
        return x


def fit_pca(x, components):
    """Deterministic train-row Gram eigendecomposition, no labels or holdouts."""
    x = np.asarray(x, float)
    mean = x.mean(axis=0)
    centered = x-mean
    vals, vectors = eigh(centered @ centered.T, subset_by_index=[len(x)-components, len(x)-1])
    vals, vectors = vals[::-1], vectors[:, ::-1]
    basis = centered.T @ vectors / np.sqrt(np.maximum(vals, 1e-12))
    for j in range(components):
        if basis[np.argmax(abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    scale = np.maximum(np.sqrt(vals/len(x)), .1)
    return {'mean': mean, 'basis': basis, 'scale': scale,
            'explained_fraction': np.array(vals.sum()/np.sum(centered**2))}


def project(x, pca):
    return ((x-pca['mean']) @ pca['basis'])/pca['scale']


def initialize(d, hidden, rng):
    p = {'w1': rng.normal(0, 1/np.sqrt(d), (d, hidden)), 'b1': np.zeros(hidden),
         'w2': rng.normal(0, .01, (hidden, 14)), 'b2': np.zeros(14)}
    p['b2'][4:][DIAG] = np.log(np.expm1(.95))
    return p


def predict(x, p):
    h = np.tanh(x @ p['w1']+p['b1'])
    raw = h @ p['w2']+p['b2']
    a = np.zeros((len(x), 4, 4))
    a[:, ROWS, COLS] = raw[:, 4:]
    diagonal = np.logaddexp(0, raw[:, 4:][:, DIAG])+.05
    a[:, np.arange(4), np.arange(4)] = diagonal
    return raw[:, :4], a, (h, raw)


def loss_grad(x, y, p, decay=0., gradient=True):
    mu, a, (h, raw) = predict(x, p)
    e = y-mu
    v = np.einsum('bij,bj->bi', a, e)
    diag = np.diagonal(a, axis1=1, axis2=2)
    nll = .5*np.sum(v*v, axis=1)-np.log(diag).sum(axis=1)+2*np.log(2*np.pi)
    reg = .5*decay*(np.sum(p['w1']**2)+np.sum(p['w2']**2))
    if not gradient:
        return nll
    dmu = -np.einsum('bji,bj->bi', a, v)
    da = v[:, :, None]*e[:, None, :]
    da[:, np.arange(4), np.arange(4)] -= 1/diag
    draw = da[:, ROWS, COLS]
    draw[:, DIAG] *= expit(raw[:, 4:][:, DIAG])
    dout = np.concatenate((dmu, draw), axis=1)/len(x)
    dh = (dout @ p['w2'].T)*(1-h*h)
    grads = {'w2': h.T @ dout+decay*p['w2'], 'b2': dout.sum(axis=0),
             'w1': x.T @ dh+decay*p['w1'], 'b1': dh.sum(axis=0)}
    return float(nll.mean()+reg), grads


def sample(x, p, n, rng):
    mu, a, _ = predict(x, p)
    noise = rng.normal(size=(len(x), 4, n))
    z = np.linalg.solve(a, noise).transpose(0, 2, 1)+mu[:, None, :]
    return z


def covariance(a):
    inverse = np.linalg.inv(a)
    return inverse @ inverse.transpose(0, 2, 1)
