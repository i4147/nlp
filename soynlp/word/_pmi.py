import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from sklearn.metrics import pairwise_distances
from soynlp.utils import get_process_memory
from soynlp.vectorizer import sent_to_word_contexts_matrix


def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag


def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    n, m = exp_pmi.shape

    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    indices = np.where(data >= min_exp_pmi)[0]
    rows = rows[indices]
    cols = cols[indices]
    data = data[indices]

    data = np.log(data)

    exp_pmi_ = csr_matrix((data, (rows, cols)), shape=(n, m))
    return exp_pmi_


def pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):

    assert 0 < beta <= 1

    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py**beta
        py /= py.sum()
    pxy = X / X.sum()

    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py


def pmi_memory_friendly(X, py=None, min_pmi=0, alpha=0.0, beta=1.0, verbose=False):

    assert 0 < beta <= 1

    px = (X.sum(axis=1) / X.sum()).reshape(-1)
    if py is None:
        py = (X.sum(axis=0) / X.sum()).reshape(-1)
    pxy = X / X.sum()

    assert py.shape[0] == pxy.shape[1]

    if beta < 1:
        py = py**beta
        py /= py.sum()

    px_diag = diags(px.tolist()[0])
    py_diag = diags((py).tolist()[0])

    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / v for v in px_diag.data[0]])
    py_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in py_diag.data[0]])

    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)

    indices = np.where(exp_pmi.data > min_exp_pmi)[0]

    pmi_dok = dok_matrix(exp_pmi.shape)

    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    for _n_idx, idx in enumerate(indices):
        if verbose and _n_idx % 10000 == 0:
            print(
                "\rcomputing pmi {:.3} %  mem={} Gb    ".format(
                    100 * _n_idx / indices.shape[0], "%.3f" % get_process_memory()
                ),
                flush=True,
                end="",
            )
        pmi_dok[rows[idx], cols[idx]] = np.log(data[idx])
    if verbose:
        print("\rcomputing pmi was done{}".format(" " * 30), flush=True)

    return pmi_dok, px, py
