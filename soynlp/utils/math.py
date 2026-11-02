from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd


def svd(X, n_components, n_iter=5, random_state=None):

    if (random_state == None) or isinstance(random_state, int):
        random_state = check_random_state(random_state)

    n_features = X.shape[1]

    if n_components >= n_features:
        raise ValueError("n_components must be < n_features; got %d >= %d" % (n_components, n_features))

    U, Sigma, VT = randomized_svd(X, n_components, n_iter=n_iter, random_state=random_state)

    return U, Sigma, VT
