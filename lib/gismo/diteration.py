from numba import njit
import numpy as np

from gismo.parameters import ALPHA, N_ITER, OFFSET, MEMORY


# diffusion: starting point is on the feature (words) space
# provides ranking on features (X) and documents (Y)
@njit
def jit_diffusion(x_pointers, x_indices, x_data,
                  y_pointers, y_indices, y_data,
                  z_indices, z_data,
                  x_relevance, y_relevance,
                  alpha, n_iter, offset: float,
                  x_fluid, y_fluid):
    """
    Core diffusion engine written to be compatible with `Numba <https://numba.pydata.org/>`_.
    This is where the `DIteration <https://arxiv.org/pdf/1501.06350.pdf>`_
    algorithm is applied inline.

    Parameters
    ----------
    x_pointers: :class:`~numpy.ndarray`
        Pointers of the :class:`~scipy.sparse.csr_matrix` embedding of documents.
    x_indices: :class:`~numpy.ndarray`
        Indices of the :class:`~scipy.sparse.csr_matrix` embedding of documents.
    x_data: :class:`~numpy.ndarray`
        Data of the :class:`~scipy.sparse.csr_matrix` embedding of documents.
    y_pointers: :class:`~numpy.ndarray`
        Pointers of the :class:`~scipy.sparse.csr_matrix` embedding of features.
    y_indices: :class:`~numpy.ndarray`
        Indices of the :class:`~scipy.sparse.csr_matrix` embedding of features.
    y_data: :class:`~numpy.ndarray`
        Data of the :class:`~scipy.sparse.csr_matrix` embedding of features.
    z_indices: :class:`~numpy.ndarray`
        Indices of the :class:`~scipy.sparse.csr_matrix` embedding of the query projection.
    z_data: :class:`~numpy.ndarray`
        Data of the :class:`~scipy.sparse.csr_matrix` embedding of the query_projection.
    x_relevance: :class:`~numpy.ndarray`
        Placeholder for relevance of documents.
    y_relevance: :class:`~numpy.ndarray`
        Placeholder for relevance of features.
    alpha: float in range [0.0, 1.0]
        Damping factor. Controls the trade-off between closeness and centrality.
    n_iter: int
        Number of round-trip diffusions to perform. Higher value means better precision
        but longer execution time.
    offset: float in range [0.0, 1.0]
        Controls how much of the initial fluid should be deduced form the relevance.
    x_fluid: :class:`~numpy.ndarray`
        Placeholder for fluid on the side of documents.
    y_fluid: :class:`~numpy.ndarray`
        Placeholder for fluid on the side of features.
    """
    n = len(x_pointers) - 1
    m = len(y_pointers) - 1

    # Reset fluids
    x_fluid[:] = 0
    y_fluid[:] = 0
    for ind, data in zip(z_indices, z_data):
        y_relevance[ind] -= data * offset  # First round penalty
        y_fluid[ind] = data

    # Core diffusion
    for turn in range(n_iter):
        for j in range(m):
            f = y_fluid[j]
            y_fluid[j] = 0.0
            if f > 0:
                y_relevance[j] += f
                x_fluid[y_indices[y_pointers[j]:y_pointers[j + 1]]] += f * alpha * y_data[
                                                                                   y_pointers[j]:y_pointers[j + 1]]
        for i in range(n):
            f = x_fluid[i]
            x_fluid[i] = 0.0
            if f > 0:
                x_relevance[i] += f
                y_fluid[x_indices[x_pointers[i]:x_pointers[i + 1]]] += f * alpha * x_data[
                                                                                   x_pointers[i]:x_pointers[i + 1]]

    # Don't waste the last drop of fluid, it's free!
    for i in range(m):
        y_relevance[i] += y_fluid[i]


class DIteration:
    """
    This class is in charge of performing the
    `DIteration <https://arxiv.org/pdf/1501.06350.pdf>`_
    algorithm.

    Parameters
    ----------
    n: int
        Number of documents.
    m: int
        Number of features.

    Attributes
    ----------
    x_relevance: :class:`~numpy.ndarray`
        Relevance of documents.
    y_relevance: :class:`~numpy.ndarray`
        Relevance of features.
    x_order: :class:`~numpy.ndarray`
        Indices of documents sorted by relevance.
    y_order: :class:`~numpy.ndarray`
        Indices of features sorted by relevance.
    """

    def __init__(self, n, m):
        self.x_relevance = np.zeros(n)
        self.y_relevance = np.zeros(m)
        self.x_order = None
        self.y_order = None
        self._x_fluid = np.zeros(n)
        self._y_fluid = np.zeros(m)

    def __call__(self, x, y, z,
                 alpha=ALPHA, n_iter=N_ITER, offset: float = OFFSET, memory=MEMORY):
        """
        Performs DIteration algorithm and populate relevance / order vectors.

        Parameters
        ----------
        x: :class:`~scipy.sparse.csr_matrix`
            Embedding of documents in feature space.
        y: :class:`~scipy.sparse.csr_matrix`
            Embedding of features in document space.
        z:  class:`~scipy.sparse.csr_matrix`
            Embedding of query in feature space.
        alpha: float in range [0.0, 1.0]
            Damping factor. Controls the trade-off between closeness and centrality.
        n_iter: int
            Number of round-trip diffusions to perform. Higher value means better precision
            but longer execution time.
        offset: float in range [0.0, 1.0]
            Controls how much of the initial fluid should be deduced form the relevance.
        memory: float in range [0.0, 1.0]
            Controls how much of previous computation is kept
            when performing a new diffusion.
            """
        self.x_relevance[:] *= memory
        self.y_relevance[:] *= memory
        jit_diffusion(x_pointers=x.indptr, x_indices=x.indices, x_data=x.data,
                      y_pointers=y.indptr, y_indices=y.indices, y_data=y.data,
                      z_indices=z.indices, z_data=z.data,
                      x_relevance=self.x_relevance, y_relevance=self.y_relevance,
                      alpha=alpha, n_iter=n_iter, offset=offset,
                      x_fluid=self._x_fluid, y_fluid=self._y_fluid)
        self.x_order = np.argsort(-self.x_relevance)
        self.y_order = np.argsort(-self.y_relevance)
