from numba import njit
import numpy as np


# diffusion: starting point is on the feature (words) space
# provides ranking on features (X) and documents (Y)
@njit
def jit_diffusion(x_pointers, x_indices, x_data,
                  y_pointers, y_indices, y_data,
                  z_indices, z_data,
                  x_relevance, y_relevance,
                  alpha, n_iter, offset,
                  x_fluid, y_fluid):
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
                x_fluid[y_indices[y_pointers[j]:y_pointers[j + 1]]] += f * alpha * y_data[y_pointers[j]:y_pointers[j + 1]]
        for i in range(n):
            f = x_fluid[i]
            x_fluid[i] = 0.0
            if f > 0:
                x_relevance[i] += f
                y_fluid[x_indices[x_pointers[i]:x_pointers[i + 1]]] += f * alpha * x_data[x_pointers[i]:x_pointers[i + 1]]

    # Don't waste the last drop of fluid, it's free!
    for i in range(m):
        y_relevance[i] += y_fluid[i]


class DIteration:
    def __init__(self, n, m, alpha=.5, n_iter=4, offset=1.0, memory=0):
        self.x_relevance = np.zeros(n)
        self.y_relevance = np.zeros(m)
        self.x_order = None
        self.y_order = None
        self.alpha = alpha
        self.n_iter = n_iter
        self.offset = offset
        self.memory = memory
        self._x_fluid = np.zeros(n)
        self._y_fluid = np.zeros(m)

    def __call__(self, x, y, z):
        self.x_relevance[:] *= self.memory
        self.y_relevance[:] *= self.memory
        jit_diffusion(x_pointers=x.indptr, x_indices=x.indices, x_data=x.data,
                      y_pointers=y.indptr, y_indices=y.indices, y_data=y.data,
                      z_indices=z.indices, z_data=z.data,
                      x_relevance=self.x_relevance, y_relevance=self.y_relevance,
                      alpha=self.alpha, n_iter=self.n_iter, offset=self.offset,
                      x_fluid=self._x_fluid, y_fluid=self._y_fluid)
        self.x_order = np.argsort(-self.x_relevance)
        self.y_order = np.argsort(-self.y_relevance)