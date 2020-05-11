import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack, csr_matrix

from gismo.corpus import Corpus, toy_source_text
from gismo.embedding import Embedding


class Cluster:
    """
    The 'Cluster' class is used for internal representation of hierarchical cluster. It stores
    the attributes that describe a clustering structure and provides cluster addition (e.g. merge).
    Parameters
    ----------
    indice: int
        Index of the head (main element) of the cluster.
    rank: int
        The ranking order of a cluster.
    vector: csr_matrix
        The vector representation of the cluster

    Attributes
    ----------
    indice: int
        Index of the head (main element) of the cluster.
    rank: int
        The ranking order of a cluster.
    vector: csr_matrix
        The vector representation of the cluster.
    intersection_vector: csr_matrix (deprecated)
        The vector representation of the common points of a cluster.
    members: list of int
        The indices of the cluster elements.
    focus: float in range [0.0, 1.0]
        The consistency of the cluster (higher focus means that elements are more similar).
    children: list of Cluster
        The subclusters.

    Examples
    ---------
    >>> c1 = Cluster(indice=0, rank=1, vector=csr_matrix([1.0, 0.0, 1.0]))
    >>> c2 = Cluster(indice=5, rank=0, vector=csr_matrix([1.0, 1.0, 0.0]))
    >>> c3 = c1+c2
    >>> c3.members
    [0, 5]
    >>> c3.indice
    5
    >>> c3.vector.toarray()
    array([[2., 1., 1.]])
    >>> c3.intersection_vector.toarray()
    array([[1., 0., 0.]])
    >>> c1 == sum([c1])
    True
    """

    def __init__(self, indice=None, rank=None, vector=None):
        self.indice = indice
        self.rank = rank
        self.members = [indice] if indice is not None else []
        self.focus = 1.0
        self.vector = vector
        self.intersection_vector = vector
        self.children = []

    def __add__(self, cluster):
        result = Cluster()
        if self.rank < cluster.rank:
            result.indice = self.indice
            result.rank = self.rank
        else:
            result.indice = cluster.indice
            result.rank = cluster.rank
        result.members = self.members + cluster.members
        result.focus = min(self.focus, cluster.focus)  # Don't forget external focus
        result.vector = self.vector + cluster.vector
        result.intersection_vector = self.intersection_vector.multiply(cluster.intersection_vector)
        result.children = []  # Better update sons list outside class definition
        return result

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


def merge_clusters(cluster_list, focus=1.0):
    if len(cluster_list) < 2:
        return cluster_list[0]
    result = sum(cluster_list)
    result.focus = min(result.focus, focus)
    result.children = sorted(cluster_list, key=lambda c: c.rank)
    return result


def subspace_partition(subspace, resolution=.7):
    # Resolution square distortion
    resolution = 2 * resolution - resolution ** 2
    n, _ = subspace.shape
    similarity_matrix = cosine_similarity(subspace, subspace) - 2 * np.identity(n)
    similarity = np.max(similarity_matrix, axis=0)
    closest = np.argmax(similarity_matrix, axis=0)
    local_similarity = [similarity[closest[i]] for i in range(n)]
    partition = [{i} for i in range(n)]
    heads = np.arange(n)
    cluster_similarity = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sij = similarity_matrix[i, j]
            master = min(heads[i], heads[j])
            slave = max(heads[i], heads[j])
            if sij >= resolution * local_similarity[i]:
                cluster_similarity[master] = min(cluster_similarity[master], sij)
                if master != slave:
                    for k in partition[slave]:
                        heads[k] = master
                    partition[master] |= partition[slave]
                    partition[slave] = set()
    return [(c, d) for c, d in zip(partition, cluster_similarity) if c]


def rec_clusterize(cluster_list, resolution):
    if len(cluster_list) == 1:
        return cluster_list[0]
    else:
        partition = subspace_partition(vstack([c.vector for c in cluster_list]), resolution)
        return rec_clusterize([merge_clusters([cluster_list[i] for i in p[0]], p[1]) for p in partition],
                              resolution)


def subspace_clusterize(subspace, resolution=.7, indices=None):
    """
    Converts a subspace (matrix seen as a list of vectors) to a Cluster object (hierarchical clustering).

    Parameters
    ----------
    subspace: ndarray, csr_matrix
        A ``k x m`` matrix seen as a list of ``k`` ``m``-dimensional vectors sorted by importance order.
    resolution: float in range [0.0, 1.0]
        Sets the lazyness of aggregation. A 'resolution' set to 0.0 yields a one-step clustering
        (*star* structure), while a 'resolution ' set to 1.0 yields, up to tie similarities, a binary tree
        (*dendrogram*).
    indices: list, optional
        Indicates the index for each element of the subspace. Used when 'subspace'
        is extracted from a larger space (e.g. X or Y). If not set, indices are set to ``range(k)``.

    Returns
    -------
    Cluster
        A cluster whose leaves are the `k` vectors from 'subspace'.

    Example
    _________
    >>> corpus = Corpus(toy_source_text)
    >>> vectorizer = CountVectorizer(dtype=float)
    >>> embedding = Embedding(vectorizer=vectorizer)
    >>> embedding.fit_transform(corpus)
    >>> subspace = embedding.x[1:, :]
    >>> cluster = subspace_clusterize(subspace)
    >>> len(cluster.children)
    2
    >>> cluster = subspace_clusterize(subspace, resolution=.02)
    >>> len(cluster.children)
    4
    """
    if indices is None:
        n, _ = subspace.shape
        indices = range(n)
    return rec_clusterize([Cluster(indice=r, rank=i, vector=subspace[i, :]) for i, r in enumerate(indices)],
                          resolution)


class BFS:
    def __init__(self):
        self.result = []
        self.heap = []
        self.used = set()

    def set(self, cluster):
        self.result = []
        self.heap = []
        self.used = set()
        self.push(cluster)

    def push(self, cluster):
        heapq.heappush(self.heap, (cluster.focus, cluster.rank, cluster))

    def update(self, item):
        if item not in self.used:
            self.used.add(item)
            self.result.append(item)

    def pop(self):
        return heapq.heappop(self.heap)[2]

    def core(self, cluster):
        self.set(cluster)
        while len(self.heap) > 0:
            c = self.pop()
            self.update(c.indice)
            for child in c.children:
                self.push(child)
        return self.result

    def wide(self, cluster):
        self.set(cluster)
        while len(self.heap) > 0:
            c = self.pop()
            for child in c.children:
                self.push(child)
                self.update(child.indice)
        return self.result


def bfs(cluster, wide=True):
    if wide:
        return BFS().wide(cluster)
    else:
        return BFS().core(cluster)
