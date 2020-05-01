import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack


class Cluster:
    def __init__(self, indice=None, rank=None, vector=None):
        self.indice = indice
        self.rank = rank
        self.sim = None
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
    result.children = sorted(cluster_list, key = lambda c: c.rank)
    return result


def subspace_partition(subspace, resolution=.9):
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


def subspace_clusterize(subspace, resolution=.9, indices=None):
    if indices is None:
        n, _ = subspace.shape
        indices = range(n)
    return rec_clusterize([Cluster(indice=r, rank=i, vector=subspace[i, :]) for i, r in enumerate(indices)],
                          resolution)
