"""Main module."""

from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial

from gismo.common import MixInIO, toy_source_dict
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.diteration import DIteration
from gismo.clustering import subspace_clusterize
from gismo.post_processing import post_document, post_document_content, post_document_cluster, \
    post_feature, post_feature_cluster, print_document_cluster, print_feature_cluster


class Gismo(MixInIO):
    """
    Example
    -------
    >>> corpus = Corpus(toy_source_dict, lambda x: x['content'])
    >>> vectorizer = CountVectorizer(dtype=float)
    >>> embedding = Embedding(vectorizer=vectorizer)
    >>> embedding.fit_transform(corpus)
    >>> gismo = Gismo(corpus, embedding)
    >>> gismo.embedding.m
    35
    >>> gismo.post_document = partial(post_document_content, max_size=42)
    >>> gismo.diteration.alpha = .7
    >>> gismo.rank("Gizmo")
    >>> gismo.get_ranked_documents(3)
    ['Gizmo is a Mogwaï.', 'This very long sentence, with a lot of stu', 'In chinese folklore, a Mogwaï is a demon.']
    >>> gismo.diteration.alpha = .8
    >>> gismo.rank("Gizmo")
    >>> gismo.get_ranked_documents(3)
    ['Gizmo is a Mogwaï.', 'In chinese folklore, a Mogwaï is a demon.', 'This very long sentence, with a lot of stu']
    >>> gismo.post_document_cluster = print_document_cluster
    >>> gismo.get_clustered_ranked_documents() # doctest: +NORMALIZE_WHITESPACE
    F: 0.05. R: 1.85. S: 0.99.
    - F: 0.68. R: 1.76. S: 0.98.
    -- Gizmo is a Mogwaï. (R: 1.25; S: 0.98)
    -- In chinese folklore, a Mogwaï is a demon. (R: 0.28; S: 0.72)
    -- This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to
    the Gremlins movie by comparing Gizmo and Yoda. (R: 0.24; S: 0.67)
    - F: 0.71. R: 0.09. S: 0.19.
    -- This is a sentence about Shadoks. (R: 0.04; S: 0.17)
    -- This is a sentence about Blade. (R: 0.04; S: 0.17)
    >>> gismo.post_feature_cluster = print_feature_cluster
    >>> gismo.get_clustered_ranked_features() # doctest: +NORMALIZE_WHITESPACE
     F: 0.01. R: 1.27. S: 0.94.
    - F: 0.08. R: 1.23. S: 0.94.
    -- F: 0.99. R: 1.05. S: 0.97.
    --- mogwaï (R: 0.47; S: 0.98)
    --- gizmo (R: 0.45; S: 0.96)
    --- is (R: 0.13; S: 0.98)
    -- F: 1.00. R: 0.18. S: 0.21.
    --- chinese (R: 0.05; S: 0.21)
    --- demon (R: 0.05; S: 0.21)
    --- folklore (R: 0.05; S: 0.21)
    --- in (R: 0.05; S: 0.21)
    - F: 0.62. R: 0.04. S: 0.08.
    -- shadoks (R: 0.01; S: 0.03)
    -- blade (R: 0.01; S: 0.03)
    -- this (R: 0.01; S: 0.13)
    """

    def __init__(self, corpus, embedding):
        self.corpus = corpus
        self.embedding = embedding
        self.diteration = DIteration(n=embedding.n, m=embedding.m)

        self.post_document = post_document
        self.post_feature = post_feature
        self.post_document_cluster = post_document_cluster
        self.post_feature_cluster = post_feature_cluster

    # Ranking Part
    def rank(self, query=""):
        """
        Runs the Diteration using query as starting point

        Parameters
        ----------
        query: str
               Text that starts DIteration
        """
        z = self.embedding.query_projection(query)
        self.diteration(self.embedding.x, self.embedding.y, z)

    def get_ranked_documents(self, k=10):
        return [self.post_document(self, i) for i in self.diteration.x_order[:k]]

    def get_ranked_features(self, k=10):
        return [self.post_feature(self, i) for i in self.diteration.y_order[:k]]

    # Cluster part
    def get_clustered_documents(self, indices, resolution=.9):
        subspace = csr_matrix(vstack([self.embedding.x[i, :].multiply(self.diteration.y_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        return self.post_document_cluster(self, cluster)

    def get_clustered_ranked_documents(self, k=10, resolution=.9):
        return self.get_clustered_documents(self.diteration.x_order[:k], resolution)

    def get_clustered_features(self, indices, resolution=.9):
        subspace = csr_matrix(vstack([self.embedding.y[i, :].multiply(self.diteration.x_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        return self.post_feature_cluster(self, cluster)

    def get_clustered_ranked_features(self, k=10, resolution=.9):
        return self.get_clustered_features(self.diteration.y_order[:k], resolution)

