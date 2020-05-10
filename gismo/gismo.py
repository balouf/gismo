"""Main module."""

from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial

from gismo.common import MixInIO, toy_source_dict, auto_k
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

    The Corpus class defines how documents of a source should be converted to plain text.

    >>> corpus = Corpus(toy_source_dict, lambda x: x['content'])

    The Embedding class extracts features (e.g. words) and computes weights between documents and features.

    >>> vectorizer = CountVectorizer(dtype=float)
    >>> embedding = Embedding(vectorizer=vectorizer)
    >>> embedding.fit_transform(corpus)
    >>> embedding.m # number of features
    36

    The Gismo class combines them for performing queries.
    After a query is performed, one can ask for the best items.
    The number of items to return can be specified with parameter ``k`` or automatically adjusted.

    >>> gismo = Gismo(corpus, embedding)
    >>> success = gismo.rank("Gizmo")
    >>> gismo.auto_k_target = .2 # The toy dataset is very small, so we lower the auto_k parameter.
    >>> gismo.get_ranked_documents()
    [{'title': 'First Document', 'content': 'Gizmo is a Mogwaï.'}, {'title': 'Fourth Document', 'content': 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.'}, {'title': 'Fifth Document', 'content': 'In chinese folklore, a Mogwaï is a demon.'}]

    Post processing functions can be used to tweak the returned object (the underlying ranking is unchanged)

    >>> gismo.post_document = partial(post_document_content, max_size=42)
    >>> gismo.get_ranked_documents()
    ['Gizmo is a Mogwaï.', 'This very long sentence, with a lot of stu', 'In chinese folklore, a Mogwaï is a demon.']

    Ranking also works on features.

    >>> gismo.get_ranked_features()
    ['mogwaï', 'gizmo', 'is', 'in', 'demon', 'chinese', 'folklore']

    Clustering organizes results can provide additional hints on their relationships.

    >>> gismo.post_document_cluster = print_document_cluster
    >>> gismo.get_clustered_ranked_documents(k=5) # doctest: +NORMALIZE_WHITESPACE
     F: 0.05. R: 0.66. S: 0.99.
    - F: 0.70. R: 0.65. S: 0.98.
    -- Gizmo is a Mogwaï. (R: 0.54; S: 0.99)
    -- This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda. (R: 0.08; S: 0.69)
    -- In chinese folklore, a Mogwaï is a demon. (R: 0.04; S: 0.71)
    - F: 0.96. R: 0.01. S: 0.18.
    -- This is a sentence about Blade. (R: 0.01; S: 0.18)
    -- This is another sentence about Shadoks. (R: 0.01; S: 0.18)
    >>> gismo.post_feature_cluster = print_feature_cluster
    >>> gismo.get_clustered_ranked_features(k=10) # doctest: +NORMALIZE_WHITESPACE
     F: 0.01. R: 0.29. S: 0.99.
    - F: 0.03. R: 0.29. S: 0.98.
    -- F: 1.00. R: 0.27. S: 0.99.
    --- mogwaï (R: 0.12; S: 0.99)
    --- gizmo (R: 0.12; S: 0.99)
    --- is (R: 0.03; S: 0.99)
    -- F: 1.00. R: 0.02. S: 0.07.
    --- in (R: 0.00; S: 0.07)
    --- demon (R: 0.00; S: 0.07)
    --- chinese (R: 0.00; S: 0.07)
    --- folklore (R: 0.00; S: 0.07)
    - F: 1.00. R: 0.00. S: 0.15.
    -- star (R: 0.00; S: 0.15)
    -- the (R: 0.00; S: 0.15)
    -- of (R: 0.00; S: 0.15)
    """

    def __init__(self, corpus, embedding):
        self.corpus = corpus
        self.embedding = embedding
        self.diteration = DIteration(n=embedding.n, m=embedding.m)

        self.auto_k_max_k = 100
        self.auto_k_target = 1.0

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

        Returns
        -------
        success: bool
            success of the query projection. If projection fails, a default ranking on uniform distribution is performed.
        """
        z, success = self.embedding.query_projection(query)
        self.diteration(self.embedding.x, self.embedding.y, z)
        return success

    def get_ranked_documents(self, k=None):
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return [self.post_document(self, i) for i in self.diteration.x_order[:k]]

    def get_ranked_features(self, k=None):
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return [self.post_feature(self, i) for i in self.diteration.y_order[:k]]

    # Cluster part
    def get_clustered_documents(self, indices, resolution=.9):
        subspace = csr_matrix(vstack([self.embedding.x[i, :].multiply(self.diteration.y_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        return self.post_document_cluster(self, cluster)

    def get_clustered_ranked_documents(self, k=None, resolution=.9):
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return self.get_clustered_documents(self.diteration.x_order[:k], resolution)

    def get_clustered_features(self, indices, resolution=.9):
        subspace = csr_matrix(vstack([self.embedding.y[i, :].multiply(self.diteration.x_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        return self.post_feature_cluster(self, cluster)

    def get_clustered_ranked_features(self, k=None, resolution=.9):
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return self.get_clustered_features(self.diteration.y_order[:k], resolution)

