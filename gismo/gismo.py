"""Main module."""

from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial

from gismo.common import MixInIO, toy_source_dict, auto_k
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.diteration import DIteration
from gismo.clustering import subspace_clusterize, bfs
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
    >>> gismo.get_clustered_ranked_documents(resolution=.9) # doctest: +NORMALIZE_WHITESPACE
     F: 0.60. R: 0.65. S: 0.98.
    - F: 0.71. R: 0.57. S: 0.98.
    -- Gizmo is a Mogwaï. (R: 0.54; S: 0.99)
    -- In chinese folklore, a Mogwaï is a demon. (R: 0.04; S: 0.71)
    - This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda. (R: 0.08; S: 0.69)
    >>> gismo.post_feature_cluster = print_feature_cluster
    >>> gismo.get_clustered_ranked_features() # doctest: +NORMALIZE_WHITESPACE
     F: 0.03. R: 0.29. S: 0.98.
    - F: 1.00. R: 0.27. S: 0.99.
    -- mogwaï (R: 0.12; S: 0.99)
    -- gizmo (R: 0.12; S: 0.99)
    -- is (R: 0.03; S: 0.99)
    - F: 1.00. R: 0.02. S: 0.07.
    -- in (R: 0.00; S: 0.07)
    -- demon (R: 0.00; S: 0.07)
    -- chinese (R: 0.00; S: 0.07)
    -- folklore (R: 0.00; S: 0.07)

    The class also offers ``get_covering_documents`` and ``get_covering_features`` that yield
    a list of results obtained from a BFS-like traversal of the ranked cluster.

    To demonstrate it, we first add an outsider document to the corpus and rebuild Gismo.

    >>> new_entry = {'title': 'Minority Report', 'content': 'Totally unrelated stuff.'}
    >>> corpus = Corpus(toy_source_dict+[new_entry], lambda x: x['content'])
    >>> vectorizer = CountVectorizer(dtype=float)
    >>> embedding = Embedding(vectorizer=vectorizer)
    >>> embedding.fit_transform(corpus)
    >>> gismo = Gismo(corpus, embedding)
    >>> gismo.post_document = post_document_content
    >>> success = gismo.rank("Gizmo")
    >>> gismo.auto_k_target=.3

    Remind the classical rank-based result.

    >>> gismo.get_ranked_documents()
    ['Gizmo is a Mogwaï.', 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.', 'In chinese folklore, a Mogwaï is a demon.']

    Gismo can use the cluster to propose alternate results that try to cover more subjects.

    >>> gismo.get_covering_documents()
    ['Gizmo is a Mogwaï.', 'Totally unrelated stuff.', 'This is a sentence about Blade.']

    Note how the new entry, which has nothing to do with the rest, is pushed into the results.
    By setting the ``wide`` option to False, we get an alternative that focuses on mainstream results.

    >>> gismo.get_covering_documents(wide=False)
    ['Gizmo is a Mogwaï.', 'This is a sentence about Blade.', 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.']

    The same principle applies for features.

    >>> gismo.get_ranked_features()
    ['mogwaï', 'gizmo', 'is', 'in', 'chinese', 'folklore', 'demon']

    >>> gismo.get_covering_features()
    ['mogwaï', 'this', 'in', 'by', 'gizmo', 'is', 'chinese']
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
            success of the query projection. If projection fails, a ranking on uniform distribution is performed.
        """
        z, success = self.embedding.query_projection(query)
        self.diteration(self.embedding.x, self.embedding.y, z)
        return success

    def get_ranked_documents(self, k=None):
        """
        Returns a list of top documents according to the current ranking.
        The documents are post_processed through the post_document method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.

        Returns
        -------
        list
        """
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return [self.post_document(self, i) for i in self.diteration.x_order[:k]]

    def get_ranked_features(self, k=None):
        """
        Returns a list of top features according to the current ranking.
        The features are post_processed through the post_feature method.

        Parameters
        ----------
        k: int, optional
            Number of features to output. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.

        Returns
        -------
        list
        """
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return [self.post_feature(self, i) for i in self.diteration.y_order[:k]]

    # Cluster part
    def get_clustered_documents(self, indices, resolution=.7, post=True):
        """
        Returns a cluster of documents.
        The cluster is by default post_processed through the post_document_cluster method.

        Parameters
        ----------
        indices: list of int
            The indices of documents to be processed. It is assumed that the documents
            are sorted by importance.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        post: bool
            Set to False to disable post-processing and return a raw Cluster object.

        Returns
        -------
        Object

        """
        subspace = csr_matrix(vstack([self.embedding.x[i, :].multiply(self.diteration.y_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        if post:
            return self.post_document_cluster(self, cluster)
        return cluster

    def get_clustered_ranked_documents(self, k=None, resolution=.7, post=True):
        """
        Returns a cluster of the best ranked documents.
        The cluster is by default post_processed through the post_document_cluster method.

        Parameters
        ----------
        k: int, optional
            Number of top documents to use. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        post: bool
            Set to False to disable post-processing and return a raw Cluster object.

        Returns
        -------
        Object

        """
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return self.get_clustered_documents(self.diteration.x_order[:k], resolution, post)

    def get_clustered_features(self, indices, resolution=.7, post=True):
        """
        Returns a cluster of features.
        The cluster is by default post_processed through the post_feature_cluster method.

        Parameters
        ----------
        indices: list of int
            The indices of features to be processed. It is assumed that the features
            are sorted by importance.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        post: bool
            Set to False to disable post-processing and return a raw Cluster object.

        Returns
        -------
        Object

        """
        subspace = csr_matrix(vstack([self.embedding.y[i, :].multiply(self.diteration.x_relevance) for i in indices]))
        cluster = subspace_clusterize(subspace, resolution, indices)
        if post:
            return self.post_feature_cluster(self, cluster)
        return cluster

    def get_clustered_ranked_features(self, k=None, resolution=.7, post=True):
        """
        Returns a cluster of the best ranked features.
        The cluster is by default post_processed through the post_feature_cluster method.

        Parameters
        ----------
        k: int, optional
            Number of top features to use. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        post: bool
            Set to False to disable post-processing and return a raw Cluster object.

        Returns
        -------
        Object
        """
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        return self.get_clustered_features(self.diteration.y_order[:k], resolution, post)

    # Covering part

    def get_covering_documents(self, k=None, resolution=.7, strech=2.0, wide=True):
        """
        Returns a list of top covering documents.
        The documents are post_processed through the post_document method.

        Parameters
        ----------
        k: int, optional
            Number of documents to return. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        strech: float >= 1.0
            The list if build by traversing a cluster of size ``int(k*stretch)``.
            High stretch will more easily eliminate redundant results, but increases
            the odds of returning less relevant entries.
        wide: bool
            If True, uses a traversal that gives an advantage to results not similar to others.

        Returns
        -------
        list

        """
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        cluster = self.get_clustered_ranked_documents(k=int(k * strech),
                                                      resolution=resolution,
                                                      post=False)
        indices = bfs(cluster, wide=wide)[:k]
        return [self.post_document(self, i) for i in indices]

    def get_covering_features(self, k=None, resolution=.7, strech=2.0, wide=True):
        """
        Returns a list of top covering features.
        The features are post_processed through the post_feature method.

        Parameters
        ----------
        k: int, optional
            Number of features to return. If not set, k is automatically computed
            using the auto_k_max_k and auto_k_target attributes.
        resolution: float in range [0.0, 1.0]
            Depth of the cluster. Use 0.0 to a star, 1.0 for a pseudo-dendrogram.
        strech: float >= 1.0
            The list if build by traversing a cluster of size ``int(k*stretch)``.
            High stretch will more easily eliminate redundant results, but increases
            the odds of returning less relevant entries.
        wide: bool
            If True, uses a traversal that gives an advantage to results not similar to others.

        Returns
        -------
        list

        """
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=self.auto_k_max_k,
                       target=self.auto_k_target)
        cluster = self.get_clustered_ranked_features(k=int(k * strech),
                                                     resolution=resolution,
                                                     post=False)
        indices = bfs(cluster, wide=wide)[:k]
        return [self.post_feature(self, i) for i in indices]
