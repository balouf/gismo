"""Main module."""
import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial

from gismo.common import MixInIO, toy_source_dict, auto_k
from gismo.datasets.dblp import url2source
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.diteration import DIteration
from gismo.parameters import Parameters
from gismo.clustering import subspace_clusterize, covering_order, subspace_distortion
from gismo.post_processing import post_documents_item_raw, post_documents_item_content, post_documents_cluster_json, \
    post_features_item_raw, post_features_cluster_json, post_documents_cluster_print, post_features_cluster_print


class Gismo(MixInIO):
    """
    Gismo mixes a corpus and its embedding to provide search and structure methods.

    Parameters
    ----------
    corpus: Corpus
        Defines the documents of the gismo.
    embedding: Embedding
        Defines the embedding of the gismo.
    filename: str, optional
                If set, will load gismo from file.
    path: str or Path, optional
        Directory where the gismo is to be loaded from.
    kwargs: dict
        Custom default runtime parameters.
        You just need to specify the parameters that differ from :obj:`~gismo.parameters.DEFAULT_PARAMETERS`.


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
    >>> gismo.parameters.target_k = .2 # The toy dataset is very small, so we lower the auto_k parameter.
    >>> gismo.get_documents_by_rank()
    [{'title': 'First Document', 'content': 'Gizmo is a Mogwaï.'}, {'title': 'Fourth Document', 'content': 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.'}, {'title': 'Fifth Document', 'content': 'In chinese folklore, a Mogwaï is a demon.'}]

    Post processing functions can be used to tweak the returned object (the underlying ranking is unchanged)

    >>> gismo.post_documents_item = partial(post_documents_item_content, max_size=44)
    >>> gismo.get_documents_by_rank()
    ['Gizmo is a Mogwaï.', 'This very long sentence, with a lot of stuff', 'In chinese folklore, a Mogwaï is a demon.']

    Ranking also works on features.

    >>> gismo.get_features_by_rank()
    ['mogwaï', 'gizmo', 'is', 'in', 'demon', 'chinese', 'folklore']

    Clustering organizes results can provide additional hints on their relationships.

    >>> gismo.post_documents_cluster = post_documents_cluster_print
    >>> gismo.get_documents_by_cluster(resolution=.9) # doctest: +NORMALIZE_WHITESPACE
     F: 0.60. R: 0.65. S: 0.98.
    - F: 0.71. R: 0.57. S: 0.98.
    -- Gizmo is a Mogwaï. (R: 0.54; S: 0.99)
    -- In chinese folklore, a Mogwaï is a demon. (R: 0.04; S: 0.71)
    - This very long sentence, with a lot of stuff (R: 0.08; S: 0.69)
    >>> gismo.post_features_cluster = post_features_cluster_print
    >>> gismo.get_features_by_cluster() # doctest: +NORMALIZE_WHITESPACE
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

    The class also offers :meth:`~gismo.gismo.Gismo.get_documents_by_coverage` and
    :meth:`~gismo.gismo.Gismo.get_features_by_coverage` that yield
    a list of results obtained from a Covering-like traversal of the ranked cluster.

    To demonstrate it, we first add an outsider document to the corpus and rebuild Gismo.

    >>> new_entry = {'title': 'Minority Report', 'content': 'Totally unrelated stuff.'}
    >>> corpus = Corpus(toy_source_dict+[new_entry], lambda x: x['content'])
    >>> vectorizer = CountVectorizer(dtype=float)
    >>> embedding = Embedding(vectorizer=vectorizer)
    >>> embedding.fit_transform(corpus)
    >>> gismo = Gismo(corpus, embedding)
    >>> gismo.post_documents_item = post_documents_item_content
    >>> success = gismo.rank("Gizmo")
    >>> gismo.parameters.target_k = .3

    Remind the classical rank-based result.

    >>> gismo.get_documents_by_rank()
    ['Gizmo is a Mogwaï.', 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.', 'In chinese folklore, a Mogwaï is a demon.']

    Gismo can use the cluster to propose alternate results that try to cover more subjects.

    >>> gismo.get_documents_by_coverage()
    ['Gizmo is a Mogwaï.', 'Totally unrelated stuff.', 'This is a sentence about Blade.']

    Note how the new entry, which has nothing to do with the rest, is pushed into the results.
    By setting the ``wide`` option to False, we get an alternative that focuses on mainstream results.

    >>> gismo.get_documents_by_coverage(wide=False)
    ['Gizmo is a Mogwaï.', 'This is a sentence about Blade.', 'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.']

    The same principle applies for features.

    >>> gismo.get_features_by_rank()
    ['mogwaï', 'gizmo', 'is', 'in', 'chinese', 'folklore', 'demon']

    >>> gismo.get_features_by_coverage()
    ['mogwaï', 'this', 'in', 'by', 'gizmo', 'is', 'chinese']
    """

    def __init__(self, corpus=None, embedding=None, filename=None, path=".", **kwargs):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.corpus = corpus
            self.embedding = embedding
            self.diteration = DIteration(n=embedding.n, m=embedding.m)

            self.parameters = Parameters(**kwargs)

            self.post_documents_item = post_documents_item_raw
            self.post_features_item = post_features_item_raw
            self.post_documents_cluster = post_documents_cluster_json
            self.post_features_cluster = post_features_cluster_json

    # Ranking Part
    def rank(self, query="", **kwargs):
        """
        Runs the Diteration using query as starting point

        Parameters
        ----------
        query: str
               Text that starts DIteration
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        success: bool
            success of the query projection. If projection fails, a ranking on uniform distribution is performed.
        """
        p = self.parameters(**kwargs)
        z, success = self.embedding.query_projection(query)
        self.diteration(self.embedding.x, self.embedding.y, z,
                        alpha=p['alpha'], n_iter=p['n_iter'],
                        offset=p['offset'], memory=p['memory'])
        return success

    def get_documents_by_rank(self, k=None, **kwargs):
        """
        Returns a list of top documents according to the current ranking.
        By default, the documents are post_processed through the post_documents_item method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.


        Returns
        -------
        list
        """
        p = self.parameters(**kwargs)
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        if p['post']:
            return [self.post_documents_item(self, i) for i in self.diteration.x_order[:k]]
        else:
            return self.diteration.x_order[:k]

    def get_features_by_rank(self, k=None, **kwargs):
        """
        Returns a list of top features according to the current ranking.
        By default, the features are post_processed through the post_features_item method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        list
        """
        p = self.parameters(**kwargs)
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        if p['post']:
            return [self.post_features_item(self, i) for i in self.diteration.y_order[:k]]
        else:
            return self.diteration.y_order[:k]

    # Cluster part
    def get_documents_by_cluster_from_indices(self, indices, **kwargs):
        """
        Returns a cluster of documents.
        The cluster is by default post_processed through the post_documents_cluster method.

        Parameters
        ----------
        indices: list of int
            The indices of documents to be processed. It is assumed that the documents
            are sorted by importance.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        object

        """
        p = self.parameters(**kwargs)
        subspace = vstack([self.embedding.x[i, :] for i in indices])
        if p['distortion']>0:
            subspace_distortion(indices=subspace.indices, data=subspace.data,
                                relevance=self.diteration.y_relevance, distortion=p['distortion'])
        cluster = subspace_clusterize(subspace, p['resolution'], indices)
        if p['post']:
            return self.post_documents_cluster(self, cluster)
        return cluster

    def get_documents_by_cluster(self, k=None, **kwargs):
        """
        Returns a cluster of the best ranked documents.
        The cluster is by default post_processed through the post_documents_cluster method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        object

        """
        p = self.parameters(**kwargs)
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        return self.get_documents_by_cluster_from_indices(self.diteration.x_order[:k], **kwargs)

    def get_features_by_cluster_from_indices(self, indices, **kwargs):
        """
        Returns a cluster of features.
        The cluster is by default post_processed through the post_features_cluster method.

        Parameters
        ----------
        indices: list of int
            The indices of features to be processed. It is assumed that the features
            are sorted by importance.
        kwargs: dict, optional
            Custom runtime parameters

        Returns
        -------
        object

        """
        p = self.parameters(**kwargs)
        subspace = vstack([self.embedding.y[i, :] for i in indices])
        if p['distortion']>0:
            subspace_distortion(indices=subspace.indices, data=subspace.data,
                                relevance=self.diteration.x_relevance, distortion=p['distortion'])
        cluster = subspace_clusterize(subspace, p['resolution'], indices)
        if p['post']:
            return self.post_features_cluster(self, cluster)
        return cluster

    def get_features_by_cluster(self, k=None, **kwargs):
        """
        Returns a cluster of the best ranked features.
        The cluster is by default post_processed through the post_features_cluster method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        object
        """
        p = self.parameters(**kwargs)
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        return self.get_features_by_cluster_from_indices(self.diteration.y_order[:k], **kwargs)

    # Covering part

    def get_documents_by_coverage(self, k=None, **kwargs):
        """
        Returns a list of top covering documents.
        By default, the documents are post_processed through the post_documents_item method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        list

        """
        p = self.parameters(**kwargs)
        post = p['post']
        if k is None:
            k = auto_k(data=self.diteration.x_relevance,
                       order=self.diteration.x_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        p['post'] = False
        cluster = self.get_documents_by_cluster(k=int(k * p['stretch']),
                                                **p)
        indices = covering_order(cluster, wide=p['wide'])[:k]
        if post:
            return [self.post_documents_item(self, i) for i in indices]
        else:
            return indices

    def get_features_by_coverage(self, k=None, **kwargs):
        """
        Returns a list of top covering features.
        By default, the features are post_processed through the post_features_item method.

        Parameters
        ----------
        k: int, optional
            Number of documents to output. If not set, k is automatically computed
            using the max_k and target_k runtime parameters.
        kwargs: dict, optional
            Custom runtime parameters.

        Returns
        -------
        list

        """
        p = self.parameters(**kwargs)
        post = p['post']
        if k is None:
            k = auto_k(data=self.diteration.y_relevance,
                       order=self.diteration.y_order,
                       max_k=p['max_k'],
                       target=p['target_k'])
        p['post'] = False
        cluster = self.get_features_by_cluster(k=int(k * p['stretch']),
                                               **p)
        indices = covering_order(cluster, wide=p['wide'])[:k]
        if post:
            return [self.post_features_item(self, i) for i in indices]
        else:
            return indices


class XGismo(Gismo):
    """
    Given two distinct embeddings base on the same set of documents, builds a new gismo.
    The features of ``x_embedding`` are the corpus of this new gismo.
    The features of ``y_embedding`` are the features of this new gismo.
    The dual embedding of the new gismo is obtained by crossing the two input dual embeddings.

    xgismo behaves essentially as a gismo object. The main difference is an additional parameter ``y`` for the
    rank method, to control if the query projection should be performed on the ``y_embedding`` or on the
    ``x_embedding``.

    Parameters
    ----------
    x_embedding: Embedding
        The *left* embedding, which defines the documents of the xgismo.
    y_embedding: Embedding
        The *right* embedding, which defines the features of the xgismo.
    filename: str, optional
                If set, will load xgismo from file.
    path: str or Path, optional
        Directory where the xgismo is to be loaded from.
    kwargs: dict
        Custom default runtime parameters.
        You just need to specify the parameters that differ from :class:`~gismo.parameters.DEFAULT_PARAMETERS`.

    Examples
    ---------
    One the main use case for XGismo consists in transforming a list of articles into a Gismo that relates authors
    and the words they use. Let's start by retrieving a few articles.

    >>> toy_url = "https://dblp.org/pers/xx/m/Mathieu:Fabien.xml"
    >>> source = [a for a in url2source(toy_url) if int(a['year'])<2020]

    Then we build the embedding of words.

    >>> corpus = Corpus(source, to_text=lambda x: x['title'])
    >>> w_count = CountVectorizer(dtype=float, stop_words='english')
    >>> w_embedding = Embedding(w_count)
    >>> w_embedding.fit_transform(corpus)

    And the embedding of authors.

    >>> to_authors_text = lambda dic: " ".join([a.replace(' ', '_') for a in dic['authors']])
    >>> corpus.to_text = to_authors_text
    >>> a_count = CountVectorizer(dtype=float, preprocessor=lambda x:x, tokenizer=lambda x: x.split(' '))
    >>> a_embedding = Embedding(a_count)
    >>> a_embedding.fit_transform(corpus)

    We can now combine the two embeddings in one xgismo.

    >>> xgismo = XGismo(a_embedding, w_embedding)
    >>> xgismo.post_documents_item = lambda g, i: g.corpus[i].replace('_', ' ')

    We can use xgismo to query keyword(s).

    >>> success = xgismo.rank("Pagerank")
    >>> xgismo.get_documents_by_rank()
    ['Mohamed Bouklit', 'Dohy Hong', 'The Dang Huynh']

    We can use it to query researcher(s).

    >>> success = xgismo.rank("Anne_Bouillard", y=False)
    >>> xgismo.get_documents_by_rank()
    ['Anne Bouillard', 'Elie de Panafieu', 'Céline Comte', 'Philippe Sehier', 'Thomas Deiß', 'Dmitry Lebedev']
    """
    def __init__(self, x_embedding=None, y_embedding=None, filename=None, path=".", **kwargs):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            embedding = Embedding()
            embedding.n = x_embedding.m
            embedding.m = y_embedding.m
            embedding.features = y_embedding.features
            embedding.x = np.dot(x_embedding.y, y_embedding.x)
            embedding.x_norm = np.ones(embedding.n)
            embedding.y = np.dot(y_embedding.y, x_embedding.x)
            embedding.y_norm = np.ones(embedding.m)
            embedding.idf = y_embedding.idf
            super().__init__(corpus=Corpus(x_embedding.features, to_text=lambda x: x), embedding=embedding, **kwargs)

            self.x_projection = x_embedding.query_projection
            self.y_projection = y_embedding.query_projection

    def rank(self, query="", y=True, **kwargs):
        """
        Runs the DIteration using query as starting point.
        ``query`` can be evaluated on features (``y=True``) or documents (``y=False``).

        Parameters
        ----------
        query: str
           Text that starts DIteration
        y: bool
           Determines if query should be evaluated on features (``True``) or documents (``False``).
        kwargs: dict, optional
            Custom runtime parameters.


        Returns
        -------
        success: bool
            success of the query projection. If projection fails, a ranking on uniform distribution is performed.
        """
        p = self.parameters(**kwargs)
        if y:
            z, found = self.y_projection(query)
            offset = 1.0
        else:
            z, found = self.x_projection(query)
            z = np.dot(z, self.embedding.x)
            offset = 0.0
        self.embedding._result_found = found
        self.diteration(self.embedding.x, self.embedding.y, z,
                        alpha=p['alpha'], n_iter=p['n_iter'],
                        offset=offset, memory=p['memory'])
        return found
