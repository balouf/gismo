#!/usr/bin/env python
# coding: utf-8
#
# GISMO: a Generic Information Search with a Mind of its Own

import numpy as np

from scipy.sparse import csr_matrix
from numba import njit
from sklearn.feature_extraction.text import CountVectorizer

from gismo.common import MixInIO, toy_source_text
from gismo.corpus import Corpus


# 1-norm for diffusion (input is X or Y indptr and data, inplace modification)
@njit
def l1_normalize(indptr, data):
    """
    Computes L1 norm on sparse embedding (x or y) and applies inplace normalization.

    Parameters
    ----------
    indptr: :class:`~numpy.ndarray`
        Pointers of the embedding (e.g. x.indptr).
    data: :class:`~numpy.ndarray`
        Values of the embedding  (e.g. x.data).

    Returns
    -------
    l1_norm: :class:`~numpy.ndarray`
        L1 norms of all vectors of the embedding before normalization.
    """

    """
    Normalize inplace the embedding (X or Y)
    :param indptr: the pointers of the vector
    :param data:  the data of the vector
    :return: the l1 norm. Data is normalized in place
    """
    n = len(indptr) - 1
    l1_norms = np.zeros(n)
    for i in range(n):
        l1_norms[i] = np.sum(data[indptr[i]:indptr[i + 1]])
        if l1_norms[i] > 0:
            data[indptr[i]:indptr[i + 1]] /= l1_norms[i]
    return l1_norms


# Note: the use of external embedding breaks a symmetry between X and Y. IDF needs to be stored if one wants to switch.

# ITF transformation
@njit
def itf_fit_transform(indptr, data, m):
    """
    Applies inplace Inverse-Term-Frequency transformation on sparse embedding x.

    Parameters
    ----------
    indptr: :class:`~numpy.ndarray`
        Pointers of the embedding (e.g. x.indptr).
    data: :class:`~numpy.ndarray`
        Values of the embedding  (e.g. x.data).
    m: int
        Number of features
    """
    n = len(indptr) - 1
    log_m = np.log(1 + m)
    for i in range(n):
        data[indptr[i]:indptr[i + 1]] *= log_m - np.log(1 + indptr[i + 1] - indptr[i])


# IDF computation
@njit
def idf_fit(indptr, n):
    """
    Computes the Inverse-Document-Frequency vector on sparse embedding y.

    Parameters
    ----------
    indptr: :class:`~numpy.ndarray`
        Pointers of the embedding y (e.g. y.indptr).
    n: int
        Number of documents.

    Returns
    -------
    idf_vector: :class:`~numpy.ndarray`
        IDF vector of size `m`.
    """
    m = len(indptr) - 1
    idf_vector = np.log(1 + n) * np.ones(m)
    for i in range(m):
        idf_vector[i] -= np.log(1 + indptr[i + 1] - indptr[i])
    return idf_vector


# IDF transformation
@njit
def idf_transform(indptr, data, idf_vector):
    """
    Applies inplace Inverse-Document-Frequency transformation on sparse embedding y.

    Parameters
    ----------
    indptr: :class:`~numpy.ndarray`
        Pointers of the embedding y (e.g. y.indptr).
    data: :class:`~numpy.ndarray`
        Values of the embedding y (e.g. y.data).
    idf_vector: :class:`~numpy.ndarray`
        IDF vector of the embedding, obtained from :func:`~gismo.embedding.idf_fit`.
    """
    m = len(indptr) - 1
    for i in range(m):
        data[indptr[i]:indptr[i + 1]] *= idf_vector[i]


@njit
def query_shape(indices, data, idf):
    """
    Applies inplace logarithmic smoothing, IDF weighting, and normalization
    to the output of the
    :class:`~sklearn.feature_extraction.text.CountVectorizer`
    :meth:`~sklearn.feature_extraction.text.CountVectorizer.transform` method.

    Parameters
    ----------
    indices: :class:`~numpy.ndarray`
        Indice attribute of the :class:`~scipy.sparse.csr_matrix` obtained from
        :meth:`~sklearn.feature_extraction.text.CountVectorizer.transform`.
    data: :class:`~numpy.ndarray`
        Data attribute of the :class:`~scipy.sparse.csr_matrix` obtained from
        :meth:`~sklearn.feature_extraction.text.CountVectorizer.transform`.
    idf: :class:`~numpy.ndarray`
        IDF vector of the embedding, obtained from :func:`~gismo.embedding.idf_fit`.

    Returns
    -------
    norm: float
        The norm of the vector before normalization.
    """
    # log normalization
    data[:] = 1 + np.log(data)
    # IdF
    for i, indice in enumerate(indices):
        data[i] *= idf[indice]
    # Normalization
    norm = np.sum(data)
    if norm > 0:
        data[:] /= norm
    return norm


def auto_vect(corpus=None):
    """
    Creates a default :class:`~sklearn.feature_extraction.text.CountVectorizer`
    compatible with the
    :class:`~gismo.embedding.Embedding` constructor.
    For not-too-small corpi, a slight frequency-filter is applied.

    Parameters
    ----------
    corpus: :class:`~gismo.corpus.Corpus`, optional
        The corpus for which the
        :class:`~sklearn.feature_extraction.text.CountVectorizer`
        is intended.

    Returns
    -------
    :class:`~sklearn.feature_extraction.text.CountVectorizer`
        A :class:`~sklearn.feature_extraction.text.CountVectorizer`
        object compatible with the
        :class:`~gismo.embedding.Embedding` constructor.
    """
    n = len(corpus) if corpus is not None else 1
    (min_df, max_df) = (3, .15) if n > 100 else (1, 1.0)
    return CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=[1, 1],
        stop_words="english",
        dtype=float
    )


class Embedding(MixInIO):
    """
    This class leverages the :class:`~sklearn.feature_extraction.text.CountVectorizer`
    class to build the dual embedding of a :class:`~gismo.corpus.Corpus`.

    * Documents are embedded in the space of features;
    * Features are embedded in the space of documents.

    See the examples and methods below for all usages of the class.

    Parameters
    ----------
    vectorizer: :class:`~sklearn.feature_extraction.text.CountVectorizer`, optional
                Custom :class:`~sklearn.feature_extraction.text.CountVectorizer`
                to override default behavior (recommended).
                Having a :class:`~sklearn.feature_extraction.text.CountVectorizer`
                adapted to the :class:`~gismo.corpus.Corpus` is good practice.
    filename: :py:class:`str`, optional
        If set, load embedding from corresponding file.
    path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
        If set, specify the directory where the embedding is located.
    """

    def __init__(
        self,
        vectorizer=None,
        filename=None,
        path='.'
    ):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.vectorizer = vectorizer
            self.n = 0  # Number of documents
            self.m = 0  # Number of features
            self.x = None  # TF-IDTF X embedding of documents into features, normalized
            self.x_norm = None  # memory of X norm for hierarchical merge
            self.y = None  # Y embedding of features into documents
            self.y_norm = None  # memory of Y norm for hierarchical merge
            self.idf = None  # idf vector
            self.features = None  # vocabulary list
            self._result_found = True  # keep track of projection successes
            self._query = ""  # keep track of projection query

    def fit_transform(self, corpus):
        """
        Ingest a corpus of documents.

        * If not yet set, a default :class:`~sklearn.feature_extraction.text.CountVectorizer` is created.
        * Features are computed and stored (fit).
        * Inverse-Document-Frequency weights of features are computed (fit).
        * TF-IDF embedding of documents is computed and stored (transform).
        * TF-ITF embedding of features is computed and stored (transform).

        Parameters
        ----------
        corpus: :class:`~gismo.corpus.Corpus`
            The corpus to ingest.

        Example
        -------
        >>> corpus=Corpus(toy_source_text)
        >>> embedding = Embedding()
        >>> embedding.fit_transform(corpus)
        >>> embedding.x  # doctest: +NORMALIZE_WHITESPACE
        <5x21 sparse matrix of type '<class 'numpy.float64'>'
            with 25 stored elements in Compressed Sparse Row format>
        >>> embedding.features[:8]
        ['blade', 'chinese', 'comparing', 'demon', 'folklore', 'gizmo', 'gremlins', 'inside']
        """
        if self.vectorizer is None:
            self.vectorizer = auto_vect(corpus)

        # THE FIT PART
        # Start with a simple CountVectorizer X
        x = self.vectorizer.fit_transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vectorizer.stop_words_ = None
        # Populate vocabulary
        self.features = self.vectorizer.get_feature_names()
        # Extract number of documents and features
        (self.n, self.m) = x.shape
        # PART OF TRANSFORM, MUTUALIZED: Apply sublinear smoothing
        x.data = 1 + np.log(x.data)
        # PART OF TRANSFORM, MUTUALIZED: Apply ITF transformation
        itf_fit_transform(indptr=x.indptr, data=x.data, m=self.m)
        # Compute transposed CountVectorizer Y
        self.y = x.tocsc()
        # Compute IDF
        self.idf = idf_fit(self.y.indptr, self.n)

        # THE TRANSFORM PART
        idf_transform(indptr=self.y.indptr, data=self.y.data, idf_vector=self.idf)
        # back to x
        self.x = self.y.tocsr(copy=True)
        # Transpose y
        self.y = self.y.T
        # Normalize
        self.x_norm = l1_normalize(indptr=self.x.indptr, data=self.x.data)
        self.y_norm = l1_normalize(indptr=self.y.indptr, data=self.y.data)

    def fit(self, corpus):
        """
        Learn features from a corpus of documents.

        * If not yet set, a default :class:`~sklearn.feature_extraction.text.CountVectorizer` is created.
        * Features are computed and stored.
        * Inverse-Document-Frequency weights of features are computed.

        Parameters
        ----------
        corpus: :class:`~gismo.corpus.Corpus`
            The corpus to ingest.

        Example
        -------
        >>> corpus=Corpus(toy_source_text)
        >>> embedding = Embedding()
        >>> embedding.fit(corpus)
        >>> len(embedding.idf)
        21
        >>> embedding.features[:8]
        ['blade', 'chinese', 'comparing', 'demon', 'folklore', 'gizmo', 'gremlins', 'inside']
        """
        assert corpus
        if self.vectorizer is None:
            self.vectorizer = auto_vect(corpus)

        # THE FIT PART
        # Start with a simple CountVectorizer X
        x = self.vectorizer.fit_transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vectorizer.stop_words_ = None
        # Populate vocabulary
        self.features = self.vectorizer.get_feature_names()
        # Extract number of documents (required for idf) and features (required in fit)
        (self.n, self.m) = x.shape
        # Compute transposed CountVectorizer Y
        self.y = x.tocsc()
        # Compute IDF
        self.idf = idf_fit(self.y.indptr, self.n)

    def fit_ext(self, embedding):
        """
        Use learned features from another :class:`~gismo.embedding.Embedding`.
        This is useful for the fast creation of local embeddings
        (e.g. at sentence level) out of a global embedding.

        Parameters
        ----------
        embedding: :class:`~gismo.embedding.Embedding`
                  External embedding to copy.

        Examples
        --------
        >>> corpus=Corpus(toy_source_text)
        >>> other_embedding = Embedding()
        >>> other_embedding.fit(corpus)
        >>> embedding = Embedding()
        >>> embedding.fit_ext(other_embedding)
        >>> len(embedding.idf)
        21
        >>> embedding.features[:8]
        ['blade', 'chinese', 'comparing', 'demon', 'folklore', 'gizmo', 'gremlins', 'inside']
        """
        self.m = embedding.m
        self.vectorizer = embedding.vectorizer
        self.idf = embedding.idf
        self.features = embedding.features

    def transform(self, corpus):
        """
        Ingest a corpus of documents using existing features.
        Requires that the embedding has been fitted beforehand.

        * TF-IDF embedding of documents is computed and stored.
        * TF-ITF embedding of features is computed and stored.

        Parameters
        ----------
        corpus: :class:`~gismo.corpus.Corpus`
            The corpus to ingest.

        Example
        -------
        >>> corpus=Corpus(toy_source_text)
        >>> embedding = Embedding()
        >>> embedding.fit_transform(corpus)
        >>> [embedding.features[i] for i in embedding.x.indices[:8]]
        ['gizmo', 'mogwaï', 'blade', 'sentence', 'sentence', 'shadoks', 'comparing', 'gizmo']
        >>> small_corpus = Corpus(["I only talk about Yoda", "Gizmo forever!"])
        >>> embedding.transform(small_corpus)
        >>> [embedding.features[i] for i in embedding.x.indices]
        ['yoda', 'gizmo']
        """
        # The fit part
        assert corpus

        # THE FIT PART
        # Start with a simple CountVectorizer X
        x = self.vectorizer.transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vectorizer.stop_words_ = None
        # Extract number of documents and features
        (self.n, _) = x.shape
        # PART OF TRANSFORM, MUTUALIZED: Apply sublinear smoothing
        x.data = 1 + np.log(x.data)
        # PART OF TRANSFORM, MUTUALIZED: Apply ITF transformation
        itf_fit_transform(indptr=x.indptr, data=x.data, m=self.m)
        # Compute transposed CountVectorizer Y
        self.y = x.tocsc()

        # THE TRANSFORM PART
        idf_transform(indptr=self.y.indptr, data=self.y.data, idf_vector=self.idf)
        # back to x
        self.x = self.y.tocsr(copy=True)
        # Transpose y
        self.y = self.y.T
        # Normalize
        self.x_norm = l1_normalize(indptr=self.x.indptr, data=self.x.data)
        self.y_norm = l1_normalize(indptr=self.y.indptr, data=self.y.data)

    def query_projection(self, query):
        """
        Project a query in the feature space.

        Parameters
        ----------
        query: :class:`str`
               Text to project.

        Returns
        --------
        z: :class:`~scipy.sparse.csr_matrix`
            result of the query projection (IDF distribution if query does not match any feature).
        success: :class:`bool`
            projection success (``True`` if at least one feature been found).

        Example
        -------
        >>> corpus=Corpus(toy_source_text)
        >>> embedding = Embedding()
        >>> embedding.fit_transform(corpus)
        >>> z, success = embedding.query_projection("Gizmo is not Yoda but he rocks!")
        >>> for i in range(len(z.data)):
        ...    print(f"{embedding.features[z.indices[i]]}: {z.data[i]}")
        gizmo: 0.3868528072345416
        yoda: 0.6131471927654585
        >>> success
        True
        >>> z, success = embedding.query_projection("That content does not intersect toy corpus")
        >>> success
        False
        """
        self._query = query
        z = self.vectorizer.transform([query])
        norm = query_shape(indices=z.indices, data=z.data, idf=self.idf)
        if norm == 0:
            z = csr_matrix(self.idf) / np.sum(self.idf)
            self._result_found = False
        else:
            self._result_found = True
        return z, self._result_found
