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
    Normalize inplace the embedding (X or Y)
    :param indptr: the pointers of the vector
    :param data:  the data of the vector
    :return: the l1 norm. Data is normalized in place
    """
    n = len(indptr) - 1
    l = np.zeros(n)
    for i in range(n):
        l[i] = np.sum(data[indptr[i]:indptr[i + 1]])
        if l[i] > 0:
            data[indptr[i]:indptr[i + 1]] /= l[i]
    return l


# Note: the use of external embedding breaks a symmetry between X and Y. IDF needs to be stored if one wants to switch.

# ITF transformation
@njit
def itf_fit_transform(indptr, data, m):
    """
    Apply ITF transformation on embedding X.
    :param indptr: pointers of X
    :param data: data of X
    :param m: Number of features
    :return: None (inplace)
    """
    n = len(indptr) - 1
    log_m = np.log(1 + m)
    for i in range(n):
        data[indptr[i]:indptr[i + 1]] *= log_m - np.log(1 + indptr[i + 1] - indptr[i])


# IDF computation
@njit
def idf_fit(indptr, n):
    """
    Computes idf vector based on embedding Y
    :param indptr: pointers of Y
    :param n: Number of documents
    :return: IDF vector of size m
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
    Applies IDF on embedding Y
    :param indptr:
    :param data:
    :param idf_vector:
    :return: None (inplace)
    """
    m = len(indptr) - 1
    for i in range(m):
        data[indptr[i]:indptr[i + 1]] *= idf_vector[i]


@njit
def query_shape(indices, data, idf):
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

def auto_vect(corpus):
    """
    Creates a default vectorizer according to corpus size.
    Filter words based on their word frequency.
    Keep only word having a frequency in [min_df, max_df].
    You may pass an int (#occurences) or [0, 1] float (frequency).
    :param corpus:
    :return: a CountVectorizer object
    """
    n = len(corpus)
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
    Use vectorizer to compute forward and backward embeddings in a sklearn fashion

    Parameters
    ----------
    vectorizer: CountVectorizer, optional
                Custom vectorizer to override default behavior (recommended).
                Having a vectorizer adapted to the corpus is good practice
    filename: str, optional
                If set, will load embedder from file
    """
    def __init__(
            self,
            vectorizer: CountVectorizer = None,
            filename=None,
            path='.'
    ):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.vect = vectorizer
#            self.document_to_str = document_to_str

            self.n = 0  # Number of documents
            self.m = 0  # Number of features
            self.x = None  # TF-IDTF X embedding of documents into features, normalized
            self.x_norm = None  # memory of X norm for hierarchical merge
            self.y = None  # Y embedding of features into documents
            self.y_norm = None  # memory of Y norm for hierarchical merge
            self.idf = None  # idf vector
            self.features = None # vocabulary list
            self._result_found = True # keep track of projection successes


    def fit_transform(self, corpus: Corpus):
        """
        Ingest a corpus of documents.
        If not yet set, a default vectorizer is created.
        Maps each word with an index.
        Compute for each word its IDF weights (fit).
        For each document, deduce its TF-IDF weights (coordinates in the embedding) (transform) (self.x).
        The transposed matrix is explicitely built because we use sparse matrices (self.y).

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
        if self.vect is None:
            self.vect = auto_vect(corpus)

        # THE FIT PART
        # Start with a simple CountVectorizer X
        x = self.vect.fit_transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vect.stop_words_ = None
        # Populate vocabulary
        self.features = self.vect.get_feature_names()
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

    def fit(self, corpus: Corpus):
        """
        Use a corpus to shape the vectorizer

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
        if self.vect is None:
            self.vect = auto_vect(corpus)

        # THE FIT PART
        # Start with a simple CountVectorizer X
        x = self.vect.fit_transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vect.stop_words_ = None
        # Populate vocabulary
        self.features = self.vect.get_feature_names()
        # Extract number of documents (required for idf) and features (required in fit)
        (self.n, self.m) = x.shape
        # Compute transposed CountVectorizer Y
        self.y = x.tocsc()
        # Compute IDF
        self.idf = idf_fit(self.y.indptr, self.n)

    def fit_ext(self, embedder):
        """
        Use fit from existing embedder. Useful to launch mini-gismos at sentence level.

        Parameters
        ----------
        embedder: Embedding
                  the (external) embedding to copy

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
        self.m = embedder.m
        self.vect = embedder.vect
        self.idf = embedder.idf
        self.features = self.vect.get_feature_names()

    def transform(self, corpus: Corpus):
        """
        Ingest a corpus of documents, assuming the fit has already been done.

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
        x = self.vect.transform(corpus.iterate_text())
        # Release stop_words_ from vectorizer
        self.vect.stop_words_ = None
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
        Project a query in the feature space

        Parameters
        ----------
        query: str
               Text to project to the feature space

        Returns
        --------
        z: csr_matrix
            result of the query projection (uniform distribution if projection fails)
        success: bool
            projection success (has at least one feature been found)

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
        z = self.vect.transform([query])
        norm = query_shape(indices=z.indices, data=z.data, idf=self.idf)
        if norm == 0:
            z = csr_matrix(self.idf) /  np.sum(self.idf)
            self._result_found = False
        else:
            self._result_found = True
        return z, self._result_found
