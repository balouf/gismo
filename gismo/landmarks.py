import numpy as np
import logging
from scipy.sparse import csr_matrix, vstack, hstack

from gismo.common import auto_k, toy_source_text
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.gismo import Gismo
from gismo.clustering import subspace_clusterize, Cluster

logging.basicConfig()
log = logging.getLogger("Gismo")


def post_landmarks_item_default(landmark, i):
    """
    Default post processor for individual landmarks.

    Parameters
    ----------
    landmark: Landmarks
        A Landmarks instance
    i: int
        Indice of the landmark to process.

    Returns
    -------
    object
        The landmark of indice i.
    """
    return landmark[i]


def post_landmarks_cluster_default(landmark, cluster):
    """
    Default post processor for a cluster of landmarks.

    Parameters
    ----------
    landmark: Landmarks
        A Landmarks instance
    cluster: Cluster
        Cluster of the landmarks to process.

    Returns
    -------
    dict
        A dict with the head landmark, cluster focus, and list of children.
    """
    return {'landmark': landmark[cluster.indice],
            'focus': cluster.focus,
            'children': [post_landmarks_cluster_default(landmark, child) for child in cluster.children]}


def get_direction(reference, balance):
    """
    Converts a reference object into a `n+m` direction (dense or sparse depending on reference type).

    Parameters
    ----------
    reference: Gismo or Landmarks or Cluster or np.ndarray or csr_matrix.
        The object from which a direction will be extracted.
    balance: float in range [0.0, 1.0]
        The trade-off between documents and features.
        Set to 0.0, only the feature space will be used.
        Set to 1.0, only the document space will be used.

    Returns
    -------
    np.ndarray or csr_matrix
        A `n+m` direction.
    """
    if isinstance(reference, Gismo):
        return np.hstack([balance * reference.diteration.x_relevance,
                          (1 - balance) * reference.diteration.y_relevance])
    if isinstance(reference, Landmarks):
        return np.hstack([balance * reference.x_direction, (1 - balance) * reference.y_direction])
    if isinstance(reference, Cluster):
        return reference.vector
    return reference


class Landmarks(Corpus):
    """
    The `Landmarks` class is a subclass :py:class:`~gismo.corpus.Corpus`.
    It offers the capability to batch-rank all its entries against a :py:class:`~gismo.gismo.Gismo` instance.
    After it has been processed, a `Landmarks` can be used to analyze/classify
    :py:class:`~gismo.gismo.Gismo` queries, :py:class:`~gismo.clustering.Cluster`,
    or :py:class:`~gismo.landmarks.Landmarks`.

    Landmarks also offers the possibility to reduce a source or a gismo to its neighborhood.
    This can be useful if the source is huge and one wants something smaller for performance.

    Parameters
    ------------

    source: list
        The list of items that form Landmarks.
    to_text: function
        The function that transforms an item into text
    filename: str, optional
        Load landmarks from filename
    path: str or Path, optional
        Directory where the landmarks instance is to be loaded from.
    x_density: int
        nnz entries to keep on the documents space.
    y_density: int
        nnz entries to keep on the features space.
    ranking_function: function, optional
        Function that uses a gismo and a query as inputs and runs the query on the gismo. This is useful for
        :py:class:`~gismo.gismo.Gismo` subclasses like :py:class:`~gismo.gismo.XGismo`, which have multiple ways
        to run queries.

    Examples
    --------

    Landmarks lean on a Gismo. We can use a toy Gismo to start with.

    >>> corpus = Corpus(toy_source_text)
    >>> embedding = Embedding()
    >>> embedding.fit_transform(corpus)
    >>> gismo = Gismo(corpus, embedding)
    >>> print(toy_source_text) # doctest: +NORMALIZE_WHITESPACE
    ['Gizmo is a Mogwaï.',
    'This is a sentence about Blade.',
    'This is another sentence about Shadoks.',
    'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.',
    'In chinese folklore, a Mogwaï is a demon.']

    Landmarks are constructed exactly like a Gismo object, with a source and a `to_text` function.

    >>> landmarks_source = [{'name': 'Movies', 'content': 'Star Wars, Gremlins, and Blade are movies.'},
    ... {'name': 'Gremlins', 'content': 'The Gremlins movie features a Mogwai.'},
    ... {'name': 'Star Wars', 'content': 'The Star Wars movies feature Yoda.'},
    ... {'name': 'Shadoks', 'content': 'Shadoks is a French sarcastic show.'},]
    >>> landmarks = Landmarks(landmarks_source, to_text=lambda e: e['content'])

    The :func:`~gismo.landmarks.Landmarks.fit` method compute gismo queries for all landmarks and retain the results.

    >>> landmarks.fit(gismo)

    We run the request *Yoda* and look at the key landmarks.
    Note that *Gremlins* comes before *Star Wars*. This is actually *correct* in this small dataset:
    the word `Yoda` only exists in one sentence, which contains the words `Gremlins` and `Gizmo`.

    >>> success = gismo.rank('yoda')
    >>> landmarks.get_ranked_landmarks(gismo) # doctest: +NORMALIZE_WHITESPACE
    [{'name': 'Gremlins', 'content': 'The Gremlins movie features a Mogwai.'},
    {'name': 'Star Wars', 'content': 'The Star Wars movies feature Yoda.'},
    {'name': 'Movies', 'content': 'Star Wars, Gremlins, and Blade are movies.'}]

    For better readibility, we set the item post_processing to return the `name` of a landmark item.

    >>> landmarks.post_rank = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_ranked_landmarks(gismo)
    ['Gremlins', 'Star Wars', 'Movies']

    The balance adjusts between documents and features spaces.
    A balance set to 1.0 focuses only on documents.

    >>> success = gismo.rank('blade')
    >>> landmarks.get_ranked_landmarks(gismo, balance=1)
    ['Movies']

    A balance set to 0.0 focuses only on features.
    For *blade*, this triggers *Shadoks* as a secondary result, because of the shared word *sentence*.

    >>> landmarks.get_ranked_landmarks(gismo, balance=0)
    ['Movies', 'Shadoks']

    Landmarks can be used to analyze landmarks.

    >>> landmarks.get_ranked_landmarks(landmarks)
    ['Gremlins', 'Star Wars']

    See again how balance can change things.
    Here a balance set to 0.0 (using only features) fully changes the results.

    >>> landmarks.get_ranked_landmarks(landmarks, balance=0)
    ['Shadoks']

    Like for :py:class:`~gismo.gismo.Gismo`, landmarks can provide clusters.

    >>> success = gismo.rank('gizmo')
    >>> landmarks.get_clustered_landmarks(gismo) # doctest: +NORMALIZE_WHITESPACE
    {'landmark': {'name': 'Gremlins',
                  'content': 'The Gremlins movie features a Mogwai.'},
                  'focus': 0.9999983623793101,
                  'children': [{'landmark': {'name': 'Gremlins', 'content': 'The Gremlins movie features a Mogwai.'},
                                'focus': 1.0, 'children': []},
                             {'landmark': {'name': 'Star Wars', 'content': 'The Star Wars movies feature Yoda.'},
                              'focus': 1.0, 'children': []},
                             {'landmark': {'name': 'Movies', 'content': 'Star Wars, Gremlins, and Blade are movies.'},
                              'focus': 1.0, 'children': []}]}

    We can set the `post_cluster` attribute to customize the output.

    >>> def post_cluster(lmk, cluster, depth=0):
    ...     name = f"|{'-'*depth}"
    ...     if len(cluster.children)==0:
    ...         name = f"{name} {lmk[cluster.indice]['name']}"
    ...     print(name)
    ...     for c in cluster.children:
    ...         post_cluster(lmk, c, depth=depth+1)
    >>> landmarks.post_cluster = post_cluster
    >>> landmarks.get_clustered_landmarks(gismo)
    |
    |- Gremlins
    |- Star Wars
    |- Movies

    Like for :py:class:`~gismo.gismo.Gismo`, parameters like `k`, `distortion`, or `resolution` can be used.

    >>> landmarks.get_clustered_landmarks(gismo, k=4, distortion=False, resolution=.9)
    |
    |-
    |--
    |--- Gremlins
    |--- Star Wars
    |-- Movies
    |- Shadoks

    Note that a :py:class:`~gismo.clustering.Cluster` can also be used as reference for the
    :func:`~gismo.landmarks.Landmarks.get_ranked_landmarks` and
    :func:`~gismo.landmarks.Landmarks.get_clustered_landmarks` methods.

    >>> cluster = landmarks.get_clustered_landmarks(gismo, post=False)
    >>> landmarks.get_ranked_landmarks(cluster)
    ['Gremlins', 'Star Wars', 'Movies']

    Yet, you cannot use anything as reference. For example, you cannot use a string as such.

    >>> landmarks.get_ranked_landmarks("Landmarks do not use external queries (pass them to a gismo")  # doctest.ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: bad operand type for unary -: 'NoneType'

    Last but not least, landmarks can be used to reduce the size of a source or a :py:class:`~gismo.gismo.Gismo`.
    The reduction is controlled by the `x_density` attribute that tells the number of documents each landmark will
    allow to keep.

    >>> landmarks.x_density = 1
    >>> reduced_gismo = landmarks.get_reduced_gismo(gismo)
    >>> reduced_gismo.corpus.source # doctest: +NORMALIZE_WHITESPACE
    ['This is another sentence about Shadoks.',
    'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the
    Gremlins movie by comparing Gizmo and Yoda.']

    Side remark #1: in the constructor, `to_text` indicates how to convert an item to `str`, while `ranking_function`
    specifies how to run a query on a :py:class:`~gismo.gismo.Gismo`. Yet, it is possible to construct the example
    above with the text conversion handled by the `ranking_function`.

    >>> landmarks = Landmarks(landmarks_source, ranking_function=lambda g, q: g.rank(q['content']))
    >>> landmarks.fit(gismo)
    >>> success = gismo.rank('yoda')
    >>> landmarks.post_rank = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_ranked_landmarks(gismo)
    ['Star Wars', 'Movies', 'Gremlins']

    However, this is bad practice. When you only need to customize the way an item is converted to text, you should
    stick to `to_text`. `ranking_function` is for more elaborated filters that require to change the default way
    gismo does queries.

    Side remark #2: if a landmark item query fails (its text does not intersect the gismo features),
    the default uniform projection will be used and a warning will be issued. This may yield to
    undesired results.

    >>> landmarks_source.append({'name': 'unrelated', 'content': 'unrelated.'})
    >>> landmarks = Landmarks(landmarks_source, to_text=lambda e: e['content'])
    >>> landmarks.fit(gismo)
    >>> success = gismo.rank('gizmo')
    >>> landmarks.post_rank = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_ranked_landmarks(gismo)
    ['Shadoks', 'unrelated']
    """

    def __init__(self, source=None, to_text=None, filename=None, path='.',
                 x_density=1000, y_density=1000, ranking_function=None):
        if ranking_function is None:
            self.rank = lambda g, q: g.rank(q)
        else:
            self.rank = ranking_function

        self.x_density = x_density
        self.y_density = y_density

        self.x_vectors = None
        self.y_vectors = None

        self.x_direction = None
        self.y_direction = None

        self.x_len = None
        self.y_len = None

        self.post_rank = post_landmarks_item_default
        self.post_cluster = post_landmarks_cluster_default

        super().__init__(source=source, to_text=to_text, filename=filename, path=path)

    def embed_entry(self, gismo, entry):
        log.debug(f"Processing {entry}.")
        success = self.rank(gismo, entry)
        if not success:
            log.warning(f"Query {entry} didn't match any feature.")

        indptr = [0, min(self.y_density, self.y_len)]
        indices = gismo.diteration.y_order[:self.y_density]
        data = gismo.diteration.y_relevance[indices]
        y = csr_matrix((data, indices, indptr), shape=(1, gismo.embedding.m))

        indptr = [0, min(self.x_density, self.x_len)]
        indices = gismo.diteration.x_order[:self.x_density]
        data = gismo.diteration.x_relevance[indices]
        x = csr_matrix((data, indices, indptr), shape=(1, gismo.embedding.n))
        log.debug(f"Landmarks of {entry} computed.")
        return x / np.sum(x), y / np.sum(y)

    def fit(self, gismo):
        """
        Runs gismo queries on all landmarks.
        The relevance results are used to build two set of vectors:
        `x_vectors` is the vectors on the document space;
        `y_vectors` is the vectors on the document space.
        On each space, vectors are summed to build a direction, which is a sort of
        vector summary of the landmarks.

        Parameters
        ----------
        gismo: Gismo

        Returns
        -------
        None

        """
        log.info(f"Start computation of {len(self)} landmarks.")
        self.x_len = gismo.embedding.n
        self.y_len = gismo.embedding.m
        xy = [self.embed_entry(gismo, entry) for entry in self.iterate_text()]
        self.x_vectors = vstack([v[0] for v in xy])
        self.x_direction = np.squeeze(np.asarray(np.sum(self.x_vectors, axis=0)))
        self.y_vectors = vstack([v[1] for v in xy])
        self.y_direction = np.squeeze(np.asarray(np.sum(self.y_vectors, axis=0)))
        log.info(f"All landmarks are built.")

    def get_base(self, balance):
        return csr_matrix(hstack([balance * self.x_vectors, (1 - balance) * self.y_vectors]))

    def get_ranked_landmarks(self, reference, k=None, target=1.0, max_k=100, balance=0.5,
                             base=None, post=True):
        if base is None:
            base = self.get_base(balance)
        direction = get_direction(reference, balance)
        if isinstance(direction, np.ndarray):
            similarities = base.dot(direction)
        elif isinstance(direction, csr_matrix):
            similarities = np.squeeze(base.dot(direction.T).toarray())
        else:
            log.error("Direction type not supported. Direction must be gismo.gismo.Gismo, gismo.clustering.Cluster, "
                      "gismo.landmarks.Landmarks, numpy.ndarray or scipy.sparse.csr_matrix.")
            similarities = None
        order = np.argsort(-similarities)
        if k is None:
            k = auto_k(data=similarities, order=order, max_k=max_k, target=target)
        if post:
            return [self.post_rank(self, i) for i in order[:k]]
        else:
            return order[:k]

    def get_clustered_landmarks(self, reference, k=None, target=1.0, max_k=100, balance=0.5,
                                resolution=.7, distortion=True, post=True):
        base = self.get_base(balance)
        direction = get_direction(reference, balance)
        order = self.get_ranked_landmarks(reference=direction, k=k, target=target, max_k=max_k,
                                          balance=balance, base=base, post=False)
        if distortion:
            subspace = csr_matrix(vstack([base[i, :].multiply(direction) for i in order]))
        else:
            subspace = csr_matrix(vstack([base[i, :] for i in order]))
        cluster = subspace_clusterize(subspace, resolution=resolution, indices=order)
        if post:
            return self.post_cluster(self, cluster)
        else:
            return cluster

    def get_reduced_source(self, gismo, rebuild=True):
        if rebuild:
            self.fit(gismo)
        reduced_indices = {i for i in self.x_vectors.indices}
        return [a for i, a in enumerate(gismo.corpus.source) if i in reduced_indices]

    def get_reduced_gismo(self, gismo, rebuild=True):
        reduced_corpus = Corpus(self.get_reduced_source(gismo, rebuild=rebuild),
                                to_text=gismo.corpus.to_text)
        reduced_embedding = Embedding(vectorizer=gismo.embedding.vect)
        reduced_embedding.fit_transform(reduced_corpus)
        return Gismo(reduced_corpus, reduced_embedding)
