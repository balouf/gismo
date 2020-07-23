import numpy as np
import logging
from scipy.sparse import csr_matrix, vstack, hstack

from gismo.common import auto_k, toy_source_text
from gismo.parameters import Parameters, DEFAULT_LANDMARKS_PARAMETERS
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.gismo import Gismo
from gismo.clustering import subspace_clusterize, Cluster, subspace_distortion
from gismo.post_processing import post_landmarks_item_raw, post_landmarks_cluster_json

logging.basicConfig()
log = logging.getLogger("Gismo")


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
    kwargs: dict
        Custom default runtime parameters.
        You just need to specify the parameters that differ from :obj:`~gismo.parameters.DEFAULT_LANDMARKS_PARAMETERS`.

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
    >>> landmarks.get_landmarks_by_rank(gismo) # doctest: +NORMALIZE_WHITESPACE
    [{'name': 'Gremlins', 'content': 'The Gremlins movie features a Mogwai.'},
    {'name': 'Star Wars', 'content': 'The Star Wars movies feature Yoda.'},
    {'name': 'Movies', 'content': 'Star Wars, Gremlins, and Blade are movies.'}]

    For better readibility, we set the item post_processing to return the `name` of a landmark item.

    >>> landmarks.post_item = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_landmarks_by_rank(gismo)
    ['Gremlins', 'Star Wars', 'Movies']

    The balance adjusts between documents and features spaces.
    A balance set to 1.0 focuses only on documents.

    >>> success = gismo.rank('blade')
    >>> landmarks.get_landmarks_by_rank(gismo, balance=1)
    ['Movies']

    A balance set to 0.0 focuses only on features.
    For *blade*, this triggers *Shadoks* as a secondary result, because of the shared word *sentence*.

    >>> landmarks.get_landmarks_by_rank(gismo, balance=0)
    ['Movies', 'Shadoks']

    Landmarks can be used to analyze landmarks.

    >>> landmarks.get_landmarks_by_rank(landmarks)
    ['Gremlins', 'Star Wars']

    See again how balance can change things.
    Here a balance set to 0.0 (using only features) fully changes the results.

    >>> landmarks.get_landmarks_by_rank(landmarks, balance=0)
    ['Shadoks']

    Like for :py:class:`~gismo.gismo.Gismo`, landmarks can provide clusters.

    >>> success = gismo.rank('gizmo')
    >>> landmarks.get_landmarks_by_cluster(gismo) # doctest: +NORMALIZE_WHITESPACE
    {'landmark': {'name': 'Gremlins',
                  'content': 'The Gremlins movie features a Mogwai.'},
                  'focus': 0.9999983623793101,
                  'children': [{'landmark': {'name': 'Gremlins', 'content': 'The Gremlins movie features a Mogwai.'},
                                'focus': 1.0, 'children': []},
                             {'landmark': {'name': 'Star Wars', 'content': 'The Star Wars movies feature Yoda.'},
                              'focus': 1.0, 'children': []},
                             {'landmark': {'name': 'Movies', 'content': 'Star Wars, Gremlins, and Blade are movies.'},
                              'focus': 1.0, 'children': []}]}

    We can set the `post_cluster` attribute to customize the output. Gismo provides a simple display.

    >>> from gismo.post_processing import post_landmarks_cluster_print
    >>> landmarks.post_cluster = post_landmarks_cluster_print
    >>> landmarks.get_landmarks_by_cluster(gismo) # doctest: +NORMALIZE_WHITESPACE
    F: 1.00.
    - Gremlins
    - Star Wars
    - Movies

    Like for :py:class:`~gismo.gismo.Gismo`, parameters like `k`, `distortion`, or `resolution` can be used.

    >>> landmarks.get_landmarks_by_cluster(gismo, k=4, distortion=False, resolution=.9) # doctest: +NORMALIZE_WHITESPACE
    F: 0.03.
    - F: 0.93.
    -- F: 1.00.
    --- Gremlins
    --- Star Wars
    -- Movies
    - Shadoks

    Note that a :py:class:`~gismo.clustering.Cluster` can also be used as reference for the
    :meth:`~gismo.landmarks.Landmarks.get_landmarks_by_rank` and
    :meth:`~gismo.landmarks.Landmarks.get_landmarks_by_cluster` methods.

    >>> cluster = landmarks.get_landmarks_by_cluster(gismo, post=False)
    >>> landmarks.get_landmarks_by_rank(cluster)
    ['Gremlins', 'Star Wars', 'Movies']

    Yet, you cannot use anything as reference. For example, you cannot use a string as such.

    >>> landmarks.get_landmarks_by_rank("Landmarks do not use external queries (pass them to a gismo")  # doctest.ELLIPSIS
    Traceback (most recent call last):
    ...
    TypeError: bad operand type for unary -: 'NoneType'

    Last but not least, landmarks can be used to reduce the size of a source or a :py:class:`~gismo.gismo.Gismo`.
    The reduction is controlled by the `x_density` attribute that tells the number of documents each landmark will
    allow to keep.

    >>> landmarks.parameters.x_density = 1
    >>> reduced_gismo = landmarks.get_reduced_gismo(gismo)
    >>> reduced_gismo.corpus.source # doctest: +NORMALIZE_WHITESPACE
    ['This is another sentence about Shadoks.',
    'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the
    Gremlins movie by comparing Gizmo and Yoda.']

    Side remark #1: in the constructor, `to_text` indicates how to convert an item to `str`, while `ranking_function`
    specifies how to run a query on a :py:class:`~gismo.gismo.Gismo`. Yet, it is possible to have the text conversion
    handled by the `ranking_function`.

    >>> landmarks = Landmarks(landmarks_source, rank=lambda g, q: g.rank(q['content']))
    >>> landmarks.fit(gismo)
    >>> success = gismo.rank('yoda')
    >>> landmarks.post_item = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_landmarks_by_rank(gismo)
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
    >>> landmarks.post_item = lambda lmk, i: lmk[i]['name']
    >>> landmarks.get_landmarks_by_rank(gismo)
    ['Shadoks', 'unrelated']
    """

    def __init__(self, source=None, to_text=None, filename=None, path='.', **kwargs):
        self.parameters = Parameters(parameter_list=DEFAULT_LANDMARKS_PARAMETERS, **kwargs)

        self.x_vectors = None
        self.y_vectors = None

        self.x_direction = None
        self.y_direction = None

        self.post_item = post_landmarks_item_raw
        self.post_cluster = post_landmarks_cluster_json

        super().__init__(source=source, to_text=to_text, filename=filename, path=path)

    def embed_entry(self, gismo, entry, **kwargs):
        p = self.parameters(**kwargs)
        log.debug(f"Processing {entry}.")
        success = p['rank'](gismo, entry)
        if not success:
            log.warning(f"Query {entry} didn't match any feature.")

        indptr = [0, min(p['y_density'], gismo.embedding.m)]
        indices = gismo.diteration.y_order[:p['y_density']]
        data = gismo.diteration.y_relevance[indices]
        y = csr_matrix((data, indices, indptr), shape=(1, gismo.embedding.m))

        indptr = [0, min(p['x_density'], gismo.embedding.n)]
        indices = gismo.diteration.x_order[:p['x_density']]
        data = gismo.diteration.x_relevance[indices]
        x = csr_matrix((data, indices, indptr), shape=(1, gismo.embedding.n))
        log.debug(f"Landmarks of {entry} computed.")
        return x / np.sum(x), y / np.sum(y)

    def fit(self, gismo, **kwargs):
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
            The Gismo on which vectors will be computed.
        kwargs: dict
            Custom Landmarks runtime parameters.

        Returns
        -------
        None

        """
        log.info(f"Start computation of {len(self)} landmarks.")
        xy = [self.embed_entry(gismo, entry, **kwargs) for entry in self.iterate_text()]
        self.x_vectors = vstack([v[0] for v in xy])
        self.x_direction = np.squeeze(np.asarray(np.sum(self.x_vectors, axis=0)))
        self.y_vectors = vstack([v[1] for v in xy])
        self.y_direction = np.squeeze(np.asarray(np.sum(self.y_vectors, axis=0)))
        log.info(f"All landmarks are built.")

    def get_base(self, balance):
        return csr_matrix(hstack([balance * self.x_vectors, (1 - balance) * self.y_vectors]))

    def get_landmarks_by_rank(self, reference, k=None, base=None, **kwargs):
        p = self.parameters(**kwargs)
        if base is None:
            base = self.get_base(p['balance'])
        direction = get_direction(reference, p['balance'])
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
            k = auto_k(data=similarities, order=order, max_k=p['max_k'], target=p['target_k'])
        if p['post']:
            return [self.post_item(self, i) for i in order[:k]]
        else:
            return order[:k]

    def get_landmarks_by_cluster(self, reference, k=None, **kwargs):
        p = self.parameters(**kwargs)
        base = self.get_base(p['balance'])
        direction = get_direction(reference, p['balance'])
        order = self.get_landmarks_by_rank(reference=direction, k=k, base=base,
                                           target_k=p['target_k'], max_k=p['max_k'],
                                           balance=p['balance'], post=False)


        subspace = vstack([base[i, :] for i in order])
        if p['distortion']>0:
            subspace_distortion(indices=subspace.indices, data=subspace.data,
                                relevance=direction, distortion=p['distortion'])

        cluster = subspace_clusterize(subspace, resolution=p['resolution'], indices=order)
        if p['post']:
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
        reduced_embedding = Embedding(vectorizer=gismo.embedding.vectorizer)
        reduced_embedding.fit_transform(reduced_corpus)
        reduced_gismo = Gismo(reduced_corpus, reduced_embedding)
        reduced_gismo.parameters = gismo.parameters
        return reduced_gismo
