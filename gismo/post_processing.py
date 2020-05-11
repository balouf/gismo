from sklearn.metrics.pairwise import cosine_similarity


def post_document(gismo, i):
    """
    Document indice to document entry

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    i : int
        document indice

    Returns
    -------
    Object
        Document i from corpus
    """
    return gismo.corpus[i]


def post_document_content(gismo, i, max_size=None):
    """
    Document indice to document content.

    Assumes that document has a 'content' key.

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    i : int
        document indice
    max_size: int, optional
        Maximum number of chars to return

    Returns
    -------
    str
        Content of document i from corpus
    """
    return gismo.corpus[i]['content'][:max_size]


def post_feature(gismo, i):
    """
    Feature indice to feature name

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    i : int
        feature indice

    Returns
    -------
    str
        Feature i from embedding
    """
    return gismo.embedding.features[i]


def post_document_cluster(gismo, cluster):
    """
    Convert cluster of documents into basic json

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    cluster: Cluster
        Cluster of documents

    Returns
    -------
    dict
        dictionary with keys 'document', 'focus', and recursive 'children'
    """
    return {'document': gismo.corpus[cluster.indice],
            'focus': cluster.focus,
            'children': [post_document_cluster(gismo, c) for c in cluster.children]}


def print_document_cluster(gismo, cluster, depth=""):
    """
    Print an ASCII view of a document cluster with metrics (focus, relevance, similarity)

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    cluster: Cluster
        Cluster of documents
    depth: str, optional
        Current depth string used in recursion
    """
    sim = cosine_similarity(cluster.vector, gismo.diteration.y_relevance.reshape(1, -1))[0][0]
    if len(cluster.children) == 0:
        txt = gismo.corpus.to_text(gismo.corpus[cluster.indice])
        print(f"{depth} {txt} "
              f"(R: {gismo.diteration.x_relevance[cluster.indice]:.2f}; "
              f"S: {sim:.2f})")
    else:
        print(f"{depth} F: {cluster.focus:.2f}. "
              f"R: {sum(gismo.diteration.x_relevance[cluster.members]):.2f}. "
              f"S: {sim:.2f}.")
    for c in cluster.children:
        print_document_cluster(gismo, c, depth=depth + '-')


def post_feature_cluster(gismo, cluster):
    """
    Convert feature cluster into basic json

    Parameters
    ----------
    gismo: Gismo
        Gismo instance
    cluster: Cluster
        Cluster of features

    Returns
    -------
    dict
        dictionary with keys 'feature', 'focus', and recursive 'children'
    """
    return {'feature': gismo.embedding.features[cluster.indice],
            'focus': cluster.focus,
            'children': [post_feature_cluster(gismo, c) for c in cluster.children]}


def print_feature_cluster(gismo, cluster, depth=""):
    """
        Print an ASCII view of a feature cluster with metrics (focus, relevance, similarity)

        Parameters
        ----------
        gismo: Gismo
            Gismo instance
        cluster: Cluster
            Cluster of features
        depth: str, optional
            Current depth string used in recursion
        """
    sim = cosine_similarity(cluster.vector, gismo.diteration.x_relevance.reshape(1, -1))[0][0]
    if len(cluster.children) == 0:
        print(f"{depth} {gismo.embedding.features[cluster.indice]} "
              f"(R: {gismo.diteration.y_relevance[cluster.indice]:.2f}; "
              f"S: {sim:.2f})")
    else:
        print(f"{depth} F: {cluster.focus:.2f}. "
              f"R: {sum(gismo.diteration.y_relevance[cluster.members]):.2f}. "
              f"S: {sim:.2f}.")
    for c in cluster.children:
        print_feature_cluster(gismo, c, depth=depth + '-')

