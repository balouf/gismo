from sklearn.metrics.pairwise import cosine_similarity


def post_documents_item_raw(gismo, i):
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
    object
        Document i from corpus
    """
    return gismo.corpus[i]


def post_documents_item_content(gismo, i, max_size=None):
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


def post_features_item_raw(gismo, i):
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


def post_documents_cluster_json(gismo, cluster):
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
            'children': [post_documents_cluster_json(gismo, c) for c in cluster.children]}


def post_documents_cluster_print(gismo, cluster, depth=""):
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
        post_documents_cluster_print(gismo, c, depth=depth + '-')


def post_features_cluster_json(gismo, cluster):
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
            'children': [post_features_cluster_json(gismo, c) for c in cluster.children]}


def post_features_cluster_print(gismo, cluster, depth=""):
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
        post_features_cluster_print(gismo, c, depth=depth + '-')


def post_landmarks_item_raw(landmark, i):
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


def post_landmarks_cluster_json(landmark, cluster):
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
            'children': [post_landmarks_cluster_json(landmark, child) for child in cluster.children]}
