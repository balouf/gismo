#!/usr/bin/env python

"""Tests for `gismo` package."""

from pytest import fixture

from sklearn.feature_extraction.text import CountVectorizer

from gismo.common import toy_source_dict
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.gismo import Gismo


@fixture()
def my_gismo():
    corpus = Corpus(toy_source_dict, lambda x: x['content'])
    vectorizer = CountVectorizer(dtype=float)
    embedding = Embedding(vectorizer=vectorizer)
    embedding.fit_transform(corpus)
    gismo = Gismo(corpus, embedding)
    gismo.rank("Gizmo")
    return gismo


def test_default_cluster_document_post(my_gismo):
    cluster = my_gismo.get_clustered_ranked_documents(k=5)
    assert f"{cluster['focus']:.2f}" == "0.05"
    assert len(cluster["children"]) == 2


def test_default_cluster_features_post(my_gismo):
    cluster = my_gismo.get_clustered_ranked_features(k=10)
    assert cluster['feature'] == "mogwaï"
    assert len(cluster["children"]) == 2
