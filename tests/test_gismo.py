#!/usr/bin/env python

"""Tests for `gismo` package."""

from pytest import fixture
import tempfile

from sklearn.feature_extraction.text import CountVectorizer

from gismo.common import toy_source_dict
from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.gismo import Gismo, XGismo


@fixture()
def my_gismo():
    corpus = Corpus(toy_source_dict, lambda x: x['content'])
    vectorizer = CountVectorizer(dtype=float)
    embedding = Embedding(vectorizer=vectorizer)
    embedding.fit_transform(corpus)
    gismo = Gismo(corpus, embedding)
    gismo.query_distortion = False
    gismo.rank("Gizmo")
    return gismo


def test_default_cluster_document_post(my_gismo):
    cluster = my_gismo.get_clustered_ranked_documents(k=5)
    assert f"{cluster['focus']:.2f}" == "0.04"
    assert len(cluster["children"]) == 2


def test_default_cluster_features_post(my_gismo):
    cluster = my_gismo.get_clustered_ranked_features(k=10)
    assert cluster['feature'] == "mogwaï"
    assert len(cluster["children"]) == 2


def test_default_no_post(my_gismo):
    indices = my_gismo.get_ranked_documents(k=3, post=False)
    assert list(indices) == [0, 3, 4]
    indices = my_gismo.get_ranked_features(post=False)
    assert list(indices) == [18, 10, 14]
    indices = my_gismo.get_covering_documents(k=3, post=False)
    assert list(indices) == [0, 3, 1]
    indices = my_gismo.get_covering_features(post=False)
    assert list(indices) == [18, 12, 10]


def test_io_gismo(my_gismo):
    with tempfile.TemporaryDirectory() as tmpdirname:
        my_gismo.save(filename="mygismo", path=tmpdirname)
        gismo2 = Gismo(filename="mygismo", path=tmpdirname)
    gismo2.rank("sentence")
    assert len(gismo2.get_ranked_documents()) == 2


def test_io_xgismo(my_gismo):
    xgismo = XGismo(my_gismo.embedding, my_gismo.embedding)
    with tempfile.TemporaryDirectory() as tmpdirname:
        xgismo.save(filename="mygismo", path=tmpdirname)
        xgismo2 = XGismo(filename="mygismo", path=tmpdirname)
    xgismo2.rank("gizmo")
    assert xgismo2.get_ranked_documents() == ['mogwaï', 'gizmo', 'is']

