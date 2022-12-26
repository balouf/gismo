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
    gismo.parameters.distortion = 0.0
    gismo.rank("Gizmo")
    return gismo


def test_default_cluster_document_post(my_gismo):
    cluster = my_gismo.get_documents_by_cluster(k=5)
    assert f"{cluster['focus']:.2f}" == "0.04"
    assert len(cluster["children"]) == 2


def test_default_cluster_features_post(my_gismo):
    cluster = my_gismo.get_features_by_cluster(k=10)
    assert cluster['feature'] == "mogwaï"
    assert len(cluster["children"]) == 2


def test_default_no_post(my_gismo):
    indices = my_gismo.get_documents_by_rank(k=3, post=False)
    assert list(indices) == [0, 3, 4]
    indices = my_gismo.get_features_by_rank(post=False)
    assert list(indices) == [18, 10, 14]
    indices = my_gismo.get_documents_by_coverage(k=3, post=False)
    assert list(indices) == [0, 3, 1]
    indices = my_gismo.get_features_by_coverage(post=False)
    assert list(indices) == [18, 12, 10]


def test_io_gismo(my_gismo):
    with tempfile.TemporaryDirectory() as tmpdirname:
        my_gismo.save(filename="mygismo", path=tmpdirname)
        gismo2 = Gismo(filename="mygismo", path=tmpdirname)
    gismo2.rank("sentence")
    assert len(gismo2.get_documents_by_rank()) == 2


def test_io_xgismo(my_gismo):
    xgismo = XGismo(my_gismo.embedding, my_gismo.embedding)
    with tempfile.TemporaryDirectory() as tmpdirname:
        xgismo.save(filename="mygismo", path=tmpdirname)
        xgismo2 = XGismo(filename="mygismo", path=tmpdirname)
    xgismo2.rank("gizmo")
    assert xgismo2.get_documents_by_rank() == ['mogwaï', 'gizmo', 'is']

