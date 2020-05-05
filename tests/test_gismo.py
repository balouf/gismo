#!/usr/bin/env python

"""Tests for `gismo` package."""

import pytest
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
    cluster = my_gismo.get_clustered_ranked_documents()
    assert f"{cluster['focus']:.2f}" == "0.05"
    assert len(cluster["children"]) == 2


def test_default_cluster_features_post(my_gismo):
    cluster = my_gismo.get_clustered_ranked_features()
    assert cluster['feature'] == "mogwaï"
    assert len(cluster["children"]) == 2


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
