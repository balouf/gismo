import tempfile

from gismo.common import toy_source_text
from gismo.corpus import Corpus
from gismo.embedding import Embedding


def test_embedding_io():
    corpus=Corpus(toy_source_text)
    embedding = Embedding()
    embedding.fit_transform(corpus)
    assert embedding.features[3] == 'demon'
    with tempfile.TemporaryDirectory() as tmp:
        embedding.dump(filename="test", path=tmp)
        new_embedding = Embedding.load(filename="test", path=tmp)
    assert new_embedding.features[3] == 'demon'
