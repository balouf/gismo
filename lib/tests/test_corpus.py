import tempfile

from gismo.common import toy_source_dict, toy_source_text
from gismo.corpus import Corpus, CorpusList


def test_corpus_merge_border_cases():
    corpus = Corpus()
    assert corpus.source is None
    assert type(corpus.merge_new_source(new_source=['a', 'b'])) == Corpus
    assert corpus.source is None
    corpus.merge_new_source(new_source=['a', 'b'], doc2key=lambda x: x)
    assert corpus.source == ['a', 'b']


def test_corpuslist_io():
    assert type(CorpusList()) == CorpusList
    multi_corp = CorpusList([Corpus(toy_source_text, lambda x: x[:15]+"..."),
                             Corpus(toy_source_dict, lambda e: e['title'])])
    with tempfile.TemporaryDirectory() as tmp:
        multi_corp.save(filename="test", path=tmp)
        new_corp = CorpusList(filename="test", path=tmp)
        assert len(new_corp) == 10
        assert [e for e in new_corp.iterate()][0] == 'Gizmo is a Mogwaï.'
