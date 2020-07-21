from spacy.lang.en import English

from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.gismo import Gismo


class Sentencizer:
    """
    The Sentencizer class allows to refine a document-level gismo into a sentence-level gismo.
    A simple sentence extraction is proposed.
    For more complex usages, the class can provide a full :py:class:`~gismo.gismo.Gismo` instance that operates at
    sentence-level.

    Parameters
    ----------
    gismo: Gismo
        Document-level Gismo.

    Examples
    ---------
    We use the C50 Reuters dataset (5000 news paragraphs).

    >>> from gismo.datasets.reuters import get_reuters_news
    >>> corpus = Corpus(get_reuters_news(), to_text=lambda e: e['content'])
    >>> embedding = Embedding()
    >>> embedding.fit_transform(corpus)
    >>> gismo = Gismo(corpus, embedding)
    >>> sentencer = Sentencizer(gismo)

    First Example: run explicitly the query *Orange* at document-level,
    extract 4 covering sentences with narrow bfs.

    >>> success = gismo.rank("Orange")
    >>> sentencer.get_sentences(s=4, wide=False) # doctest: +NORMALIZE_WHITESPACE
    ['Snook says the all important average retained revenue per Orange subscriber will
    rise from around 442 pounds per year, partly because dominant telecoms player British
    Telecommunications last month raised the price of a call to Orange phones from its fixed lines.',
    'Analysts said that Orange shares had good upside potential after a rollercoaster ride in their
    short time on the market.',
    'Orange, which was floated last March at 205 pence per share, initially saw its stock slump to
    157.5 pence before recovering over the last few months to trade at 218 on Tuesday, a rise of four
    pence on the day.',
    'One-2-One and Orange ORA.L, which offer only digital services, are due to release their
    connection figures next week.']

    Second example: extract *Ericsson*-related sentences

    >>> sentencer.get_sentences(query="Ericsson") # doctest: +NORMALIZE_WHITESPACE
    ['These latest wins follow a recent $350 million contract win with Telefon AB L.M. Ericsson,
    bolstering its already strong activity in the contract manufacturing of telecommuncation and
    data communciation products, he said.',
    'The restraints are few in areas such as consumer products, while in sectors such as banking,
    distribution and insurance, foreign firms are kept on a very tight leash.',
    "The company also said it had told analysts in a briefing Tuesday of new contract wins with
    Ascend Communications Inc, Harris Corp's Communications unit and Philips Electronics NV.",
    'Pocket is the first from the high-priced 1996 auction known to have filed for bankruptcy
    protection.',
    'With Ascend in particular, he said the company would be manufacturing the company\\'s
    mainstream MAX TNT remote access network equipment. "']

    Third example: extract *Communications*-related sentences from a string.

    >>> txt = gismo.corpus[4517]['content']
    >>> sentencer.get_sentences(query="communications", txt=txt) # doctest: +NORMALIZE_WHITESPACE
    ["Privately-held Pocket's big creditors include a group of Asian entrepreneurs and
    communications-equipment makers Siemens AG of Germany and L.M. Ericsson of Sweden.",
    "2 bidder at the government's high-flying wireless phone auction last year has filed for
    bankruptcy protection from its creditors, underscoring the problems besetting the
    auction's winners.",
    "The Federal Communications Commission on Monday gave PCS companies from last year's
    auction some breathing space when it suspended indefinitely a March 31 deadline for
    them to make payments to the agency for their licenses."]
    """

    def __init__(self, gismo):
        self.parser = English()
        self.parser.add_pipe(self.parser.create_pipe('sentencizer'))
        self.doc_gismo = gismo
        self.sent_corpus = None
        self.sent_gismo = None

    def splitter(self, txt):
        """
        Transform input content into a corpus of sentences stored into the :py:attr:`sent_corpus` attribute.

        Parameters
        ----------
        txt: str or list
            Text or list of documents to split in sentences. For the latter, documents are assumed
            to be provided as `(content, id)` pairs, where `content` is the actual text and `id` a
            reference of the document.

        Returns
        -------
        Sentencizer
        """
        if type(txt) is str:
            source = [str(sent).strip() for sent in self.parser(txt).sents if len(sent) > 10]
            self.sent_corpus = Corpus(source, to_text=lambda x: x)
        else:
            source = [{'source': source[1],
                       'content': str(sent).strip()} for source in txt
                      for sent in self.parser(source[0]).sents if len(sent) > 10]
            self.sent_corpus = Corpus(source, to_text=lambda x: x['content'])
        return self

    def make_sent_gismo(self, query=None, txt=None, k=None, **kwargs):
        """
        Construct a sentence-level Gismo stored in the :py:attr:`sent_gismo` attribute.

        Parameters
        ----------
        query: str (optional)
            Query to run on the document-level Gismo.
        txt: str (optional)
            Text to use for sentence extraction.
            If not set, the sentences will be extracted from the top-documents.
        k: int (optional)
            Number of top-documents used for the built.
            If not set, the :py:func:`~gismo.common.auto_k` heuristic will be used.
        kwargs: dict
            Custom default runtime parameters to pass to the sentence-level Gismo.
            You just need to specify the parameters that differ from :obj:`~gismo.parameters.DEFAULT_PARAMETERS`.
            Note that distortion will be automatically de-activated. If you really want it, manually change the value
            of ``self.sent_gismo.parameters.distortion`` afterwards.


        Returns
        -------
        Sentencizer
        """
        if txt is None:
            if query is not None:
                self.doc_gismo.rank(query)
            txt = [(self.doc_gismo.corpus.to_text(self.doc_gismo.corpus[i]), i)
                   for i in self.doc_gismo.get_documents_by_rank(k, post=False)]
        self.splitter(txt)
        local_embedding = Embedding()
        local_embedding.fit_ext(self.doc_gismo.embedding)
        local_embedding.transform(self.sent_corpus)
        self.sent_gismo = Gismo(self.sent_corpus, local_embedding, **kwargs)
        self.sent_gismo.parameters.distortion = 0.0
        self.sent_gismo.post_documents_item = lambda g, i: g.corpus.to_text(g.corpus[i])
        return self

    def get_sentences(self, query=None, txt=None, k=None, s=None,
                      resolution=.7, stretch=2.0, wide=True, post=True):
        """
        All-in-one method to extract covering sentences from the corpus.
        Computes sentence-level corpus, sentence-level gismo,
        and calls :py:meth:`~gismo.gismo.Gismo.get_documents_by_coverage`.

        Parameters
        ----------
        query: str (optional)
            Query to run on the document-level Gismo
        txt: str (optional)
            Text to use for sentence extraction.
            If not set, the sentences will be extracted from the top-documents.
        k: int (optional)
            Number of top-documents used for the built.
            If not set, the :py:func:`~gismo.common.auto_k` heuristic of the document-level Gismo will be used.
        s: int (optional)
            Number of sentences to return.
            If not set, the :py:func:`~gismo.common.auto_k` heuristic of the sentence-level Gismo will be used.
        resolution: float (optional)
            Tree resolution passed to the :py:meth:`~gismo.gismo.Gismo.get_documents_by_coverage` method.
        stretch: float >= 1 (optional)
            Stretch factor passed to the :py:meth:`~gismo.gismo.Gismo.get_documents_by_coverage` method.
        wide: bool (optional)
            bfs wideness passed to the :py:meth:`~gismo.gismo.Gismo.get_documents_by_coverage` method.
        post: bool (optional)
            Use of post-proccessing passed to the :py:meth:`~gismo.gismo.Gismo.get_documents_by_coverage` method.

        Returns
        -------
        list
        """

        self.make_sent_gismo(query=query, txt=txt, k=k)
        if query is None:
            query = self.doc_gismo.embedding._query
        self.sent_gismo.rank(query)
        return self.sent_gismo.get_documents_by_coverage(k=s, resolution=resolution,
                                                         stretch=stretch, wide=wide, post=post)
