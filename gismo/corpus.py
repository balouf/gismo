#!/usr/bin/env python
# coding: utf-8
#
# GISMO: a Generic Information Search with a Mind of its Own


import numpy as np
from itertools import chain

from gismo.common import MixInIO, toy_source_text, toy_source_dict

class Corpus(MixInIO):
    """
    Corpus class, to feed to Embedding

    Examples
    --------
        >>> corpus = Corpus(toy_source_text, to_text=lambda x: f"{x[:15]}...")
        >>> for c in corpus.iterate():
        ...    print(c)
        Gizmo is a Mogwaï.
        This is a sentence about Blade.
        This is a sentence about Shadoks.
        This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.
        In chinese folklore, a Mogwaï is a demon.

        >>> for c in corpus.iterate_text():
        ...    print(c)
        Gizmo is a Mogw...
        This is a sente...
        This is a sente...
        This very long ...
        In chinese folk...
    """
    def __init__(self, source=None, to_text=None, filename=None):
        if filename is not None:
            self.load(filename)
        else:
            self.source = source
            self.i = 0
            self.n = 0 if source is None else len(source)
            if to_text is None:
                self.to_text = lambda x: x
            else:
                self.to_text = to_text

    def iterate_text(self, to_text=None):
        if to_text is None:
            to_text = self.to_text
        return (to_text(entry) for entry in self)

    def iterate(self):
        return (entry for entry in self.source)

    def __getitem__(self, i):
        return self.source[i]

    def __len__(self):
        return self.n

    def merge_new_source(self, new_source, doc2key=None):
        """
        Incorporate new entries from a source

        Parameters
        ----------
        new_source: list
                    source of the same type (same to_text mostly) that the current source
        doc2key: function
                 callback  that provides unique hashable Id for documents

        Examples
        --------
        >>> corpus = Corpus(toy_source_dict.copy(), to_text=lambda x: x['content'][:14])
        >>> len(corpus)
        5
        >>> new_corpus = [{"title": "Another document", "content": "I don't know what to say!"},
        ...     {'title': 'Fifth Document', 'content': 'In chinese folklore, a Mogwaï is a demon.'}]
        >>> corpus.merge_new_source(new_corpus, doc2key=lambda e: e['title'])
        >>> len(corpus)
        6
        >>> for c in corpus.iterate_text():
        ...    print(c)
        Gizmo is a Mog
        This is a sent
        This is a sent
        This very long
        In chinese fol
        I don't know w
        """
        if doc2key is None:
            print("Incremental corpus requires to provide a doc2key function")
            return self
        if self.source is None:
            self.source = []
        new_keys = {doc2key(d) for d in new_source} - {doc2key(d) for d in self.source}
        self.source += [d for d in new_source if doc2key(d) in new_keys]
        self.n = len(self.source)


class CorpusList(MixInIO):
    """
    To concatenate a list of corpi with distinct shapes and to_text

    Example
    -------
    >>> multi_corp = CorpusList([Corpus(toy_source_text, lambda x: x[:15]+"..."), Corpus(toy_source_dict, lambda e: e['title'])])
    >>> for c in multi_corp.iterate_text():
    ...    print(c)
    Gizmo is a Mogw...
    This is a sente...
    This is a sente...
    This very long ...
    In chinese folk...
    First Document
    Second Document
    Third Document
    Fourth Document
    Fifth Document
    """

    def __init__(self, corpus_list=None, filename=None):
        if filename is not None:
            self.load(filename)
        else:
            if corpus_list is None or len(corpus_list) == 0:
                print("Please provide a non-empty list of corpi!")
            else:
                self.corpus_list = corpus_list
                self.cum_n = np.cumsum([len(corpus) for corpus in self.corpus_list])
                self.n = self.cum_n[-1]

    def iterate(self):
        return chain.from_iterable([corpus.iterate() for corpus in self.corpus_list])

    def iterate_text(self):
        return chain.from_iterable([corpus.iterate_text() for corpus in self.corpus_list])

    def __getitem__(self, i):
        corpus_indice = np.searchsorted(self.cum_n, i, side='right')
        local_i = i if corpus_indice == 0 else (i - self.cum_n[corpus_indice - 1])
        return self.corpus_list[corpus_indice][local_i]

    def __len__(self):
        return self.n
