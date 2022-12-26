=========
Reference
=========

Gismo is made of multiple small modules designed to be mixed together.

* corpus_: The module contains simple wrappers to turn a wide range of document sources into something that Gismo will be able to process.
* embedding_: This module can create and manipulate TF-IDTF embeddings out of a corpus.
* diteration_: This module transforms queries into relevance vectors that can be used to rank and organize documents and features.
* clustering_: This module implements the tree-like organization of selected items
* gismo_: The main gismo module combines all modules above to provide high level, end-to-end, analysis methods.
* landmarks_: Introduced in v0.4, this high-level module allows deeper analysis of a small corpus by using individual query results for the embedding.
* `post processing`_: This module provides a simple, unified, way to apply automatic transformations (e.g. formatting) to the results of an analysis.
* filesource_: This module can be used to read documents one-by-one from disk instead of loading them all in memory. Useful for very large corpi.
* sentencizer_: This module can leverage a document-level gismo to provide sentence-level analysis. Can be used to extract key phrases (headlines).
* datasets_: Collection of access to small or less small datasets.
* common_: Multi-purpose module of things that can be used in more than one other module.
* parameters_: Management of runtime parameters.

Corpus
-------

.. autoclass:: gismo.corpus.Corpus
    :members:
.. autoclass:: gismo.corpus.CorpusList
    :members:

.. _embedding:

Embedding
--------------------

.. automodule:: gismo.embedding
    :members:

DIteration
------------------
.. automodule:: gismo.diteration
    :members:

Clustering
-----------------
.. automodule:: gismo.clustering
    :members:

Gismo
--------------
.. autoclass:: gismo.gismo.Gismo
    :members:

.. autoclass:: gismo.gismo.XGismo
    :members:

Landmarks
--------------

.. automodule:: gismo.landmarks
    :members:

Post Processing
-----------------
.. automodule:: gismo.post_processing
    :members:

FileSource
----------------
.. automodule:: gismo.filesource
    :members:


Sentencizer
----------------
.. automodule:: gismo.sentencizer
    :members:


Datasets
--------
.. automodule:: gismo.datasets.acm
    :members:

.. automodule:: gismo.datasets.dblp
    :members:

.. automodule:: gismo.datasets.reuters
    :members:

Common
------

.. automodule:: gismo.common
    :members:


Parameters
----------

.. automodule:: gismo.parameters
    :members:
