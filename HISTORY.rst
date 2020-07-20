=======
History
=======

X.X.X (TODO-List)
-----------------
* re-working distortion on both vectors normalization and IDTF/query trade-off.

0.4.0 (2020-07-09)
------------------
* Landmarks! Half Corpus, half Gismo, the Landmarks class can simplify many analysis tasks.
    * Landmarks are (small) corpus where each entry is augmented with the computation of an associated gismo query;
    * Landmarks can be used to refine the analysis around a part of your data;
    * They can be used as soft and fast classifiers.
* Documentation summer cleaning.
* New API for Gismo runtime parameters (see new parameters module for details). Short version:
    * ``gismo = Gismo(corpus, embedding, alpha=0.85)``: create a gismo with damping factor set to 0.85 instead of default value.
    * ``gismo.parameters.alpha = 0.85``: set the damping factor of the gismo to 0.85.
    * ``gismo.rank(query, alpha=0.85)``: makes a query with damping factor temporarily set to 0.85.
* ``query_distortion`` parameter (reshape subspace for clustering) is renamed ``distortion`` and is now a float instead of a bool (e.g. you can apply distortion in a non-binary way).

0.3.1 (2020-06-12)
------------------

* New dataset: Reuters C50
* New module: sentencizer


0.3.0 (2020-05-13)
------------------

* dblp module: url2source function added to directly load a small dblp source in memory instead of using a FileSource approach.
* Possibility to disable query distortion in gismo.
* XGismo class to cross analyze embeddings.
* Tutorials updated

0.2.5 (2020-05-11)
------------------

* auto_k feature: if not specified, a query-dependent, reasonable, number of results k is estimated.
* covering methods added to gismo. It is now possible to use get_covering_* instead of get_ranked_* to maximize coverage and/or eliminate redundancy.


0.2.4 (2020-05-07)
------------------

* Tutorials for ACM and DBLP added. After cleaning, there is currently 3 tutorials:
    * Toy model, to get the hang of Gismo on a tiny example,
    * ACM, to play with Gismo on a small example,
    * DBLP, to play with a large dataset.


0.2.3 (2020-05-04)
------------------

* ACM and DBLP dataset creation added.


0.2.2 (2020-05-04)
------------------

* Notebook tutorials added (early version)

0.2.1 (2020-05-03)
------------------

* Actual code
* Coverage badge

0.1.0 (2020-04-30)
------------------

* First release on PyPI.
