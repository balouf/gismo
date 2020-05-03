=====
GISMO
=====


.. image:: https://img.shields.io/pypi/v/gismo.svg
        :target: https://pypi.python.org/pypi/gismo

.. image:: https://img.shields.io/travis/balouf/gismo.svg
        :target: https://travis-ci.org/balouf/gismo

.. image:: https://readthedocs.org/projects/gismo/badge/?version=latest
        :target: https://gismo.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://codecov.io/gh/balouf/gismo/branch/master/graphs/badge.svg
        :target: https://codecov.io/gh/balouf/gismo/branch/master/graphs/badge
        :alt: Code Coverage





GISMO is a NLP tool to rank and organize a corpus of documents according to a query.

Gismo stands for Generic Information Search... with a Mind of its Own.

* Free software: GNU General Public License v3
* Github: https://github.com/balouf/gismo.
* Documentation: https://gismo.readthedocs.io.


Features
--------

Gismo combines three main ideas:

* **TF-IDTF**: a symmetric version of the TF-IDF embedding.
* **DI-Iteration**: a fast, push-based, variant of the PageRank algorithm.
* **Fuzzy dendrogram**: a variant of the Louvain clustring algorithm.

Quickstart
----------

Install gismo:

.. code-block:: console

    $ pip install gismo

Import gismo in a Python project::

    import gismo as gs


Credits
-------

Thomas Bonald, Anne Bouillard, Marc-Olivier Buob, Dohy Hong.

This package was created with Cookiecutter_ and the `francois-durand/package_helper`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`francois-durand/package_helper`: https://github.com/francois-durand/package_helper
