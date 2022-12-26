.. image:: https://github.com/balouf/gismo/raw/master/docs/logo-line.png
    :alt: Gismo logo
    :target: https://balouf.github.io/gismo/

--------------------------------------------------------
A Generic Information Search... With a Mind of its Own!
--------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/gismo.svg
        :target: https://pypi.python.org/pypi/gismo

.. image:: https://github.com/balouf/gismo/workflows/build/badge.svg?branch=master
        :target: https://github.com/balouf/gismo/actions?query=workflow%3Abuild
        :alt: Build Status

.. image:: https://github.com/balouf/gismo/workflows/docs/badge.svg?branch=master
        :target: https://github.com/balouf/gismo/actions?query=workflow%3Adocs
        :alt: Documentation Status

.. image:: https://codecov.io/gh/balouf/gismo/branch/master/graphs/badge.svg
        :target: https://codecov.io/gh/balouf/gismo/branch/master
        :alt: Code Coverage





GISMO is a NLP tool to rank and organize a corpus of documents according to a query.

Gismo stands for Generic Information Search... with a Mind of its Own.

* Free software: GNU General Public License v3
* Github: https://github.com/balouf/gismo/
* Documentation: https://balouf.github.io/gismo/


Features
--------

Gismo combines three main ideas:

* **TF-IDTF**: a symmetric version of the TF-IDF embedding.
* **DIteration**: a fast, push-based, variant of the PageRank algorithm.
* **Fuzzy dendrogram**: a variant of the Louvain clustering algorithm.

Quickstart
----------

Install gismo:

.. code-block:: console

    $ pip install gismo

Import gismo in a Python project::

    import gismo as gs


To get the hang of a typical Gismo workflow, you can check the `Toy Example`_ notebook. For more advanced uses,
look at the other tutorials_ or directly the reference_ section.



Credits
-------

Thomas Bonald, Anne Bouillard, Marc-Olivier Buob, Dohy Hong.

This package was created with Cookiecutter_ and the `francois-durand/package_helper`_ project template.

.. _reference: https://balouf.github.io/gismo/reference.html
.. _`Toy Example`: https://balouf.github.io/gismo/tutorials/tutorial_toy_example.html
.. _tutorials: https://balouf.github.io/gismo/tutorials/index.html#
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`francois-durand/package_helper`: https://github.com/francois-durand/package_helper
