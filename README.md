[![Gismo logo](https://github.com/balouf/gismo/raw/master/docs/logo-line.png)](https://balouf.github.io/gismo/)

# A Generic Information Search... With a Mind of its Own!


[![Pypi badge](https://img.shields.io/pypi/v/gismo.svg)](https://pypi.python.org/pypi/gismo)
[![Build badge](https://github.com/balouf/gismo/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/balouf/gismo/actions?query=workflow%3Abuild)
[![Documentation badge](https://github.com/balouf/gismo/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/balouf/gismo/actions?query=workflow%3Adocs)
[![Coverage](https://codecov.io/gh/balouf/gismo/branch/master/graphs/badge.svg)](https://codecov.io/gh/balouf/gismo/branch/master)


GISMO is a NLP tool to rank and organize a corpus of documents according to a query.

Gismo stands for Generic Information Search... with a Mind of its Own.

- Free software: MIT License
- Github: <https://github.com/balouf/gismo/>
- Documentation: <https://balouf.github.io/gismo/>

## Features

Gismo combines three main ideas:

- **TF-IDTF**: a symmetric version of the TF-IDF embedding.
- **DIteration**: a fast, push-based, variant of the PageRank algorithm.
- **Fuzzy dendrogram**: a variant of the Louvain clustering algorithm.

## Quickstart

Install gismo:

```console
$ pip install gismo
```

Import gismo in a Python project:

```
import gismo as gs
```

To get the hang of a typical Gismo workflow, you can check the [Toy Example] notebook. For more advanced uses,
look at the other [tutorials] or directly the [reference] section.

## Credits

Thomas Bonald, Anne Bouillard, Marc-Olivier Buob, Dohy Hong.

This package was created with [Cookiecutter] and the [francois-durand/package_helper] project template.

[cookiecutter]: https://github.com/audreyr/cookiecutter
[francois-durand/package_helper]: https://github.com/francois-durand/package_helper
[reference]: https://balouf.github.io/gismo/reference.html
[toy example]: https://balouf.github.io/gismo/tutorials/tutorial_toy_example.html
[tutorials]: https://balouf.github.io/gismo/tutorials/index.html#
