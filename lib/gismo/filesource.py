import zlib
import json
import io
import dill as pickle
import numpy as np
from pathlib import Path
from gismo.corpus import toy_source_dict


def create_file_source(source=None, filename='mysource', path='.'):
    """
    Write a source (list of dict) to files in the same format used by FileSource. Only useful
    to transfer from a computer with a lot of RAM to a computer with less RAM. For more complex cases,
    e.g. when the initial source itself is a very large file, a dedicated converter has to be provided.

    Parameters
    ----------
    source: list of dict
        The source to write
    filename: str
        Stem of the file. Two files will be created, with suffixes *.index* and *.data*.
    path: str or Path
        Destination directory
    """
    if source is None:
        source = toy_source_dict
    path = Path(path)
    data_file = path / Path(f"{filename}.data")
    index_file = path / Path(f"{filename}.index")
    indices = [0]
    with open(data_file, "wb") as f:
        for item in source:
            f.write(zlib.compress(json.dumps(item).encode('utf8')))
            indices.append(f.tell())
    with open(index_file, "wb") as f:
        pickle.dump(indices, f)


class FileSource:
    """
    Yield a file source as a list. Assumes the existence of two files:
    The *mysource*.data file contains the stacked items. Each item is compressed with :py:mod:`zlib`;
    The  *mysource*.index files contains the list of pointers to seek items in the data file.

    The resulting source object is fully compatible with the :class:`~gismo.corpus.Corpus` class:
     * It can be iterated (``[item for item in source]``);
     * It can yield single items (``source[i]``);
     * It has a length (``len(source)``).

    More advanced functionalities like slices are not implemented.

    Parameters
    ----------
    path: str
                Location of the files
    filename: str
                Stem of the file
    load_source: bool
                Should the data be loaded in RAM

    Examples
    ---------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as dirname:
    ...    create_file_source(filename='mysource', path=dirname)
    ...    source = FileSource(filename='mysource', path=dirname, load_source=True)
    ...    content = [e['content'] for e in source]
    >>> content[:3]
    ['Gizmo is a Mogwaï.', 'This is a sentence about Blade.', 'This is another sentence about Shadoks.']

    Note: when source is read from file (``load_source=False``, default behavior), you need to close the source afterwards
    to avoid pending file handles.

    >>> with tempfile.TemporaryDirectory() as dirname:
    ...    create_file_source(filename='mysource', path=dirname)
    ...    source = FileSource(filename='mysource', path=dirname)
    ...    size = len(source)
    ...    item = source[0]
    ...    source.close()
    >>> size
    5
    >>> item
    {'title': 'First Document', 'content': 'Gizmo is a Mogwaï.'}
    """
    def __init__(self, filename="mysource", path='.', load_source=False):
        path = Path(path)
        index = path / Path(f"{filename}.index")
        data = path / Path(f"{filename}.data")
        # load index
        with open(index, "rb") as f:
            self.index = pickle.load(f)
        self.n = len(self.index) - 1
        if load_source:
            with open(data, "rb") as f:
                self.f = io.BytesIO(f.read())
        else:
            self.f = open(data, "rb")

    def __getitem__(self, i):
        self.f.seek(self.index[i])
        line = zlib.decompress(self.f.read(self.index[i + 1] - self.index[i])).decode('utf8')
        return json.loads(line)

    def __iter__(self):
        self.i = 0
        self.f.seek(0)
        return self

    def __next__(self):
        if self.i == self.n:
            raise StopIteration
        line = zlib.decompress(self.f.read(self.index[self.i + 1] - self.index[self.i])).decode('utf8')
        self.i += 1
        return json.loads(line)

    def __len__(self):
        return self.n

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
