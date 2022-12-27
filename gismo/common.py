#!/usr/bin/env python
# coding: utf-8
#
# GISMO: a Generic Information Search with a Mind of its Own

import gzip
import errno
import os
import dill as pickle
import numpy as np

from pathlib import Path


class MixInIO:
    """
    Provide basic save/load capacities to other classes.
    """

    def dump(self, filename: str, path='.', overwrite=False, compress=True):
        """
        Save instance to file.

        Parameters
        ----------
        filename: str
            The stem of the filename.
        path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
            The location path.
        overwrite: bool
            Should existing file be erased if it exists?
        compress: bool
            Should gzip compression be used?

        Examples
        ----------

        >>> import tempfile
        >>> v1 = ToyClass(42)
        >>> v2 = ToyClass()
        >>> v2.value
        0
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', compress=True, path=tmpdirname)
        ...     dir_content = [file.name for file in Path(tmpdirname).glob('*')]
        ...     v2 = ToyClass.load(filename='myfile', path=Path(tmpdirname))
        ...     v1.dump(filename='myfile', compress=True, path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl.gz already exists! Use overwrite option to overwrite.
        >>> dir_content
        ['myfile.pkl.gz']
        >>> v2.value
        42

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', compress=False, path=tmpdirname)
        ...     v1.dump(filename='myfile', compress=False, path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl already exists! Use overwrite option to overwrite.

        >>> v1.value = 51
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', path=tmpdirname, compress=False)
        ...     v1.dump(filename='myfile', path=tmpdirname, overwrite=True, compress=False)
        ...     v2 = ToyClass.load(filename='myfile', path=tmpdirname)
        ...     dir_content = [file.name for file in Path(tmpdirname).glob('*')]
        >>> dir_content
        ['myfile.pkl']
        >>> v2.value
        51

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...    v2 = ToyClass.load(filename='thisfilenamedoesnotexist')
        Traceback (most recent call last):
         ...
        FileNotFoundError: [Errno 2] No such file or directory: ...
        """
        path = Path(path)
        destination = path / Path(filename).stem
        if compress:
            destination = destination.with_suffix(".pkl.gz")
            if destination.exists() and not overwrite:
                print(f"File {destination} already exists! Use overwrite option to overwrite.")
            else:
                with gzip.open(destination, "wb") as f:
                    pickle.dump(self, f)
        else:
            destination = destination.with_suffix(".pkl")
            if destination.exists() and not overwrite:
                print(f"File {destination} already exists! Use overwrite option to overwrite.")
            else:
                with open(destination, "wb") as f:
                    pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str, path='.'):
        """
        Load instance from file.

        Parameters
        ----------
        filename: str
            The stem of the filename.
        path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
            The location path.
        """
        path = Path(path)
        dest = path / Path(filename).with_suffix(".pkl")
        if dest.exists():
            with open(dest, 'rb') as f:
                return pickle.load(f)
        else:
            dest = dest.with_suffix('.pkl.gz')
            if dest.exists():
                with gzip.open(dest) as f:
                    return pickle.load(f)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), dest)


class ToyClass(MixInIO):
    def __init__(self, value=0):
        self.value = value


def auto_k(data, order=None, max_k=100, target=1.0):
    """
    Proposes a threshold k of significant values according to a relevance vector.

    Parameters
    ----------
    data: :class:`~numpy.ndarray`
        Vector with positive relevance values.
    order: list of int, optional
        Ordered indices of ``data``
    max_k: int
        Maximal number of entries to return; also number of entries used to determine threshold.
    target: float
        Threshold modulation. Higher target means less result.
        A target set to 1.0 corresponds to using the average of the max_k top values as threshold.

    Returns
    -------
    k: int
        Recommended number of values.

    Example
    --------
    >>> data = np.array([30, 1, 2, .3, 4, 50, 80])
    >>> auto_k(data)
    3
    """
    if order is None:
        order = np.argsort(-data)
    ordered_data = data[order[:max_k]]
    max_k = min(max_k, len(data))
    threshold = np.sum(ordered_data)*target/max_k
    k = int(np.sum(ordered_data >= threshold))
    return max(1, k)


toy_source_text = ['Gizmo is a Mogwaï.',
                   'This is a sentence about Blade.',
                   'This is another sentence about Shadoks.',
                   'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side '
                   'reference to the Gremlins movie by comparing Gizmo and Yoda.',
                   'In chinese folklore, a Mogwaï is a demon.']
"""A minimal source example where items are :py:obj:`str`."""

toy_source_dict = [{'title': 'First Document', 'content': 'Gizmo is a Mogwaï.'},
                   {'title': 'Second Document', 'content': 'This is a sentence about Blade.'},
                   {'title': 'Third Document', 'content': 'This is another sentence about Shadoks.'},
                   {'title': 'Fourth Document',
                    'content': 'This very long sentence, with a lot of stuff about Star Wars inside, '
                               'makes at some point a side reference to the Gremlins movie by '
                               'comparing Gizmo and Yoda.'},
                   {'title': 'Fifth Document', 'content': 'In chinese folklore, a Mogwaï is a demon.'}]
"""A minimal source example where items are :py:obj:`dict` with keys `title` and `content`."""
