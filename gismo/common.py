#!/usr/bin/env python
# coding: utf-8
#
# GISMO: a Generic Information Search with a Mind of its Own

# DIteration parameters
ALPHA = .25  # diffusion attenuation
N_ITER = 4  # Number of round-trip diffusions
D_MAX = 10  # Maximum number of documents
K_MAX = 5  # Maximum number of keywords
S_MAX = 4  # Maximum number of sentences
F_MAX = 6  # Maximum number of features

# Hierarchical clustering parameters
STRETCH = 2  # Stretch factor that determines the coverage/relevance trade-off
C_MAX = 3  # Maximum depth of concepts
P_MAX = 4  # Maximum depth of document partitions

import gzip
import errno
import os
import dill as pickle

from pathlib import Path


class MixInIO:
    """
    Provide basic save/load capacities
    """
    def save(self, filename: str, path='.', erase=False, compress=False):
        """
        Save object to file
        :param filename: Name of the file (sufix will be ignored)
        :param path: File location
        :param erase: Authorize overwrite of previous file
        :param compress: use gzip to save space
        :return: None
        """
        if isinstance(path, str):
            path = Path(path)
        destination = path / Path(filename).stem
        if compress:
            destination = destination.with_suffix(".pkl.gz")
            if destination.exists() and not erase:
                print(f"File {destination} already exists!")
            else:
                with gzip.open(destination, "wb") as f:
                    pickle.dump(self, f)
        else:
            destination = destination.with_suffix(".pkl")
            if destination.exists() and not erase:
                print(f"File {destination} already exists!")
            else:
                with open(destination, "wb") as f:
                    pickle.dump(self, f)

    def load(self, filename: str, path='.'):
        """
        Load object from file
        :param filename: Name of the file (suffix will be ignored)
        :param path: File location
        :return: None
        """
        if isinstance(path, str):
            path = Path(path)
        dest = path / Path(filename).with_suffix(".pkl")
        if dest.exists():
            with open(dest, 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)
        else:
            dest = dest.with_suffix('.pkl.gz')
            if dest.exists():
                with gzip.open(dest) as f:
                    self.__dict__.update(pickle.load(f).__dict__)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), dest)


toy_source_text = ['Gizmo is a Mogwaï.',
                   'This is a sentence about Blade.',
                   'This is a sentence about Shadoks.',
                   'This very long sentence, with a lot of stuff about Star Wars inside, makes at some point a side reference to the Gremlins movie by comparing Gizmo and Yoda.',
                   'In chinese folklore, a Mogwaï is a demon.']

toy_source_dict = [{'title': 'First Document', 'content': 'Gizmo is a Mogwaï.'},
                   {'title': 'Second Document', 'content': 'This is a sentence about Blade.'},
                   {'title': 'Third Document', 'content': 'This is a sentence about Shadoks.'},
                   {'title': 'Fourth Document',
                    'content': 'This very long sentence, with a lot of stuff about Star Wars inside, '
                               'makes at some point a side reference to the Gremlins movie by '
                               'comparing Gizmo and Yoda.'},
                   {'title': 'Fifth Document', 'content': 'In chinese folklore, a Mogwaï is a demon.'}]
