import zlib
import json
import io
import dill as pickle
from pathlib import Path


class FileSource:
    """
    Yield a file corpus as a list. File corpus is made of two files:
    The *corpus*.data file contains the stacked items. Each item is compressed with zlib;
    The  *corpus*.index files contains the list of pointers to seek items in the data file

    Parameters
    ----------
    corpus_dir: str
                Location of the files
    corpus_name: str
                Stem of the file
    load_corpus: bool
                Should the data be loaded in RAM
    """
    def __init__(self, corpus_dir='.', corpus_name="dblp", load_corpus=False):
        if isinstance(corpus_dir, str):
            corpus_dir = Path(corpus_dir)
        index = corpus_dir / Path(f"{corpus_name}.index")
        data = corpus_dir / Path(f"{corpus_name}.data")
        # load index
        with open(index, "rb") as f:
            self.index = pickle.load(f)
        self.n = len(self.index) - 1
        if load_corpus:
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
