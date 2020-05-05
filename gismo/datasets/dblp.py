from pathlib import Path
import requests
import zlib
import json
import dill as pickle
import numpy as np
import gzip
from lxml import etree
from io import StringIO


URL = "https://dblp.uni-trier.de/xml/dblp.xml.gz"


# load the DTD parser
r = requests.get("https://dblp.uni-trier.de/xml/dblp.dtd")
with StringIO(r.text) as g:
    dtd = etree.DTD(file=g)


def fast_iter(context, func, *args, **kwargs):
    """
    Applies ``func`` to all xml elements of depth 1 of the xml parser ``context``. ``*args`` and ``**kwargs`` are passed to ``func``.

    Modified version of a modified version of Liza Daly's fast_iter
    Inspired by
    https://stackoverflow.com/questions/4695826/efficient-way-to-iterate-through-xml-elements

    Parameters
    ----------
    context: XMLparser
        A parser obtained from etree.iterparse
    func: function
        How to process the elements
    """
    depth = 0
    for event, elem in context:
        if event=='start':
            depth += 1
        if event=='end':
            depth -= 1
            if depth < 2:
                func(elem, *args, **kwargs)
                # It's safe to call clear() here because no descendants will be
                # accessed
                elem.clear()
                # Also eliminate now-empty references from the root node to elem
                for ancestor in elem.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]
    del context

def process_element(elt, data_handler, index):
    """
    Converts the xml element ``elt`` into a dict.
    If it is a paper (at least one author, a venue, a year, a title):
        * Compress and write the dict in ``data_handler``
        * Append file position in ``data_handler`` to ``index``.

    Parameters
    ----------
    elt: Any
        a XML element.
    data_handler: file_descriptor
        Where the compressed data will be stored. Must be writable.
    index:
        a list that contains the initial position of the data_handler for all previously processed elements.

    Returns
    -------
    True
    """
    children = elt.getchildren()
    if not children:
        return True
    dic = {"type": elt.tag}
    authors = []
    for c in children:
        if c.tag=="author":
            authors.append(c.text)
        elif c.tag in {"journal", "booktitle"}:
            dic["venue"] = c.text
        elif c.tag in {"year", "title"}:
            dic[c.tag]=c.text
    dic['authors'] = authors
    if dic['authors']==[] or not all(key in dic for key in ['year', 'title', 'venue']):
        return True
    data_handler.write(zlib.compress(json.dumps(dic).encode('utf8')))
    index.append(data_handler.tell())
    return True


class Dblp:
    """
    The DBLP class can download DBLP database and produce source files compatible with the FileSource class.

    Parameters
    ----------
    dblp_url: str, optional
        Alternative URL for the dblp.xml.gz file
    filename: str
        Stem of the files (suffixes will be appened)
    path: str or path, optional
            Destination of the files
    """
    def __init__(self, dblp_url=URL, filename="dblp",
                 path="."):
        self.dblp_url = dblp_url
        if isinstance(path, str):
            self.path = Path(path)
        elif isinstance(path, Path):
            self.path = path
        else:
            self.path = Path(".")
        self.dblp_xml = self.path / Path(f"{filename}.xml.gz")
        self.dblp_data = self.path / Path(f"{filename}.data")
        self.dblp_index = self.path / Path(f"{filename}.index")
        self.xml_handler = None
        self.json_handler = None
        self._index = None

    def download(self):
        r = requests.get(self.dblp_url, stream=True)
        if self.dblp_url.endswith("gz"):
            with open(self.dblp_xml, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        else:
            with gzip.open(self.dblp_xml, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        print(f"DBLP database downloaded to {self.dblp_xml}.")

    def build(self, refresh=False):
        """
        Main class method. Create the data and index files.

        Parameters
        ----------
        refresh: str
            Tell if files are to be rebuilt if they are already there.

        Example
        -------
        By default, the class downloads the full dataset. Here we will limit to one entry.

        >>> toy_url = "https://dblp.org/rec/xml/conf/teletraffic/BouillardCPM18.xml"
        >>> import tempfile
        >>> from gismo.filesource import FileSource
        >>> tmp = tempfile.TemporaryDirectory()
        >>> dblp = Dblp(dblp_url=toy_url, path=tmp.name)
        >>> dblp.build() # doctest.ELLIPSIS
        Retrieve https://dblp.org/rec/xml/conf/teletraffic/BouillardCPM18.xml from the Internet.
        DBLP database downloaded to ...xml.gz.
        Converting DBLP database from ...xml.gz (may take a while).
        Building Index.
        Conversion done.

        By default, build uses existing files.

        >>> dblp.build() # doctest.ELLIPSIS
        File ...xml.gz already exists.
        File ...data already exists.

        The refresh parameter can be used to ignore existing files

        >>> dblp.build(refresh=True) # doctest.ELLIPSIS
        Retrieve https://dblp.org/rec/xml/conf/teletraffic/BouillardCPM18.xml from the Internet.
        DBLP database downloaded to ...xml.gz.
        Converting DBLP database from ...xml.gz (may take a while).
        Building Index.
        Conversion done.

        The resulting files can be used to create a FileSource.

        >>> source = FileSource(filename="dblp", path=tmp.name)
        >>> print(source[0]['authors']) # doctest.ELLIPSIS
        ['Anne Bouillard', 'Céline Comte', 'Elie de Panafieu', 'Fabien Mathieu']

        Don't forget to close source after use.

        >>> source.close()
        >>> tmp.cleanup()
        """
        if self.dblp_xml.exists() and not refresh:
            print(f"File {self.dblp_xml} already exists.")
        else:
            print((f"Retrieve {self.dblp_url} from the Internet."))
            self.download()
        if self.dblp_data.exists() and not refresh:
            print(f"File {self.dblp_data} already exists.")
        else:
            print(f"Converting DBLP database from {self.dblp_xml} (may take a while).")

            with gzip.open(self.dblp_xml, "rb") as f:
                index = [0]
                with open(self.dblp_data, "wb") as g:
                    context=etree.iterparse(f, events=('start', 'end',), load_dtd=True)
                    elem=fast_iter(context, process_element, data_handler=g, index=index)
                print(f"Building Index.")
                with open(self.dblp_index, "wb") as g:
                    pickle.dump(np.array(index), g)
                print(f"Conversion done.")
