import io
from pathlib import Path
import requests
import zlib
import json
import dill as pickle
import numpy as np
import gzip
from lxml import etree


URL = "https://dblp.uni-trier.de/xml/dblp.xml.gz"
DTD_URL = "https://dblp.uni-trier.de/xml/dblp.dtd"


def fast_iter(context, func, d=2, **kwargs):
    """
    Applies ``func`` to all xml elements of depth 1 of the xml parser ``context``. `
    ``**kwargs`` are passed to ``func``.

    Modified version of a modified version of Liza Daly's fast_iter
    Inspired by
    https://stackoverflow.com/questions/4695826/efficient-way-to-iterate-through-xml-elements

    Parameters
    ----------
    context: XMLparser
        A parser obtained from etree.iterparse
    func: function
        How to process the elements
    d: int, optional
        Depth to process elements.
    """
    depth = 0
    for event, elem in context:
        if event == 'start':
            depth += 1
        if event == 'end':
            depth -= 1
            if depth < d:
                func(elem, **kwargs)
                # It's safe to call clear() here because no descendants will be
                # accessed
                elem.clear()
                # Also eliminate now-empty references from the root node to elem
                for ancestor in elem.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]
    del context


def xml_element_to_dict(elt):
    """
    Converts the xml element ``elt`` into a dict if it is a paper.

    Parameters
    ----------
    elt: Any
        a XML element.

    Returns
    -------
    dict or None
        Article dictionary if element contains the attributes of an article, None otherwise.
    """
    children = elt.getchildren()
    if not children:
        return None
    dic = {"type": elt.tag}
    authors = []
    for c in children:
        if not isinstance(c.text, str):
            continue
        if c.tag == "author":
            authors.append(c.text)
        elif c.tag in {"journal", "booktitle"}:
            dic["venue"] = c.text
        elif c.tag in {"year", "title"}:
            dic[c.tag] = c.text
    dic['authors'] = authors
    if dic['authors'] == [] or not all(key in dic for key in ['year', 'title', 'venue']):
        return None
    return dic


def element_to_source(elt, source):
    """
    Test if elt is an article, converts it to dictionary and appends to source

    Parameters
    ----------
    elt: Any
        a XML element
    source: list
        the source in construction
    """
    dic = xml_element_to_dict(elt)
    if dic is not None:
        source.append(dic)


def url2source(url):
    """
    Directly transform URL of a dblp xml into a list of dictionnary.
    Only use for datasets that fit into memory (e.g. articles from one author).
    If the dataset does not fit, consider using the Dblp class instead.

    Parameters
    ----------
    url: str
        the URL to fetch

    Returns
    -------
    source: list of dict
        Articles retrieved from the URL

    Example
    -------
    >>> source = url2source("https://dblp.org/pers/xx/t/Tixeuil:S=eacute=bastien.xml")
    >>> art = [s for s in source if s['title']=="Distributed Computing with Mobile Robots: An Introductory Survey."][0]
    >>> art['authors']
    ['Maria Potop-Butucaru', 'Michel Raynal', 'Sébastien Tixeuil']
    """
    r = requests.get(url)
    source = []
    with io.BytesIO(r.content) as f:
        context = etree.iterparse(f, events=('start', 'end',))
        fast_iter(context, element_to_source, d=3, source=source)
    return source


def element_to_filesource(elt, data_handler, index):
    """
    * Converts the xml element ``elt`` into a dict if it is an article.
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
    dic = xml_element_to_dict(elt=elt)
    if dic is None:
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
        self.path = Path(path)
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

    def build(self, refresh=False, d=2):
        """
        Main class method. Create the data and index files.

        Parameters
        ----------
        refresh: bool
            Tell if files are to be rebuilt if they are already there.
        d: int
            depth level where articles are. Usually 2 or 3 (2 for the main database).

        Example
        -------
        By default, the class downloads the full dataset. Here we will limit to one entry.

        >>> toy_url = "https://dblp.org/pers/xx/m/Mathieu:Fabien.xml"
        >>> import tempfile
        >>> from gismo.filesource import FileSource
        >>> tmp = tempfile.TemporaryDirectory()
        >>> dblp = Dblp(dblp_url=toy_url, path=tmp.name)
        >>> dblp.build() # doctest.ELLIPSIS
        Retrieve https://dblp.org/pers/xx/m/Mathieu:Fabien.xml from the Internet.
        DBLP database downloaded to ...xml.gz.
        Converting DBLP database from ...xml.gz (may take a while).
        Building Index.
        Conversion done.

        By default, build uses existing files.

        >>> dblp.build() # doctest.ELLIPSIS
        File ...xml.gz already exists.
        File ...data already exists.

        The refresh parameter can be used to ignore existing files.

        >>> dblp.build(d=3, refresh=True) # doctest.ELLIPSIS
        Retrieve https://dblp.org/pers/xx/m/Mathieu:Fabien.xml from the Internet.
        DBLP database downloaded to ...xml.gz.
        Converting DBLP database from ...xml.gz (may take a while).
        Building Index.
        Conversion done.

        The resulting files can be used to create a FileSource.

        >>> source = FileSource(filename="dblp", path=tmp.name)
        >>> art = [s for s in source if s['title']=="Can P2P networks be super-scalable?"][0]
        >>> art['authors'] # doctest.ELLIPSIS
        ['François Baccelli', 'Fabien Mathieu', 'Ilkka Norros', 'Rémi Varloot']

        Don't forget to close source after use.

        >>> source.close()
        >>> tmp.cleanup()
        """
        if self.dblp_xml.exists() and not refresh:
            print(f"File {self.dblp_xml} already exists.")
        else:
            print(f"Retrieve {self.dblp_url} from the Internet.")
            self.download()
        if self.dblp_data.exists() and not refresh:
            print(f"File {self.dblp_data} already exists.")
        else:
            print(f"Converting DBLP database from {self.dblp_xml} (may take a while).")
            # Download the DTD parser
            r = requests.get("https://dblp.uni-trier.de/xml/dblp.dtd")
            with open(self.path / Path("dblp.dtd"), 'w') as f:
                f.write(r.text)

            with gzip.open(self.dblp_xml, "rb") as f:
                index = [0]
                with open(self.dblp_data, "wb") as g:
                    context = etree.iterparse(f, events=('start', 'end',), load_dtd=True)
                    fast_iter(context, element_to_filesource, d=d, data_handler=g, index=index)
                print(f"Building Index.")
                with open(self.dblp_index, "wb") as g:
                    pickle.dump(np.array(index), g)
                print(f"Conversion done.")
