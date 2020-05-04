from pathlib import Path
import requests
import zlib
import json
import xmltodict
import dill as pickle
import numpy as np
import gzip

URL = "https://dblp.uni-trier.de/xml/dblp.xml.gz"


class Dblp:
    """
    The DBLP class can download DBLP database and produce source files compatible with the FileSource class.

    Parameters
    ----------
    dblp_url: str, optional
            Alternative URL for the dblp.xml.gz file
    path: str or path, optional
            Destination of the source files
    """

    def __init__(self, dblp_url=URL,
                 path="."):
        self.dblp_url = dblp_url
        if isinstance(path, str):
            self.path = Path(path)
        elif isinstance(path, Path):
            self.path = path
        else:
            self.path = Path(".")
        self.dblp_xml = self.path / Path("dblp.xml.gz")
        self.dblp_json = self.path / Path("dblp.data")
        self.dblp_index = self.path / Path("dblp.index")
        self.xml_handler = None
        self.json_handler = None
        self._index = None

    def download(self):
        print(f"Downloading DBLP database from {self.dblp_url}.")
        r = requests.get(self.dblp_url, stream=True)
        with open(self.dblp_xml, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        print(f"DBLP database downloaded to {self.dblp_xml}.")

    @staticmethod
    def author2str(a):
        a_str = a if isinstance(a, str) else a['#text']
        return a_str.replace(" ", "_")

    def clean_author_field(self, a_field):
        if isinstance(a_field, list):
            return [self.author2str(a) for a in a_field]
        else:
            return [self.author2str(a_field)]

    @staticmethod
    def clean_title_field(t):
        return t if isinstance(t, str) else t['#text']

    def handle(self, path, item):
        if not all(key in item for key in ['year', 'author', 'title']):
            return True
        venue = item.get('journal', item.get('booktitle', None))
        if not venue:
            return True
        dic = {
            'type': path[1][0],
            'year': item['year'],
            'author': self.clean_author_field(item['author']),
            'title': self.clean_title_field(item['title']),
            'venue': venue
        }
        self.json_handler.write(zlib.compress(json.dumps(dic).encode('utf8')))
        self._index.append(self.json_handler.tell())
        return True

    def build(self, rebuild=False):
        """
        Main class method. Create the data and index files.

        Parameters
        ----------
        rebuild: str
                    Tell if files are to be rebuilt if they are already there.
        Returns
        -------

        """
        if self.dblp_xml.exists() and not rebuild:
            print(f"File {self.dblp_xml} already exists")
        else:
            print((f"Retrieve {self.dblp_xml} from the Internet."))
            self.download()
        if self.dblp_json.exists() and not rebuild:
            print(f"File {self.dblp_json} already exists.")
        else:
            print(f"Converting DBLP database from {self.dblp_xml} (may take a while).")
            self._index = [0]
            with gzip.open(self.dblp_xml) as self.xml_handler:
                with open(self.dblp_json, "wb") as self.json_handler:
                    xmltodict.parse(self.xml_handler, item_depth=2, item_callback=self.handle)
            print(f"Conversion done. Building Index.")
            with open(self.dblp_index, "wb") as f:
                pickle.dump(np.array(self._index), f)
            print(f"Conversion done.")
