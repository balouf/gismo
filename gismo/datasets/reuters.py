import requests
from zipfile import ZipFile
import io

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip"


def get_reuters_entry(name, z):
    """
    Read the Reuters news referenced by `name` in the zip archive `z` and returns it as a dict.

    Parameters
    ----------
    name: str
        Location of the file inside the Reuters archive
    z: ZipFile
        Zipfile descriptor of the Reuters archive

    Returns
    -------
    entry: dict
        dict with keys `set` (`C50test` or `c50train`), `author`, `id`, and `content`
    """
    with z.open(name) as f:
        description = name.split("/")
        return {'set': description[0],
                'author': description[1],
                'id': description[2][:-4],
                'content': f.read().decode()}


def get_reuters_news(url=URL):
    """
    Returns a list of news from the Reuters C50 news datasets

    Acknowledgments

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science.

    ZhiLiu, e-mail: liuzhi8673 '@' gmail.com,
    institution: National Engineering Research Center for E-Learning, Hubei Wuhan, China

    Parameters
    ----------
    url: str
        Location of the C50 dataset

    Returns
    -------
    list
        The C50 news as a list of dict

    Example
    ---------
    Cf :py:class:`~gismo.sentencizer.Sentencizer`
    """
    r=requests.get(url)
    with ZipFile(io.BytesIO(r.content)) as z:
        return [get_reuters_entry(name, z) for name in z.namelist() if name.endswith('.txt')]
