import tempfile
import io
from pathlib import Path
from pytest import fixture

from gismo.datasets.dblp import Dblp

URL = "https://github.com/balouf/gismo/raw/master/gismo/datasets/acm.json.gz"


def test_dblp_download_gz():
    """
    Download a gz file (not a dblp file for this test).
    """
    with tempfile.TemporaryDirectory() as tmp:
        dblp = Dblp(dblp_url=URL, path=Path(tmp))
        dblp.download()
        dest = Path(tmp) / Path("dblp.xml.gz")
        assert dest.exists() == True
