"""Top-level package for GISMO."""
from importlib.metadata import metadata

infos = metadata(__name__)
__author__ = infos['Author']
__email__ = infos['Author-Email']
__version__ = infos['Version']


from gismo.corpus import Corpus
from gismo.embedding import Embedding
from gismo.diteration import DIteration
from gismo.gismo import Gismo, XGismo
