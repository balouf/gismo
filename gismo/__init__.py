"""Top-level package for GISMO."""
from importlib.metadata import metadata
from gismo.corpus import Corpus
from gismo.embedding import Embedding, CountVectorizer
from gismo.clustering import cosine_similarity
from gismo.diteration import DIteration
from gismo.gismo import Gismo, XGismo
from gismo.common import MixInIO

infos = metadata(__name__)
__author__ = infos['Author']
__email__ = infos['Author-Email']
__version__ = infos['Version']

