import logging
logging.basicConfig()
log = logging.getLogger("Gismo")


ALPHA = .5
"""
Default value for damping factor.
Controls the trade-off between closeness and centrality.
"""

N_ITER = 4
"""
Default value for the number of round-trip diffusions to perform.
Higher value means better precision but longer execution time.
"""

OFFSET = 1.0
"""
Default offset value.
Controls how much of the initial fluid should be deduced form the relevance.
"""

MEMORY = 0.0
"""
Default memory value.
Controls how much of previous computation is kept
when performing a new diffusion.
"""

STRETCH = 2.0
"""
Default stretch value.
When performing covering, defines the ratio between considered pages and
selected covering pages.
"""

RESOLUTION = 0.7
"""
Default resolution value.
Defines how strict the merging of cluster is during recursive clustering.
"""

MAX_K = 100
"""
Default top population size for estimating k.
"""

TARGET_K = 1.0
"""
Default threshold for estimating k.
"""

WIDE = True
"""
Default BFS behavior for covering.
True for wide variant, false for core variant.
"""

POST = True
"""
Default post policy.
If True, post function is applied on items and clusters.
"""

DISTORTION = 1.0
"""
Default distortion.
Controls how much of diteration relevance is mixed into the embedding
for similirity computation.
"""

DEFAULT_PARAMETERS = {'alpha': ALPHA,
                      'n_iter': N_ITER,
                      'offset': OFFSET,
                      'memory': MEMORY,
                      'stretch': STRETCH,
                      'resolution': RESOLUTION,
                      'max_k': MAX_K,
                      'target_k': TARGET_K,
                      'wide': WIDE,
                      'post': POST,
                      'distortion': DISTORTION
}
"""
Dictionary of default `runtime` Gismo parameters.
"""

class Parameters:
    """
    Manages Gismo runtime parameters. When called, an instance will yield
    a dictionary of parameters.

    Parameters
    ----------
    kwargs: dict
        Parameters that need to be distinct from default values.

    Examples
    --------

    Use default parameters.

    >>> p = Parameters()
    >>> p() # doctest: +NORMALIZE_WHITESPACE
    {'alpha': 0.5, 'n_iter': 4, 'offset': 1.0, 'memory': 0.0,
    'stretch': 2.0, 'resolution': 0.7, 'max_k': 100, 'target_k': 1.0,
    'wide': True, 'post': True, 'distortion': 1.0}

    Use default parameters with changed `stretch`.

    >>> p = Parameters(stretch=1.7)
    >>> p()['stretch']
    1.7

    Note that parameters that do not exist will be ignored and (a warning will
    be issued)

    >>> p = Parameters(strech=1.7)
    >>> p() # doctest: +NORMALIZE_WHITESPACE
    {'alpha': 0.5, 'n_iter': 4, 'offset': 1.0, 'memory': 0.0,
    'stretch': 2.0, 'resolution': 0.7, 'max_k': 100, 'target_k': 1.0,
    'wide': True, 'post': True, 'distortion': 1.0}

    You can change the value of an attribute to alter the returned parameter.

    >>> p.alpha = 0.85
    >>> p()['alpha']
    0.85

    You can also apply on-the-fly parameters by passing them when
    calling the instance.

    >>> p(resolution=0.9)['resolution']
    0.9

    Like for construction, parameters tha do not exist are ignored and a warning
    is issued.

    >>> p(resolutio = .9) # doctest: +NORMALIZE_WHITESPACE
    {'alpha': 0.85, 'n_iter': 4, 'offset': 1.0, 'memory': 0.0,
    'stretch': 2.0, 'resolution': 0.7, 'max_k': 100, 'target_k': 1.0,
    'wide': True, 'post': True, 'distortion': 1.0}
    """

    def __init__(self, **kwargs):
        for key, value in DEFAULT_PARAMETERS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
            else:
                log.warning(f"No parameter named {key}!")
    def __call__(self, **kwargs):
        d = {key: value for key, value in self.__dict__.items()}
        for key, value in kwargs.items():
            if key in d:
                if value is not None:
                    d[key] = value
            else:
                log.warning(f"No parameter named {key}!")
        return d
