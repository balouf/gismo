import numpy as np
from numba import njit


def compress_csr(mat, ratio=0.8, min_degree=10, max_degree=None):
    """
    Inplace lossy compression of CSR matrix. Compression is performed row by row.

    Parameters
    ----------
    mat: :class:`~scipy.sparse.csr_matrix`
        Matrix to compress. It is assumed that the weights are non-negative and normalized (sum of a non-null row is 1).
    ratio: :class:`float`, default .8
        Target compression ratio (quantity of weights to preserve).
    min_degree: :class:`int`, default 10
        Don't compress rows with less than `mi_degree` entries.
    max_degree: class:`int`, optional
        If set, rows are allowed at most `max_degree` entries.

    Returns
    -------
    None

    Examples
    --------

    Start with a full matrix with a csr representation:

    >>> from scipy.sparse import csr_matrix
    >>> from gismo.embedding import l1_normalize
    >>> np.random.seed(42)
    >>> x = csr_matrix(np.random.rand(10, 10))
    >>> l1_normalize(x.indptr, x.data)
    >>> x  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 100 stored elements and shape (10, 10)>
    >>> x.toarray()
    array([[0.07200801, 0.18278161, 0.14073106, 0.11509637, 0.0299957 ,
            0.02999106, 0.01116699, 0.16652855, 0.11556865, 0.13613201],
           [0.00520773, 0.24538041, 0.21060217, 0.05372031, 0.04600045,
            0.04640006, 0.07697116, 0.13275971, 0.10927907, 0.07367894],
           [0.15281528, 0.03483974, 0.07296552, 0.09150188, 0.11390722,
            0.19610414, 0.04987017, 0.12843427, 0.1479604 , 0.01160137],
           [0.11929704, 0.03348399, 0.01277348, 0.18632243, 0.18961076,
            0.15873628, 0.05981372, 0.01917882, 0.13435547, 0.086428  ],
           [0.03016589, 0.12239978, 0.00850029, 0.22476941, 0.06396626,
            0.16376487, 0.07704997, 0.12855247, 0.135138  , 0.04569305],
           [0.16851056, 0.13471549, 0.16328177, 0.15551799, 0.10391301,
            0.16021865, 0.0153797 , 0.03406116, 0.00786035, 0.05654132],
           [0.08316254, 0.05805864, 0.17731912, 0.07633199, 0.06010957,
            0.11611685, 0.03015256, 0.17164043, 0.01595108, 0.21115723],
           [0.16981899, 0.04369819, 0.00121433, 0.17932246, 0.15544009,
            0.16031091, 0.16960471, 0.01628265, 0.07882771, 0.02547996],
           [0.16460084, 0.11886802, 0.06310494, 0.01212109, 0.05930686,
            0.0620151 , 0.13914183, 0.12158739, 0.16919868, 0.09005523],
           [0.02655733, 0.15838451, 0.16894139, 0.12463829, 0.17120245,
            0.1096532 , 0.11607906, 0.09494058, 0.00564462, 0.02395858]])

    A hard compression:

    >>> y = x.copy()
    >>> compress_csr(y, ratio=.5, min_degree=2)
    >>> y  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 35 stored elements and shape (10, 10)>

    Here we keep 50% of the information with 35% of the entries.

    >>> y.toarray()
    array([[0.        , 0.29190263, 0.22474781, 0.        , 0.        ,
            0.        , 0.        , 0.26594645, 0.        , 0.21740311],
           [0.        , 0.41678747, 0.35771537, 0.        , 0.        ,
            0.        , 0.        , 0.22549715, 0.        , 0.        ],
           [0.24438164, 0.        , 0.        , 0.        , 0.        ,
            0.31360902, 0.        , 0.20539161, 0.23661773, 0.        ],
           [0.        , 0.        , 0.        , 0.34848152, 0.35463173,
            0.29688675, 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.42921769, 0.        ,
            0.31272397, 0.        , 0.        , 0.25805835, 0.        ],
           [0.26023632, 0.        , 0.25216133, 0.24017148, 0.        ,
            0.24743086, 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.31657526, 0.        , 0.        ,
            0.        , 0.        , 0.30643686, 0.        , 0.37698787],
           [0.32736433, 0.        , 0.        , 0.34568442, 0.        ,
            0.        , 0.32695126, 0.        , 0.        , 0.        ],
           [0.27685935, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.23403718, 0.20451054, 0.28459294, 0.        ],
           [0.        , 0.25416076, 0.27110146, 0.20000797, 0.27472981,
            0.        , 0.        , 0.        , 0.        , 0.        ]])

    Harder (re-compression):

    >>> compress_csr(y, ratio=.5, min_degree=1)
    >>> y  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 20 stored elements and shape (10, 10)>

    >>> y.toarray()
    array([[0.        , 0.52326452, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.47673548, 0.        , 0.        ],
           [0.        , 0.5381355 , 0.4618645 , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.43796726, 0.        , 0.        , 0.        , 0.        ,
            0.56203274, 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.49562644, 0.50437356,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.57850599, 0.        ,
            0.42149401, 0.        , 0.        , 0.        , 0.        ],
           [0.50787961, 0.        , 0.49212039, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.45644765, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.54355235],
           [0.48639022, 0.        , 0.        , 0.51360978, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.49311287, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.50688713, 0.        ],
           [0.        , 0.        , 0.49667631, 0.        , 0.50332369,
            0.        , 0.        , 0.        , 0.        , 0.        ]])


    A soft compression:

    >>> y = x.copy()
    >>> compress_csr(y, ratio=.98, min_degree=7)
    >>> y  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 89 stored elements and shape (10, 10)>
    >>> y.toarray()
    array([[0.0728212 , 0.18484578, 0.14232035, 0.11639616, 0.03033444,
            0.03032975, 0.        , 0.16840917, 0.11687378, 0.13766936],
           [0.        , 0.24666498, 0.21170467, 0.05400154, 0.04624126,
            0.04664296, 0.0773741 , 0.1334547 , 0.10985114, 0.07406465],
           [0.15460896, 0.03524867, 0.07382196, 0.09257589, 0.11524421,
            0.19840592, 0.05045552, 0.12994178, 0.14969709, 0.        ],
           [0.12084059, 0.03391723, 0.        , 0.18873321, 0.19206409,
            0.16079013, 0.06058764, 0.01942697, 0.13609386, 0.08754628],
           [0.03042451, 0.12344914, 0.        , 0.22669639, 0.06451465,
            0.16516886, 0.07771054, 0.12965457, 0.13629656, 0.04608479],
           [0.1698456 , 0.13578279, 0.16457539, 0.1567501 , 0.10473628,
            0.161488  , 0.01550155, 0.03433102, 0.        , 0.05698928],
           [0.08451057, 0.05899975, 0.18019339, 0.07756931, 0.06108393,
            0.11799906, 0.03064132, 0.17442265, 0.        , 0.21458001],
           [0.17284322, 0.04447639, 0.        , 0.18251594, 0.15820826,
            0.16316582, 0.17262513, 0.        , 0.08023152, 0.02593372],
           [0.16662046, 0.12032651, 0.06387923, 0.        , 0.06003454,
            0.06277602, 0.14084908, 0.12307925, 0.17127472, 0.09116019],
           [0.02670808, 0.1592836 , 0.16990041, 0.12534582, 0.17217431,
            0.11027566, 0.116738  , 0.09547953, 0.        , 0.02409458]])
    """
    n, m = mat.shape
    if max_degree is None:
        max_degree = m
    pt, ind, dat = jit_compress(
        mat.indptr, mat.indices, mat.data, n, ratio, min_degree, max_degree
    )
    mat.indptr, mat.indices, mat.data = pt, ind, dat


@njit
def jit_compress(ptrs, indices, datas, n, ratio, min_degree, m):
    in_start = 0
    out_start = 0

    for i in range(n):
        in_end = ptrs[i + 1]
        if in_end > in_start:
            dat = datas[in_start:in_end]
            ind = indices[in_start:in_end]
            target = ratio * np.sum(dat)

            cov = 0.0
            order = np.argsort(-dat)
            for j, w in enumerate(order):
                cov += dat[w]
                if (j + 1 >= min_degree) and (cov >= target or j + 1 >= m):
                    break

            out_end = out_start + j + 1
            datas[out_start:out_end] = dat[order[: j + 1]] / cov
            indices[out_start:out_end] = ind[order[: j + 1]]

        ptrs[i + 1] = out_end
        in_start = in_end
        out_start = out_end
    return ptrs, indices[:out_start], datas[:out_start]
