import pkgutil
import gzip
import json


def get_acm_from_package():
    data = pkgutil.get_data(__package__, 'acm.json.gz')
    return json.loads(gzip.decompress(data))


def get_acm():
    """
    Returns
    -------
    acm: list of dicts
        Each dict is an ACM domain. It contains category name,
        query (concatenation of names from domain and subdomains),
        size (number of subdomains including itself), and children (list of domain dicts).

    Examples
    --------

    >>> acm = get_acm()
    >>> subdomain = acm[4]['children'][2]['children'][1]
    >>> subdomain['name']
    'Software development process management'
    >>> subdomain['size']
    10
    >>> subdomain['query']
    'Software development process management, Software development methods, Rapid application development, Agile software development, Capability Maturity Model, Waterfall model, Spiral model, V-model, Design patterns, Risk management'
    >>> len(acm)
    13
    """
    return get_acm_from_package()


def flatten_acm(acm, min_size=5, max_depth=100, exclude=None, depth=0):
    """
    Select subdomains of an acm tree and return them as a list.

    Parameters
    ----------
    acm: list of dicts
        acm tree from get_acm.
    min_size: int
        size threshold to select a domain (avoids small domains)
    max_depth: int
        depth threshold to select a domain (avoids deep domains)
    exclude: list
        list of domains to exclude from the results

    Returns
    -------
    list
        A flat list of domains described by name and query.

    Example
    -------
    >>> acm = flatten_acm(get_acm())
    >>> acm[111]['name']
    'Graph theory'
    """
    if exclude is None:
        exclude = set()
    result = [{'name': t['name'], 'query': t['query']} for t in acm if t['size'] > min_size]
    for t in acm:
        if len(t['children']) > 0 and depth < max_depth:
            result += flatten_acm(t['children'], min_size=min_size, max_depth=max_depth, depth=depth + 1)
    if depth == 0:
        result = [{'name': key, 'query': value}
                  for key, value in {t['name']: t['query'] for t in result}.items()
                  if key not in exclude]
    return result
