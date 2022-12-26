import requests
import urllib
import pkgutil
import gzip
import json

from bs4 import BeautifulSoup as bs


def get_acm_from_package():
    data = pkgutil.get_data(__package__, 'acm.json.gz')
    return json.loads(gzip.decompress(data))


def rec_classification(domain):
    result = {'name': domain.find('a').text}
    ul = domain.find('ul')
    subdomains = ul.findChildren('li', recursive=False) if ul else []
    result['children'] = [rec_classification(sd) for sd in subdomains]
    children_query = ", ".join([r['query'] for r in result['children']])
    jonction = ", " if children_query else ""
    result['query'] = f"{result['name']}{jonction}{children_query}"
    result['size'] = 1 + sum([r['size'] for r in result['children']])
    return result


def get_acm_from_internet():
    s = requests.session()
    r = s.get('https://dl.acm.org/ccs')
    soup = bs(r.text, features="lxml")
    wid = soup.find('input', {'id': 'widgetID'})['value']
    pbc = soup.find('meta', {'name': 'pbContext'})['content']
    params = urllib.parse.urlencode({'widgetId': wid, 'pbContext': pbc}
                                    , quote_via=urllib.parse.quote)
    url = f"https://dl.acm.org/pb/widgets/acmCCS/fetchFlatView?{params}"
    r = s.get(url)
    soup = bs(r.text, features="lxml")
    domains = soup.find('div').find('div').findChildren('li', recursive=False)
    acm = [rec_classification(d) for d in domains]
    return acm


def get_acm(refresh=False):
    """

    Parameters
    ----------
    refresh: bool
        If ``True``, builds a new forest from the Internet, otherwise use a static version.

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

    >>> acm = get_acm(refresh=True)
    >>> len(acm)
    13
    """
    if refresh:
        return get_acm_from_internet()
    else:
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
