import re
import time
import xml.etree.ElementTree as ET

import requests

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"  # namespace of the feed
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"  # namespace of totalResults
MIN_SECONDS_BETWEEN_REQUESTS = 3.0  # arXiv's API terms ask for <= 1 request / 3s
TIMEOUT_SECONDS = 15
HEADERS = {
    "User-Agent": "capstone-literature-scout/0.1",
    "Accept": "*/*",  # what curl sends; arXiv's front end has been returning 406 to bare urllib
}

_last_request_time = 0.0


class ArxivUnavailableError(Exception):
    """arXiv could not be reached or refused the request (network, timeout, HTTP error)."""


def _get(params):
    """GET the arXiv API with polite rate limiting. Returns the raw XML text.

    Raises ArxivUnavailableError on any network or HTTP failure, so callers
    don't need to know which HTTP library is used underneath.
    """
    global _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)

    try:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise ArxivUnavailableError(str(e)) from e
    finally:
        _last_request_time = time.monotonic()


def _text(element, tag):
    """Text of a child tag with whitespace collapsed ('' if missing)."""
    child = element.find(ATOM + tag)
    return " ".join(child.text.split()) if child is not None and child.text else ""


def parse_feed(xml_text):
    """Parse an arXiv Atom feed.

    Returns {"total": <total matches on arXiv>, "papers": [paper dicts]}.
    Raises ValueError if arXiv returned an error entry (malformed query or
    invalid ID), and xml.etree.ElementTree.ParseError on invalid XML.
    """
    root = ET.fromstring(xml_text)

    papers = []
    for entry in root.findall(ATOM + "entry"):
        entry_id = _text(entry, "id")
        # arXiv reports errors as a special entry rather than an HTTP error.
        if "/api/errors" in entry_id:
            raise ValueError(f"arXiv API error: {_text(entry, 'summary')}")
        papers.append({
            "arxiv_id": re.sub(r"v\d+$", "", entry_id.split("/abs/")[-1]),
            "title": _text(entry, "title"),
            "authors": [_text(author, "name") for author in entry.findall(ATOM + "author")],
            "published": _text(entry, "published")[:10],  # YYYY-MM-DD
            "abstract": _text(entry, "summary"),
            "url": entry_id,
        })

    total_element = root.find(OPENSEARCH + "totalResults")
    total = int(total_element.text) if total_element is not None else len(papers)
    return {"total": total, "papers": papers}


def search_raw(keywords, max_results, sort_by, category=None):
    """Title-keyword search, optionally restricted to one arXiv category.

    Every keyword must appear in the TITLE (ti:word AND ti:word ...), which your
    curl experiments showed gives far less noise than searching all fields.
    Note: arXiv stems words, so 'detection' also matches 'detecting'.

    sort_by: 'relevance' | 'submittedDate' | 'lastUpdatedDate'
    category: e.g. 'cs.CV', or None for all of arXiv.
    Returns {"total": int, "papers": [...]} with abstracts included.
    """
    clauses = [f"ti:{word}" for word in keywords.split()]
    if category:
        clauses.append(f"cat:{category}")
    xml_text = _get({
        "search_query": " AND ".join(clauses),
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    })
    return parse_feed(xml_text)


def fetch_by_ids_raw(arxiv_ids):
    """Fetch several papers in ONE request (id_list is comma-separated).

    Returns a list of papers; IDs that don't exist are simply missing from it.
    Raises ValueError if arXiv rejects an ID's format.
    """
    xml_text = _get({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})
    return parse_feed(xml_text)["papers"]
