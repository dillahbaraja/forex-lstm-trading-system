import urllib.request
import urllib.parse
import json

def fetch_openalex(query):
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:2018-2026,has_doi:true&sort=relevance_score:desc&per-page=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"\n--- Results for: {query} ---")
            for w in data.get('results', []):
                title = w.get('title', '')
                doi = w.get('doi', '')
                year = w.get('publication_year', '')
                host_venue = w.get('primary_location', {}).get('source', {})
                venue = host_venue.get('display_name', 'Unknown Journal') if host_venue else 'Unknown Journal'
                authorships = w.get('authorships', [])
                author_names = [a.get('author', {}).get('display_name', '') for a in authorships if a.get('author', {}).get('display_name')]
                authors_str = ", ".join(author_names[:3]) + (" et al." if len(author_names) > 3 else "")
                print(f"[{authors_str}, \"{title},\" *{venue}*, {year}. {doi}]")
    except Exception as e:
        print("Error:", e)

queries = [
    "algorithmic trading real-time execution risk drawdown",
    "financial machine learning transaction costs offline metrics",
    "walk forward validation machine learning time series",
    "triple barrier labeling machine learning"
]

for q in queries:
    fetch_openalex(q)
