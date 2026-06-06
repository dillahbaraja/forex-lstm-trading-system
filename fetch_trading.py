import urllib.request
import json

url = "https://api.openalex.org/works?search=deep+learning+stock+trading+LSTM&filter=publication_year:2021-2026,has_doi:true&sort=cited_by_count:desc&per-page=10"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    for w in data.get('results', []):
        title = w.get('title', '')
        doi = w.get('doi', '')
        year = w.get('publication_year', '')
        host_venue = w.get('primary_location', {}).get('source', {})
        venue = host_venue.get('display_name', 'Unknown Journal') if host_venue else 'Unknown Journal'
        
        authorships = w.get('authorships', [])
        author_names = [a.get('author', {}).get('display_name', '') for a in authorships if a.get('author', {}).get('display_name')]
        authors_str = ", ".join(author_names) if author_names else "Unknown Authors"
        print(f"{authors_str}, \"{title},\" *{venue}*, {year}. {doi}")
