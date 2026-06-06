import urllib.request
import json
import urllib.parse
from datetime import datetime

def search_openalex(query, max_results=30, year_from=2020):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&filter=publication_year:>{year_from},is_paratext:false,type:article|proceedings-article|book-chapter&per-page={max_results}&sort=cited_by_count:desc"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:research@example.com"})
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                results = []
                for work in data.get('results', []):
                    authors = ", ".join([a['author']['display_name'] for a in work.get('authorships', []) if a.get('author')])
                    venue = work.get('primary_location', {}).get('source', {})
                    venue_name = venue.get('display_name', 'Unknown') if venue else 'Unknown'
                    results.append({
                        'title': work.get('title'),
                        'authors': authors,
                        'year': work.get('publication_year'),
                        'venue': venue_name,
                        'doi': work.get('doi'),
                        'citations': work.get('cited_by_count')
                    })
                return results
    except Exception as e:
        print(f"Error for query '{query}': {e}")
    return []

queries = [
    "LSTM forex trading", 
    "deep learning foreign exchange prediction", 
    "triple barrier method trading", 
    "walk-forward validation trading", 
    "financial machine learning execution"
]
all_results = []
for q in queries:
    res = search_openalex(q, max_results=15, year_from=2020)
    all_results.extend(res)

seen_titles = set()
unique_results = []
for r in all_results:
    if r['title'] and r['title'].lower() not in seen_titles:
        seen_titles.add(r['title'].lower())
        unique_results.append(r)

with open('C:/Users/dilla/OneDrive/Documents/Obsidian Vault/LSTM Neural Network/refs.json', 'w', encoding='utf-8') as f:
    json.dump(unique_results[:35], f, indent=2, ensure_ascii=False)
print(f"Saved {len(unique_results[:35])} references.")
