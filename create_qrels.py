import json
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher("indices/recipes_lucene")
searcher.set_bm25(k1=0.9, b=0.4)

queries = open("data/recipes/recipes-queries.txt").read().strip().split('\n')

with open("data/recipes/recipes-qrels.txt", 'w') as f:
    for i, query in enumerate(queries, 1):
        hits = searcher.search(query, k=10)
        for rank, hit in enumerate(hits[:10]):
            doc = json.loads(searcher.doc(hit.docid).raw())
            title = doc.get('title', '').lower()
            query_lower = query.lower()
            
            # Simple relevance if query words appear in title
            query_words = set(query_lower.split())
            title_words = set(title.split())
            overlap = len(query_words & title_words)
            
            # 2 = 2+ words match, 1 = 1 word matches, 0 = no match
            if overlap >= 2:
                rel = 2
            elif overlap >= 1:
                rel = 1
            else:
                rel = 0
            
            f.write(f"{i} {hit.docid} {rel}\n")
