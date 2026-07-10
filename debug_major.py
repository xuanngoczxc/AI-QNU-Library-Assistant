import sys; sys.path.insert(0, '.')
from lib.search_engine import BM25SearchEngine, HybridSearchEngine, DocumentIndexer
from lib.catalog import search_topic_cache, build_topic_cache
documents = DocumentIndexer.load_from_data('Data')
bm25 = BM25SearchEngine(documents)
hybrid = HybridSearchEngine(bm25)
build_topic_cache(documents)

nganh = 'Tài chính - Ngân hàng'
print('Test search voi nganh:', nganh)
print()

# 1. Topic cache
cache_hits = search_topic_cache(nganh, documents)
print('Topic cache hits:', len(cache_hits) if cache_hits else 0)
if cache_hits:
    for h in cache_hits[:5]:
        d = h.get('doc', h)
        print('  -', d.get('title','?'), '| major:', d.get('major','?'))

print()

# 2. BM25 search
results = hybrid.search_by_query(nganh, top_k=50)
print('BM25 results:', len(results))
for r in results[:10]:
    d = r['doc']
    print('  score=%.1f | %s | major: %s' % (r['score'], d.get('title','?'), d.get('major','?')))
