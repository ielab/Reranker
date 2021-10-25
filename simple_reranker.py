from pyserini.search import SimpleSearcher
from reranker import RerankerForInference


index_dir = '/Users/koo01a/ir.collections/agask-sources/index/grdc_journal_document_passage_3_index_20210524'
searcher = SimpleSearcher(index_dir)
rk = RerankerForInference.from_pretrained("./examples/agask/models/agask_model_custom_params")  # load checkpoint


query = input("Enter your query:")
hits = searcher.search(query)

results = {}
for hit in hits:
    inputs = rk.tokenize(query, hit.contents, return_tensors='pt')
    logits = score = rk(inputs).logits
    score = logits[:, 1]
    results[hit.docid] = score

for r in sorted(results, key=lambda item: item[1]):
    print(f"{r} - {results[r]}")

