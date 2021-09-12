import argparse
import os
from pyserini.search import SimpleSearcher
from tqdm import tqdm
from trectools import TrecRun, TrecQrel, TrecEval


def load_queries(f):
    for line in f:
        if len(line.strip()) == 0:
            continue
        qid, text = line.strip().split(",", 1)
        yield qid, text


def search(searcher, query, depth):
    hits = searcher.search(query, depth)
    rank_list = []
    for rank, hit in enumerate(hits):
        rank_list.append({"query": query, "doc": hit.docid, "rank": rank + 1, "score": hit.score})
    return rank_list


def write_collection(documents, collection_file, index_dir, append=False):
    searcher = SimpleSearcher(index_dir)
    with open(collection_file, 'a' if append else 'w') as out_file:
        for docid in tqdm(documents, desc="Writing collection"):
            text = searcher.doc(docid).contents().replace("\n", " ").replace("\t", " ")
            out_file.write(f"{docid}\t{text}\n")
    print(f"Collection written to {collection_file}")


def write_trec_results(results, run_file):
    with open(run_file, 'w') as out_file:
        for qid, docid, rank, score in results:
            out_file.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.4f}\t{run_file}\n")
    print(f"Seacrch results written to {run_file}")


def run_one_query(searcher: SimpleSearcher, qid, query_str, depth, query_result):
    hits = searcher.search(query_str, depth)
    for rank, hit in enumerate(hits):
        query_result.append((qid, hit.docid, rank + 1, hit.score))


def run_queries(index_dir, queries, depth=1000, b=0.4, k1=0.9, rm3=False):
    searcher = SimpleSearcher(index_dir)
    searcher.set_bm25(k1, b)
    if rm3:
        searcher.set_rm3()
    all_results = []
    for qid, query_str in tqdm(queries.items(), desc=f"Running {len(queries)} queries"):
        run_one_query(searcher, qid, query_str, depth, all_results)
    return all_results


def bm25_tune(index_dir, queries, depth, run_file, rm3):
    for k1 in tqdm([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]):
        for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
            results = run_queries(index_dir, queries, depth, b, k1, rm3)
            write_trec_results(results, f"{run_file}-b{b}-k1{k1}.res")


if __name__ == '__main__':

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=dir_path, required=True)
    parser.add_argument('--queries', type=argparse.FileType('r'), required=True)
    parser.add_argument('-d', '--depth', type=int, default=1000, help='Retrieve up to rank depth.')
    parser.add_argument('--collection', help="File to write collection")
    parser.add_argument('--k1', default=0.6, type=float, help='BM25 k1 parameter.')
    parser.add_argument('--b', default=0.7, type=float, help='BM25 b parameter.')
    parser.add_argument('--bm25_tune', action='store_true', help='Run a suit of BM25 params.')
    parser.add_argument('--rm3', action='store_true', help='Run RM3.')
    parser.add_argument('ranking', help="File to write ranking")

    args = parser.parse_args()

    qs = dict(load_queries(args.queries))

    print("Params:\n " + "\n ".join([f"{k} = {v}" for k, v in vars(args).items()]))

    if args.bm25_tune:
        bm25_tune(args.index_dir, qs, depth=args.depth, run_file=args.ranking, rm3=args.rm3)
    else:
        ret_results = run_queries(index_dir=args.index_dir, queries=qs, depth=args.depth, rm3=args.rm3)
        docs = [docid for qid, docid, rank, score in ret_results]
        write_trec_results(docs, run_file=args.ranking)
        if args.collection:
            write_collection(ret_results, args.collection, args.index_dir)
