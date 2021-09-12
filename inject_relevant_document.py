import argparse
import random
from collections import defaultdict
from pathlib import Path
from bm25_ranker import write_collection
from tqdm import tqdm

def read_qrel(qrel_file):
    relevant = defaultdict(list)
    with open(qrel_file, 'rt', encoding='utf8') as qfh:
        for line in qfh:
            topicid, _, docid, rel = line.split()
            if int(rel) > 0:
                relevant[topicid].append(docid)
    return relevant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('qrels')
    parser.add_argument('rank_file')
    parser.add_argument('-d', '--depth', type=int, default=500)
    parser.add_argument('--collection_file', help="Collection file.")
    parser.add_argument('--index_dir', help="Anserini index for creating collection file.")
    args = parser.parse_args()

    qrels = read_qrel(args.qrels)

    ranking = defaultdict(list)
    with open(args.rank_file) as fh:
        for line in fh:
            qid, _, doc, rank, score, _ = line.split()
            ranking[qid].append(doc)

    for qid, docs in ranking.items():
        if len([d for d in docs[:args.depth] if d in qrels[qid]]) == 0:
            ranking[qid][args.depth] = random.choice(qrels[qid])

    inject_file_name = args.rank_file.replace(Path(args.rank_file).stem, "injected-"+Path(args.rank_file).stem)
    with open(inject_file_name, 'w') as outfile:
        for qid, docs in tqdm(ranking.items(), desc="Writing new ranking"):
            for count, doc in enumerate(docs):
                outfile.write(f"{qid}\tQ0\t{doc}\t{count + 1}\t{-count}\tinjected-{Path(args.rank_file).stem}\n")
                if count == args.depth:
                    break
    print(f"New ranking written to {inject_file_name}.")

    if args.collection_file:
        docs = [doc for docs in ranking.values() for doc in docs]
        write_collection(docs, args.collection_file, args.index_dir, append=True)
