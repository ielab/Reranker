import argparse
from pyserini.search import SimpleSearcher
from tqdm import tqdm


def build_collection(index_dir, collection_outfile, qrels=None):
    searcher = SimpleSearcher(index_dir)

    docs = []
    if qrels:
        with open(qrels) as qfh:
            docs = [line.split()[2] for line in qfh]
    else:
        docs = [searcher.doc(i).docid() for i in range(searcher.num_docs)]

    with open(collection_outfile, 'w') as out_file:
        for docid in tqdm(docs, desc=f"Outputting {len(docs)} docs"):
            text = searcher.doc(docid).contents().replace("\n", " ").replace("\t", " ")
            out_file.write(f"{docid}\t{text}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', required=True, help="Anserini index")
    parser.add_argument('--collection', help="File to write collection", required=True)
    parser.add_argument('--qrels', help="Use this qrel file for docs")

    args = parser.parse_args()

    build_collection(index_dir=args.index_dir, collection_outfile=args.collection, qrels=args.qrels)
    print(f"Collection written to {args.collection}")
