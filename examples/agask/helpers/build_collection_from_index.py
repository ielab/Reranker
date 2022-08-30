import argparse
import json
from pyserini.search import SimpleSearcher
from tqdm import tqdm


def write_doc_to_collection(docid, searcher: SimpleSearcher, out_file):
    text = searcher.doc(docid).contents().replace("\n", " ").replace("\t", " ")
    out_file.write(f"{docid}\t{text}\n")


def write_doc_to_docs(docid, searcher: SimpleSearcher, out_file):
    doc_json = json.loads(searcher.doc(docid).raw())
    items = [doc_json[field].replace(',', '') for field in ['report_id', 'pdf_url', 'report_title', 'text']]
    out_file.write(f"{','.join(items)}\n")


def build_collection(index_dir, outfile, qrels=None, docs_format=True):
    searcher = SimpleSearcher(index_dir)

    with open(outfile, 'w') as out_file:
        if qrels:
            with open(qrels) as qfh:
                for line in tqdm(qfh, f"Writing docs from {qrels}"):
                    write_doc_to_collection(line.split()[2], searcher, out_file)
        else:
            for i in tqdm(range(searcher.num_docs), desc=f"Outputting {searcher.num_docs} docs"):
                if docs_format:
                    write_doc_to_docs(searcher.doc(i).docid(), searcher, out_file)
                else:
                    write_doc_to_collection(searcher.doc(i).docid(), searcher, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', required=True, help="Anserini index")
    parser.add_argument('-o', '--outfile', help="File to write output", required=True)
    parser.add_argument('--qrels', help="Use this qrel file for docs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collection', action='store_true', help='Write in collection format.')
    group.add_argument('--docs', action='store_true', help='Write in docs format')

    args = parser.parse_args()

    build_collection(index_dir=args.index_dir, outfile=args.outfile, qrels=args.qrels, docs_format=args.docs)
    print(f"Collection written to {args.collection}")
