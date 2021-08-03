# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--score_file', required=True)
    parser.add_argument('--run_id', default='marco')
    args = parser.parse_args()

    all_scores = defaultdict(dict)

    with open(args.score_file) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            qid, did, score = line.strip().split()
            score = float(score.replace('[', '').replace(']',''))
            all_scores[qid][did] = score

    qq = list(all_scores.keys())

    with open(args.score_file + '.marco', 'w') as f:
        for qid in qq:
            score_list = sorted(list(all_scores[qid].items()), key=lambda x: x[1], reverse=True)
            for rank, (did, score) in enumerate(score_list):
                f.write(f'{qid}\t{did}\t{rank+1}\n')

