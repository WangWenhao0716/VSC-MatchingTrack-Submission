#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module implements a baseline matching method based on SSCD features.

Usage:

First, adapt the SSCD "sscd_disc_mixup" torchscript model to remove L2
normalization. See `adapt_sscd_model.py`.

Second, run inference on both the queries and reference datasets, by
calling the inference script on each dataset with the adapted SSCD model.

Finally, run this script to perform retrieval and matching.
"""
import argparse
import logging
import os
from typing import List, Tuple
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from vsc.baseline.localization_ens import (
    VCSLLocalizationCandidateScore,
    VCSLLocalizationMaxSim,
)
from vsc.baseline.score_normalization import score_normalize, transform_features
from vsc.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.index import VideoFeature
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    Dataset,
    evaluate_matching_track,
    Match,
)
from vsc.storage import load_features, store_features


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sscd_baseline.py")
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--query_features",
    nargs='+',
    help="Path to query descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_features",
    nargs='+',
    help="Path to reference descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--score_norm_features",
    nargs='+',
    help="Path to score normalization descriptors",
    type=str,
)
parser.add_argument(
    "--output_path",
    help="The path to write match predictions.",
    type=str,
    required=True,
)
parser.add_argument(
    "--ground_truth",
    help="Path to the ground truth (labels) CSV file.",
    type=str,
)
parser.add_argument(
    "--overwrite",
    help="Overwrite prediction files, if found.",
    action="store_true",
)


def search(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    retrieve_per_query: float = 1200.0*2,
    candidates_per_query: float = 25.0,
) -> List[CandidatePair]:
    aggregation = MaxScoreAggregation()
    logger.info("Searching")
    cg = CandidateGeneration(refs, aggregation)
    num_to_retrieve = int(retrieve_per_query * len(queries))
    candidates = cg.query(queries, global_k=num_to_retrieve)
    num_candidates = int(candidates_per_query * len(queries))
    candidates = candidates[:num_candidates]
    logger.info("Got %d candidates", len(candidates))
    return candidates


def localize_and_verify(
    queriess: List[VideoFeature],
    refss: List[VideoFeature],
    candidates: List[CandidatePair],
    localize_per_query: float = 5.0,
    score_normalization: bool = False,
) -> List[Match]:
    num_to_localize = int(len(queriess[0]) * localize_per_query)
    candidates = candidates[:num_to_localize]# ???

    sb = 0.4
    print("similarity bias is: ", sb)
    if score_normalization:
        alignment = VCSLLocalizationMaxSim(
            queriess,
            refss,
            model_type="TN",
            tn_max_step=4,
            min_length=3,
            concurrency=16,
            tn_top_k=3,
            max_path=100,
            min_sim=0.2,
            max_iou=0.2,
            similarity_bias=sb, #0.5,
        )
    else:
        exit()
        alignment = VCSLLocalizationCandidateScore(
            transform_features(queries, normalize),
            transform_features(refs, normalize),
            model_type="TN",
            tn_max_step=5,
            min_length=4,
            concurrency=16,
        )

    matches = []
    logger.info("Aligning %s candidate pairs", len(candidates))
    BATCH_SIZE = 512
    i = 0
    while i < len(candidates):
        batch = candidates[i : i + BATCH_SIZE]
        matches.extend(alignment.localize_all(batch))
        i += len(batch)
        logger.info(
            "Aligned %d pairs of %d; %d predictions so far",
            i,
            len(candidates),
            len(matches),
        )

    return matches


def match(
    queriess: List[VideoFeature],
    refss: List[VideoFeature],
    output_path: str,
    score_normalization: bool = False,
) -> Tuple[str, str]:
    # Search
    assert len(queriess) == len(refss)
    length = len(queriess)
    candidates_list = []
    for i in range(length):
        candidates = search(queriess[i], refss[i])
        candidates_df = CandidatePair.to_dataframe(candidates)
        candidates_list.append(candidates_df)
    
    candidates_all = pd.concat(candidates_list)
    candidates_all = candidates_all.groupby(['query_id','ref_id']).max()
    candidates_all.to_csv('candidates_all.csv')
    candidates_all = pd.read_csv('candidates_all.csv')
    candidates_all = candidates_all.sort_values(by='score', ascending=False)
    
    os.makedirs(output_path, exist_ok=True)
    candidate_file = os.path.join(output_path, "candidates.csv")
    candidates_all.to_csv(candidate_file, index=False)
    
    candidates = CandidatePair.read_csv(candidate_file)

    # Localize and verify
    matches = localize_and_verify(
        queriess,
        refss,
        candidates,
        score_normalization=score_normalization,
    )
    matches_file = os.path.join(output_path, "matches.csv")
    Match.write_csv(matches, matches_file)
    return candidate_file, matches_file


def create_pr_plot(ap: AveragePrecision, filename: str):
    ap.pr_curve.plot(linewidth=1)
    plt.savefig(filename)
    plt.show()


def main(args):
    print(args)
    if os.path.exists(args.output_path) and not args.overwrite:
        raise Exception(
            f"Output path already exists: {args.output_path}. Do you want to --overwrite?"
        )
        
    queriess = []
    refss = []
    assert len(args.query_features) == len(args.ref_features)
    
    length = len(args.query_features)
    
    for i in range(length):
        queries = load_features(args.query_features[i], Dataset.QUERIES)
        refs = load_features(args.ref_features[i], Dataset.REFS)
        queriess.append(queries)
        refss.append(refs)
        
    score_normalization = False
    if args.score_norm_features:
        for i in range(length):
            queriess[i], refss[i] = score_normalize(
                queriess[i],
                refss[i],
                load_features(args.score_norm_features[i], Dataset.REFS),
                beta=1.2,
            )
        score_normalization = True
        
    candidate_file, match_file = match(
        queriess,
        refss,
        args.output_path,
        score_normalization=score_normalization,
    )

    if not args.ground_truth:
        return

    # Descriptor track uAP (approximate)
    gt_matches = Match.read_csv(args.ground_truth, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    candidate_pairs = CandidatePair.read_csv(candidate_file)
    candidate_uap = average_precision(gt_pairs, candidate_pairs)
    logger.info(f"Candidate uAP: {candidate_uap.ap:.4f}")
    candidate_pr_file = os.path.join(args.output_path, "candidate_precision_recall.pdf")
    create_pr_plot(candidate_uap, candidate_pr_file)

    # Matching track metric:
    match_metrics = evaluate_matching_track(args.ground_truth, match_file)
    logger.info(f"Matching track metric: {match_metrics.segment_ap.ap:.4f}")
    matching_pr_file = os.path.join(args.output_path, "precision_recall.pdf")
    create_pr_plot(match_metrics.segment_ap, matching_pr_file)
    logger.info(f"Candidates: {candidate_file}")
    logger.info(f"Matches: {match_file}")
    logger.info(f"Candidate PR plot: {candidate_pr_file}")
    logger.info(f"Match PR plot: {matching_pr_file}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
