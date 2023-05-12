# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List

import numpy as np

from vsc.index import VideoFeature
from vsc.metrics import CandidatePair, Match


class Localization(abc.ABC):
    @abc.abstractmethod
    def localize(self, candidate: CandidatePair) -> List[Match]:
        pass

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        matches = []
        for candidate in candidates:
            matches.extend(self.localize(candidate))
        return matches


class LocalizationWithMetadata(Localization):
    def __init__(self, queriess: List[VideoFeature], refss: List[VideoFeature]):
        
        assert len(queriess) == len(refss)
        self.length = len(queriess)
        
        self.queries_list = []
        self.refs_list = []
        for i in range(self.length):
            self.queries_list.append({m.video_id: m for m in queriess[i]})
            self.refs_list.append({m.video_id: m for m in refss[i]})

    def similarity(self, candidate: CandidatePair):
        ans = []
        for i in range(self.length):
            a = self.queries_list[i][candidate.query_id].feature
            b = self.refs_list[i][candidate.ref_id].feature
            ans.append(np.matmul(a, b.T))
        return np.max(ans, axis=0)


class VCSLLocalization(LocalizationWithMetadata):
    def __init__(self, queriess, refss, model_type, similarity_bias=0.0, **kwargs):
        super().__init__(queriess, refss)

        # Late import: allow OSS use without VCSL installed
        from vcsl.vta import build_vta_model  # @manual

        self.model = build_vta_model(model_type, **kwargs)
        self.similarity_bias = similarity_bias

    def similarity(self, candidate: CandidatePair):
        """Add an optional similarity bias.

        Some localization methods do not tolerate negative values well.
        """
        return super().similarity(candidate) + self.similarity_bias

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        sims = [(f"{c.query_id}-{c.ref_id}", self.similarity(c)) for c in candidates]
        results = self.model.forward_sim(sims)
        assert len(results) == len(candidates)
        matches = []
        for (candidate, (key, sim), result) in zip(candidates, sims, results):
            query: VideoFeature = self.queries_list[0][candidate.query_id]
            ref: VideoFeature = self.refs_list[0][candidate.ref_id]
            assert key == result[0]
            for box in result[1]:
                (x1, y1, x2, y2) = box
                match = Match(
                    query_id=candidate.query_id,
                    ref_id=candidate.ref_id,
                    query_start=query.get_timestamps(x1)[0],
                    query_end=query.get_timestamps(x2)[1],
                    ref_start=ref.get_timestamps(y1)[0],
                    ref_end=ref.get_timestamps(y2)[1],
                    score=0.0,
                )
                score = self.score(candidate, match, box, sim)
                match = match._replace(score=score)
                matches.append(match)
        return matches

    def localize(self, candidate: CandidatePair) -> List[Match]:
        return self.localize_all([candidate])

    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return 1.0


class VCSLLocalizationMaxSim(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        x1, y1, x2, y2 = box
        return similarity[x1:x2, y1:y2].max() - self.similarity_bias


class VCSLLocalizationCandidateScore(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return candidate.score
