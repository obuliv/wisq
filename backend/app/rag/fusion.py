RRF_K = 60  # standard RRF constant; also Qdrant's own documented default


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Combines multiple best-first id rankings (e.g. one per retrieval
    modality) into a single score per id via Reciprocal Rank Fusion. An id
    absent from a given ranking contributes 0 from that modality -- it is not
    penalized beyond simply not gaining a term. Higher score is better.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
