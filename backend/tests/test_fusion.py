from app.rag.fusion import reciprocal_rank_fusion


def test_known_rankings_produce_known_scores():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1)
    # a: rank1 in list1 (1/(1+1)=0.5) + rank2 in list2 (1/(1+2)=1/3)
    # b: rank2 in list1 (1/3) + rank1 in list2 (0.5)
    assert scores["a"] == 0.5 + 1 / 3
    assert scores["b"] == 1 / 3 + 0.5
    assert scores["a"] == scores["b"]


def test_deterministic_ordering_for_equal_scores():
    scores = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert scores["x"] > scores["y"]


def test_id_present_in_only_one_ranking_contributes_only_that_term():
    scores = reciprocal_rank_fusion([["a", "b"], ["a"]], k=60)
    assert scores["a"] == 1 / 61 + 1 / 61
    assert scores["b"] == 1 / 62
    assert "c" not in scores


def test_empty_rankings_produce_empty_scores():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}
