"""
Tests para reciprocal_rank_fusion() en hybrid_retriever.py.

Cobertura DTm-4:
  - Rankings vacios retorna lista vacia
  - Un solo ranking retorna mismo orden (con scores RRF)
  - Dos rankings, doc en ambos recibe score fusionado
  - Doc presente en solo un ranking
  - Pesos desiguales priorizan ranking con mayor peso
  - top_n menor que candidatos trunca resultado
  - Pesos incorrectos (len != rankings) fallback a iguales
  - Ranking con un solo doc
  - Multiples rankings (3+)
  - k parametro afecta scores RRF

Cobertura Fase 3 (cookbook formula):
  - Cookbook formula uses k=0 (rank-only)
  - Cookbook re-scores as 1/(new_rank+1) after sort
  - Cookbook preserves order with asymmetric weights
  - Cookbook empty rankings returns empty
  - Classic formula is default (backwards-compat)

Sin dependencias externas.
"""
from shared.retrieval.hybrid_retriever import reciprocal_rank_fusion


# =================================================================
# Tests
# =================================================================

def test_empty_rankings_returns_empty():
    """Sin rankings, retorna lista vacia."""
    result = reciprocal_rank_fusion([], k=60, top_n=10)
    assert result == [], f"Esperado [], obtenido {result}"


def test_single_ranking_preserves_order():
    """Un solo ranking mantiene orden (scores RRF decrecientes)."""
    ranking = [("d1", 10.0), ("d2", 5.0), ("d3", 1.0)]
    result = reciprocal_rank_fusion([ranking], k=60, top_n=10)

    ids = [doc_id for doc_id, _ in result]
    assert ids == ["d1", "d2", "d3"], f"Orden inesperado: {ids}"

    # Scores decrecientes
    scores = [s for _, s in result]
    assert scores[0] > scores[1] > scores[2], f"Scores no decrecientes: {scores}"


def test_two_rankings_doc_in_both_gets_higher_score():
    """Doc presente en ambos rankings recibe score mayor que doc en solo uno."""
    ranking_a = [("d1", 10.0), ("d2", 5.0)]
    ranking_b = [("d1", 8.0), ("d3", 6.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.5, 0.5], k=60, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    # d1 en ambos rankings (pos 1 en ambos) -> mayor score
    assert "d1" in score_map
    assert "d2" in score_map
    assert "d3" in score_map
    assert score_map["d1"] > score_map["d2"], (
        f"d1 ({score_map['d1']}) deberia tener mayor score que d2 ({score_map['d2']})"
    )
    assert score_map["d1"] > score_map["d3"], (
        f"d1 ({score_map['d1']}) deberia tener mayor score que d3 ({score_map['d3']})"
    )


def test_doc_in_only_one_ranking():
    """Doc en solo un ranking recibe score de esa unica contribucion."""
    ranking_a = [("d1", 10.0)]
    ranking_b = [("d2", 10.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.5, 0.5], k=60, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    # Ambos en posicion 1 de su ranking respectivo -> mismos scores
    assert abs(score_map["d1"] - score_map["d2"]) < 1e-10, (
        f"d1={score_map['d1']}, d2={score_map['d2']} deberian ser iguales"
    )


def test_unequal_weights_prioritize_heavier_ranking():
    """Pesos desiguales priorizan docs del ranking con mayor peso."""
    # d_a solo en ranking A (peso 0.9), d_b solo en ranking B (peso 0.1)
    ranking_a = [("d_a", 10.0)]
    ranking_b = [("d_b", 10.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.9, 0.1], k=60, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    assert score_map["d_a"] > score_map["d_b"], (
        f"d_a ({score_map['d_a']}) deberia superar a d_b ({score_map['d_b']}) "
        f"con pesos [0.9, 0.1]"
    )


def test_top_n_truncates_results():
    """top_n limita el numero de resultados."""
    ranking = [(f"d{i}", float(10 - i)) for i in range(10)]
    result = reciprocal_rank_fusion([ranking], k=60, top_n=3)

    assert len(result) == 3, f"Esperado 3 resultados, obtenido {len(result)}"
    ids = [doc_id for doc_id, _ in result]
    assert ids == ["d0", "d1", "d2"], f"Top 3 inesperado: {ids}"


def test_wrong_weights_length_fallback_to_equal():
    """Pesos con longitud incorrecta fallback a pesos iguales."""
    ranking_a = [("d1", 10.0)]
    ranking_b = [("d2", 10.0)]

    # 3 pesos para 2 rankings -> fallback a iguales
    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.3, 0.3, 0.4], k=60, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    # Con pesos iguales, ambos en pos 1 -> mismos scores
    assert abs(score_map["d1"] - score_map["d2"]) < 1e-10


def test_rrf_score_formula():
    """Verifica la formula RRF: weight / (k + rank)."""
    ranking = [("d1", 100.0), ("d2", 50.0)]
    k = 60
    result = reciprocal_rank_fusion(
        [ranking], weights=[1.0], k=k, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    # d1 en posicion 1: score = 1.0 / (60 + 1) = 1/61
    expected_d1 = 1.0 / (k + 1)
    assert abs(score_map["d1"] - expected_d1) < 1e-10, (
        f"d1 score {score_map['d1']} != esperado {expected_d1}"
    )

    # d2 en posicion 2: score = 1.0 / (60 + 2) = 1/62
    expected_d2 = 1.0 / (k + 2)
    assert abs(score_map["d2"] - expected_d2) < 1e-10, (
        f"d2 score {score_map['d2']} != esperado {expected_d2}"
    )


def test_three_rankings_accumulate():
    """Tres rankings: scores se acumulan correctamente."""
    r1 = [("d1", 10.0), ("d2", 5.0)]
    r2 = [("d2", 10.0), ("d3", 5.0)]
    r3 = [("d1", 10.0), ("d3", 5.0)]

    k = 60
    result = reciprocal_rank_fusion(
        [r1, r2, r3], k=k, top_n=10
    )
    score_map = {doc_id: s for doc_id, s in result}

    w = 1.0 / 3.0  # pesos iguales
    # d1: pos 1 en r1, no en r2, pos 1 en r3 -> 2 * w/(k+1)
    # d2: pos 2 en r1, pos 1 en r2 -> w/(k+2) + w/(k+1)
    # d3: pos 2 en r2, pos 2 en r3 -> 2 * w/(k+2)
    expected_d1 = 2 * w / (k + 1)
    expected_d2 = w / (k + 2) + w / (k + 1)
    expected_d3 = 2 * w / (k + 2)

    assert abs(score_map["d1"] - expected_d1) < 1e-10
    assert abs(score_map["d2"] - expected_d2) < 1e-10
    assert abs(score_map["d3"] - expected_d3) < 1e-10


def test_k_parameter_affects_score_magnitude():
    """Menor k produce scores mas altos (denominador mas pequeno)."""
    ranking = [("d1", 10.0)]

    result_k10 = reciprocal_rank_fusion([ranking], k=10, top_n=10)
    result_k100 = reciprocal_rank_fusion([ranking], k=100, top_n=10)

    score_k10 = result_k10[0][1]
    score_k100 = result_k100[0][1]

    assert score_k10 > score_k100, (
        f"k=10 ({score_k10}) deberia dar score mayor que k=100 ({score_k100})"
    )


def test_empty_ranking_in_list():
    """Un ranking vacio dentro de la lista no causa error."""
    ranking_a = [("d1", 10.0)]
    ranking_b = []  # vacio

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.5, 0.5], k=60, top_n=10
    )
    assert len(result) == 1
    assert result[0][0] == "d1"


# =================================================================
# Tests — Cookbook RRF formula (Fase 3)
# =================================================================

def test_cookbook_empty_rankings_returns_empty():
    """Cookbook formula: sin rankings, retorna lista vacia."""
    result = reciprocal_rank_fusion([], k=60, top_n=10, formula="cookbook")
    assert result == []


def test_cookbook_single_ranking_preserves_order():
    """Cookbook formula: un solo ranking mantiene orden."""
    ranking = [("d1", 10.0), ("d2", 5.0), ("d3", 1.0)]
    result = reciprocal_rank_fusion(
        [ranking], k=60, top_n=10, formula="cookbook"
    )

    ids = [doc_id for doc_id, _ in result]
    assert ids == ["d1", "d2", "d3"]

    # Cookbook re-scores: d1=1/1, d2=1/2, d3=1/3
    scores = [s for _, s in result]
    assert abs(scores[0] - 1.0) < 1e-10
    assert abs(scores[1] - 0.5) < 1e-10
    assert abs(scores[2] - 1.0 / 3) < 1e-10


def test_cookbook_rescoring_formula():
    """Cookbook formula: final scores are 1/(new_rank+1) after sort."""
    ranking_a = [("d1", 10.0), ("d2", 5.0)]
    ranking_b = [("d1", 8.0), ("d3", 6.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.5, 0.5],
        k=60, top_n=10, formula="cookbook",
    )

    # d1 is in both -> highest fusion score -> re-scored as 1/1
    # d2 and d3 have lower fusion scores -> re-scored as 1/2 and 1/3
    score_map = {doc_id: s for doc_id, s in result}
    assert abs(score_map["d1"] - 1.0) < 1e-10, (
        f"d1 should be re-scored as 1.0, got {score_map['d1']}"
    )

    # Second and third get 1/2 and 1/3
    non_d1_scores = sorted(
        [s for did, s in result if did != "d1"], reverse=True
    )
    assert abs(non_d1_scores[0] - 0.5) < 1e-10
    assert abs(non_d1_scores[1] - 1.0 / 3) < 1e-10


def test_cookbook_asymmetric_weights_affect_order():
    """Cookbook formula: asymmetric weights change ranking order."""
    # d_a only in ranking A (weight 0.8), d_b only in ranking B (weight 0.2)
    # Both at position 1
    ranking_a = [("d_a", 10.0)]
    ranking_b = [("d_b", 10.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.8, 0.2],
        k=60, top_n=10, formula="cookbook",
    )

    ids = [doc_id for doc_id, _ in result]
    # d_a has fusion score 0.8/1 = 0.8, d_b has 0.2/1 = 0.2
    # After sort: d_a first, d_b second
    assert ids[0] == "d_a", f"d_a should rank first, got {ids}"

    # Re-scored: d_a=1/1=1.0, d_b=1/2=0.5
    score_map = {doc_id: s for doc_id, s in result}
    assert abs(score_map["d_a"] - 1.0) < 1e-10
    assert abs(score_map["d_b"] - 0.5) < 1e-10


def test_cookbook_doc_in_both_ranks_higher_than_single():
    """Cookbook formula: doc in both rankings beats doc in only one."""
    ranking_a = [("d1", 10.0), ("d2", 5.0)]
    ranking_b = [("d1", 8.0), ("d3", 6.0)]

    result = reciprocal_rank_fusion(
        [ranking_a, ranking_b], weights=[0.5, 0.5],
        k=60, top_n=10, formula="cookbook",
    )

    ids = [doc_id for doc_id, _ in result]
    # d1 in both rankings at position 1 -> fusion = 0.5/1 + 0.5/1 = 1.0
    # d2 only in A at pos 2 -> fusion = 0.5/2 = 0.25
    # d3 only in B at pos 2 -> fusion = 0.5/2 = 0.25
    assert ids[0] == "d1", f"d1 should rank first, got {ids}"


def test_cookbook_k_parameter_ignored():
    """Cookbook formula ignores k parameter (always uses k=0)."""
    ranking = [("d1", 10.0)]

    result_k10 = reciprocal_rank_fusion(
        [ranking], k=10, top_n=10, formula="cookbook",
    )
    result_k100 = reciprocal_rank_fusion(
        [ranking], k=100, top_n=10, formula="cookbook",
    )

    # Both should produce identical re-scored results (1/1 = 1.0)
    assert abs(result_k10[0][1] - result_k100[0][1]) < 1e-10, (
        "Cookbook formula should not depend on k"
    )


def test_cookbook_top_n_truncates():
    """Cookbook formula: top_n limits output."""
    ranking = [(f"d{i}", float(10 - i)) for i in range(10)]
    result = reciprocal_rank_fusion(
        [ranking], k=60, top_n=3, formula="cookbook",
    )

    assert len(result) == 3
    ids = [doc_id for doc_id, _ in result]
    assert ids == ["d0", "d1", "d2"]

    # Re-scored after truncation: 1/1, 1/2, 1/3
    scores = [s for _, s in result]
    assert abs(scores[0] - 1.0) < 1e-10
    assert abs(scores[1] - 0.5) < 1e-10
    assert abs(scores[2] - 1.0 / 3) < 1e-10


def test_classic_is_default_formula():
    """Default formula is classic (backwards-compat)."""
    ranking = [("d1", 10.0)]
    k = 60

    # Without formula parameter (default)
    result_default = reciprocal_rank_fusion([ranking], k=k, top_n=10)
    # Explicit classic
    result_classic = reciprocal_rank_fusion(
        [ranking], k=k, top_n=10, formula="classic",
    )

    assert abs(result_default[0][1] - result_classic[0][1]) < 1e-10, (
        "Default formula should be classic"
    )

    # Classic score for d1 at rank 1: 1.0/num_rankings / (k+1)
    expected = 1.0 / (k + 1)
    assert abs(result_default[0][1] - expected) < 1e-10


def test_cookbook_three_rankings():
    """Cookbook formula: three rankings accumulate correctly."""
    r1 = [("d1", 10.0), ("d2", 5.0)]
    r2 = [("d2", 10.0), ("d3", 5.0)]
    r3 = [("d1", 10.0), ("d3", 5.0)]

    result = reciprocal_rank_fusion(
        [r1, r2, r3], k=60, top_n=10, formula="cookbook",
    )

    # Fusion scores (before re-scoring):
    # d1: w/1 + 0 + w/1 = 2w/1 (w=1/3)
    # d2: w/2 + w/1 + 0 = w/2 + w = 3w/2
    # d3: 0 + w/2 + w/2 = w
    # d1 = 2/3, d2 = 3/6 = 1/2, d3 = 1/3
    # Wait, w = 1/3
    # d1 = 2*(1/3)/1 = 2/3
    # d2 = (1/3)/2 + (1/3)/1 = 1/6 + 1/3 = 1/2
    # d3 = (1/3)/2 + (1/3)/2 = 1/3
    # Order: d1 > d2 > d3

    ids = [doc_id for doc_id, _ in result]
    assert ids == ["d1", "d2", "d3"], f"Order should be d1, d2, d3, got {ids}"

    # Re-scored: 1/1, 1/2, 1/3
    scores = [s for _, s in result]
    assert abs(scores[0] - 1.0) < 1e-10
    assert abs(scores[1] - 0.5) < 1e-10
    assert abs(scores[2] - 1.0 / 3) < 1e-10


if __name__ == "__main__":
    test_empty_rankings_returns_empty()
    test_single_ranking_preserves_order()
    test_two_rankings_doc_in_both_gets_higher_score()
    test_doc_in_only_one_ranking()
    test_unequal_weights_prioritize_heavier_ranking()
    test_top_n_truncates_results()
    test_wrong_weights_length_fallback_to_equal()
    test_rrf_score_formula()
    test_three_rankings_accumulate()
    test_k_parameter_affects_score_magnitude()
    test_empty_ranking_in_list()
    # Cookbook formula tests
    test_cookbook_empty_rankings_returns_empty()
    test_cookbook_single_ranking_preserves_order()
    test_cookbook_rescoring_formula()
    test_cookbook_asymmetric_weights_affect_order()
    test_cookbook_doc_in_both_ranks_higher_than_single()
    test_cookbook_k_parameter_ignored()
    test_cookbook_top_n_truncates()
    test_classic_is_default_formula()
    test_cookbook_three_rankings()
