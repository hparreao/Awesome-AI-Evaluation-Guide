"""Keep retrieval evidence and generation grounding as separate observations."""
from typing import Dict, Iterable, Sequence, Set


def evaluate(retrieved_ids: Sequence[str], relevant_ids: Set[str], cited_ids: Sequence[str], supported_claims: int, total_claims: int) -> Dict[str, object]:
    retrieved = set(retrieved_ids)
    precision = len(retrieved & relevant_ids) / len(retrieved) if retrieved else None
    recall = len(retrieved & relevant_ids) / len(relevant_ids) if relevant_ids else None
    citation_coverage = len(set(cited_ids) & retrieved) / len(set(cited_ids)) if cited_ids else None
    groundedness = supported_claims / total_claims if total_claims else None
    return {"retrieval": {"precision": precision, "recall": recall}, "generation": {"citation_coverage": citation_coverage, "groundedness": groundedness}}


if __name__ == "__main__":
    print(evaluate(["d1", "d2"], {"d1", "d3"}, ["d1"], 3, 4))
