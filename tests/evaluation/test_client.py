from note_rag.evaluation.client import NoteRagHttpClient, RetrievalOptions
from note_rag.evaluation.models import BenchmarkCase


def test_retrieval_options_preserve_case_filters():
    case = BenchmarkCase(
        id="q1",
        question="No answer?",
        answerable=False,
        expected_behavior="refuse",
        filters={"filenames": ["notes.md"]},
    )

    payload = RetrievalOptions(candidate_k=12).as_payload(case)

    assert payload["candidate_k"] == 12
    assert payload["filters"] == {"filenames": ["notes.md"]}


def test_cost_estimate_requires_paired_rates():
    disabled = NoteRagHttpClient("http://example.test")
    enabled = NoteRagHttpClient(
        "http://example.test",
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    assert disabled._estimated_cost(100, 50) is None
    assert enabled._estimated_cost(100, 50) == 0.0002
