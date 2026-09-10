import pytest

from note_rag.evaluation import (
    citation_support,
    duplicate_question_rate,
    evaluate_learning_materials,
    multiple_choice_quality,
)


def test_citation_support_measures_answer_concept_overlap() -> None:
    assert (
        citation_support(
            "Mitochondria produce ATP", ["Mitochondria produce ATP in cells."]
        )
        == 1
    )
    assert citation_support("Unsupported claim", []) == 0


def test_duplicate_rate_detects_near_duplicate_prompts() -> None:
    rate = duplicate_question_rate(
        ["What produces ATP in cells?", "What produces ATP in cells", "Define osmosis"]
    )
    assert rate == pytest.approx(1 / 3)


def test_multiple_choice_quality_flags_ambiguity_and_bad_distractors() -> None:
    result = multiple_choice_quality(
        "ATP",
        [
            {"text": "ATP", "correct": True},
            {"text": "ATP", "correct": True},
            {"text": "Glucose", "correct": False},
            {"text": "Glucose", "correct": False},
        ],
    )
    assert result == {"answer_ambiguity": 1.0, "distractor_quality": 0.5}


def test_learning_material_report_covers_generation_quality_signals() -> None:
    report = evaluate_learning_materials(
        [
            {
                "objective_id": "energy",
                "item_type": "multiple_choice",
                "prompt": "What stores cell energy?",
                "answer": "ATP",
                "sources": ["ATP stores usable energy."],
                "options": [
                    {"text": "ATP", "correct": True},
                    {"text": "DNA", "correct": False},
                ],
                "difficulty": 2,
                "approval_status": "approved",
            },
            {
                "objective_id": "transport",
                "item_type": "short_answer",
                "prompt": "Define osmosis.",
                "answer": "movement of water",
                "sources": ["Osmosis is the movement of water."],
                "options": [],
                "difficulty": 3,
                "approval_status": "draft",
            },
        ]
    )
    assert report["citation_support"] == 1
    assert report["objective_coverage"] == 0.5
    assert report["answer_ambiguity"] == 0
    assert report["distractor_quality"] == 1
