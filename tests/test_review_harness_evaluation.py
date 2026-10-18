"""Evaluation regressions: strict coverage, overlap accounting and source grounding."""
import json

from tools.benchmark_review_harness import aggregate, grounded, run_arm, score
from tools.review_harness_corpus import case, cases
from app.engine.rules import Match


def test_fictional_corpus_is_unique_and_contains_held_out_languages_and_controls():
    corpus = cases()
    assert len({item.identifier for item in corpus}) == len(corpus)
    holdout = [item for item in corpus if item.split == "holdout"]
    assert {item.language for item in holdout} == {"es", "en", "gl", "fr", "de", "it", "pt"}
    assert any(not item.truth for item in holdout)
    assert any(item.group == "injection" for item in holdout)
    for item in corpus:
        assert "[[" not in item.text
        for span in item.truth:
            assert item.text[span.start:span.end] == span.value


def test_overbroad_redaction_does_not_receive_perfect_character_precision():
    item = case("x", "en", "holdout", "direct", "Hello [[name|Mira Ellison]]. Useful public information.")
    findings = [Match(0, len(item.text), "other", item.text, 1)]
    result = score(item, findings)
    assert result["covered_entities"] == 1
    assert result["false_characters"] > result["true_characters"]


def test_partial_identifier_is_a_miss_and_duplicate_spans_do_not_inflate_scores():
    item = case("x", "en", "holdout", "direct", "Name [[name|Mira Ellison]].")
    match = Match(5, 9, "name", "Mira", 1)
    result = score(item, [match, match])
    assert result["covered_entities"] == 0
    assert result["predicted_characters"] == 4


def test_hallucinated_or_already_hidden_fragments_are_not_accepted():
    text = "Mira Ellison has asthma."
    known = [Match(0, 12, "name", "Mira Ellison", 1)]
    found, rejected = grounded(text, '{"datos":[{"texto":"Invented Name","tipo":"name"},'
                               '{"texto":"Mira Ellison","tipo":"name"},{"texto":"asthma","tipo":"health"}]}', known)
    assert rejected == 1
    assert [match.value for match in found] == ["asthma"]


def test_aggregate_uses_counts_rather_than_averaging_case_percentages():
    rows = [{"arm": "x", "score": {"entities": n, "covered_entities": found,
             "true_characters": found, "expected_characters": n, "false_characters": 0,
             "predicted_characters": found, "false_findings": 0, "negative_case": False,
             "negative_case_flagged": False}, "timing": {"seconds": 1, "calls": 1, "invalid_json": 0}}
            for n, found in [(1, 1), (9, 0)]]
    assert aggregate(rows)["x"]["entity_recall"] == 0.1


def test_invalid_critic_cannot_erase_base_or_grounded_candidates():
    item = case("x", "en", "holdout", "sensitive", "[[name|Mira Ellison]] has [[health|asthma]].")
    base = [Match(0, 12, "name", "Mira Ellison", 1)]

    class ScriptedServer:
        def __init__(self):
            self.calls = []
            self.answers = iter(['{"datos":[]}', '{"datos":[{"texto":"asthma","tipo":"health"}]}',
                                 '{"keep":[999]}'])

        def chat(self, messages, max_tokens=700, json_mode=True):
            self.calls.append({"valid_json": True})
            return next(self.answers)

    matches, timing = run_arm(ScriptedServer(), item, base, "specialists_critic")
    assert timing["critic_invalid"]
    assert {match.value for match in matches} == {"Mira Ellison", "asthma"}


def test_critic_can_only_select_proposals_and_base_is_protected():
    item = case("x", "en", "holdout", "sensitive", "[[name|Mira Ellison]] has [[health|asthma]].")
    base = [Match(0, 12, "name", "Mira Ellison", 1)]

    class RejectingServer:
        def __init__(self):
            self.calls = []

        def chat(self, messages, max_tokens=700, json_mode=True):
            self.calls.append({"valid_json": True})
            if len(self.calls) == 3:
                candidates = json.loads(messages[1]["content"])["candidates"]
                assert [candidate["texto"] for candidate in candidates] == ["asthma"]
                return '{"keep":[]}'
            return '{"datos":[{"texto":"asthma","tipo":"health"}]}'

    matches, timing = run_arm(RejectingServer(), item, base, "specialists_critic")
    assert not timing["critic_invalid"]
    assert [match.value for match in matches] == ["Mira Ellison"]
