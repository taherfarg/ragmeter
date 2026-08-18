from ragmeter.judge.prompts import (
    PROMPT_VERSION,
    answer_relevance_prompt,
    chunk_relevance_prompt,
    faithfulness_prompt,
)

CHUNKS = [{"chunk_id": "c1", "text": "Returns accepted within 30 days."},
          {"chunk_id": "c2", "text": "Items must be unused."}]


def test_faithfulness_includes_question_chunks_and_answer():
    p = faithfulness_prompt("What is the return policy?", CHUNKS, "30 days.")
    assert "What is the return policy?" in p
    assert "Returns accepted within 30 days." in p
    assert "[c1]" in p and "[c2]" in p
    assert "30 days." in p
    assert '"claims"' in p


def test_faithfulness_labels_chunks_by_id_not_position():
    # The model must cite the real chunk id, otherwise the stored audit trail
    # cannot be traced back to a source.
    p = faithfulness_prompt("q", [{"chunk_id": "xyz-9", "text": "t"}], "a")
    assert "[xyz-9]" in p


def test_faithfulness_handles_chunks_without_text():
    p = faithfulness_prompt("q", [{"chunk_id": "c1"}], "a")
    assert "[c1]" in p


def test_answer_relevance_excludes_correctness():
    p = answer_relevance_prompt("Why?", "Because.")
    assert "Why?" in p
    assert "Because." in p
    # Relevance and faithfulness must measure different things, or running both
    # tells you the same thing twice.
    assert "correctness" in p.lower()
    assert '"score"' in p


def test_chunk_relevance_lists_every_chunk():
    p = chunk_relevance_prompt("q", CHUNKS)
    assert "[c1]" in p and "[c2]" in p
    assert '"judgments"' in p


def test_prompt_version_is_a_nonempty_string():
    # The cache key includes this. Editing a prompt without bumping it serves
    # stale scores from before the edit.
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_faithfulness_prompt_tells_the_judge_how_to_treat_a_refusal():
    # "I don't know" asserts nothing about the world, so faithfulness is not
    # defined over it. Without this instruction the judge invents a claim about
    # the speaker and scores the refusal 0.0 -- the same score a confident
    # fabrication gets.
    p = faithfulness_prompt("q", [{"chunk_id": "c1", "text": "t"}], "I don't know.")
    lowered = p.lower()
    assert "refus" in lowered or "does not answer" in lowered
    assert "empty" in lowered or "[]" in p


def test_prompt_version_bumped_past_one():
    # The cache key embeds this. Editing the wording without bumping it would
    # keep serving verdicts produced by the previous prompt.
    assert PROMPT_VERSION != "1"
