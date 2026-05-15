import json

import pytest

from agents.scoring.reader import InterviewContext, load_context
from agents.scoring.schema import TranscriptTurn


def test_load_context_returns_full_payload(recruit_conn):
    ctx = load_context(recruit_conn, "int-test")
    assert isinstance(ctx, InterviewContext)
    assert ctx.interview_id == "int-test"
    assert ctx.candidate_name == "Sarah Chen"
    assert ctx.job_title == "Senior Data Engineer"
    assert "Python, Spark" in ctx.cv_text
    assert "Build pipelines" in ctx.jd_text
    assert len(ctx.transcript) == 3
    assert ctx.transcript[0].q.startswith("Walk me")


def test_load_context_unknown_interview_raises(recruit_conn):
    with pytest.raises(LookupError) as ei:
        load_context(recruit_conn, "int-nope")
    assert "int-nope" in str(ei.value)


def test_load_context_empty_transcript(empty_interview_conn):
    ctx = load_context(empty_interview_conn, "int-empty")
    assert ctx.transcript == []


def test_normalise_question_answer_keys(recruit_conn):
    # Inject a transcript that uses the {question, answer} variant
    alt = json.dumps([{"question": "How are you?", "answer": "Great."}])
    recruit_conn.execute(
        "UPDATE interviews SET transcript_json = ? WHERE id = ?",
        (alt, "int-test"),
    )
    ctx = load_context(recruit_conn, "int-test")
    assert ctx.transcript == [TranscriptTurn(q="How are you?", a="Great.")]


def test_normalise_role_content_pairs(recruit_conn):
    # Chat-style: alternating recruiter/candidate messages
    chat = json.dumps([
        {"role": "recruiter", "content": "Tell me about Spark."},
        {"role": "candidate", "content": "I tuned shuffle partitions for a 5x speedup."},
        {"role": "recruiter", "content": "What broke?"},
        {"role": "candidate", "content": "Skewed keys; we salted them."},
    ])
    recruit_conn.execute(
        "UPDATE interviews SET transcript_json = ? WHERE id = ?",
        (chat, "int-test"),
    )
    ctx = load_context(recruit_conn, "int-test")
    assert len(ctx.transcript) == 2
    assert ctx.transcript[0].q == "Tell me about Spark."
    assert ctx.transcript[0].a.startswith("I tuned")


def test_load_context_uses_inline_override(recruit_conn):
    inline = [TranscriptTurn(q="override q", a="override a")]
    ctx = load_context(recruit_conn, "int-test", inline_override=inline)
    assert ctx.transcript == inline


def test_inline_override_used_when_db_empty(empty_interview_conn):
    inline = [TranscriptTurn(q="x", a="y")]
    ctx = load_context(empty_interview_conn, "int-empty", inline_override=inline)
    assert ctx.transcript == inline


def test_empty_list_inline_override_falls_through_to_db(recruit_conn):
    """An empty inline_override [] does NOT clobber a populated DB transcript."""
    ctx = load_context(recruit_conn, "int-test", inline_override=[])
    # Falls through to DB transcript (3 turns from the seeded fixture)
    assert len(ctx.transcript) == 3
