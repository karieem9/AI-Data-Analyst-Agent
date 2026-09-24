from unittest.mock import patch

import pandas as pd

import agent


def test_retry_succeeds_on_second_attempt():
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    calls = []

    def fake_generate_code(question, profile_text, previous_code=None, previous_error=None):
        calls.append((previous_code, previous_error))
        if previous_code is None:
            return "df.sort_values('revenue', inplace=True)\nresult = df"  # real validator rejects this
        return "result = df['revenue'].sum()"  # valid fix

    with patch.object(agent, "generate_code", side_effect=fake_generate_code):
        payload = agent.answer_question("total revenue?", df, "profile text")

    assert payload["attempts"] == 2, payload
    assert payload["error"] is None, payload
    assert payload["kind"] == "metric", payload
    assert payload["metric"] == "6", payload
    assert len(calls) == 2
    assert calls[1][0] is not None and calls[1][1] is not None, "retry call should carry previous code + error"
    print(f"PASS: retry succeeds on 2nd attempt -> metric={payload['metric']}, attempts={payload['attempts']}")


def test_retry_exhausted_shows_real_error():
    df = pd.DataFrame({"revenue": [1, 2, 3]})

    def always_unsafe(question, profile_text, previous_code=None, previous_error=None):
        return "result = open('x').read()"

    with patch.object(agent, "generate_code", side_effect=always_unsafe):
        payload = agent.answer_question("total revenue?", df, "profile text")

    assert payload["attempts"] == 2, payload
    assert payload["error"] is not None
    assert "open" in payload["error"], "should be the real validator message, not a generic one"
    assert "kind" not in payload
    print(f"PASS: retry exhausted -> real error surfaced: {payload['error']!r}")


def test_no_retry_needed_when_first_attempt_succeeds():
    df = pd.DataFrame({"revenue": [1, 2, 3]})
    calls = []

    def good_first_try(question, profile_text, previous_code=None, previous_error=None):
        calls.append(1)
        return "result = df['revenue'].sum()"

    with patch.object(agent, "generate_code", side_effect=good_first_try):
        payload = agent.answer_question("total revenue?", df, "profile text")

    assert payload["attempts"] == 1, payload
    assert payload["error"] is None
    assert len(calls) == 1
    print(f"PASS: no retry needed -> attempts={payload['attempts']}")


def test_live_smoke():
    """No mocking -- confirms the real wiring (llm -> sandbox -> agent) still works."""
    from profiling import profile_dataframe, profile_to_text

    df = pd.read_csv("sample_sales.csv")
    profile_text = profile_to_text(df, profile_dataframe(df))
    payload = agent.answer_question("What was total revenue?", df, profile_text)

    assert payload["error"] is None, payload
    assert payload["attempts"] == 1, "a normal question shouldn't need a retry"
    print(f"PASS: live smoke test -> metric={payload.get('metric')}, attempts={payload['attempts']}")


if __name__ == "__main__":
    test_retry_succeeds_on_second_attempt()
    test_retry_exhausted_shows_real_error()
    test_no_retry_needed_when_first_attempt_succeeds()
    test_live_smoke()
    print("\nAll Phase 7 tests passed.")
