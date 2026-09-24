"""Forecasting tests. No API key needed: the model is loaded from models/,
and the agent test patches out the LLM calls."""
from unittest.mock import patch

import numpy as np
import pandas as pd

import agent
from charts import auto_chart
from forecasting import ForecastError, forecast
from llm import SYSTEM_PROMPT
from sandbox import run_safely

daily = pd.read_csv("sample_daily_sales.csv")
daily_revenue = daily.groupby("date")["revenue"].sum()


def expect_forecast_error(series, fragment, label):
    try:
        forecast(series)
    except ForecastError as e:
        assert fragment in str(e), f"{label}: unexpected message: {e}"
        print(f"PASS: {label} -> {e}")
        return
    raise AssertionError(f"{label}: expected ForecastError")


def test_daily_forecast_shape_and_range():
    out = forecast(daily_revenue, 7)
    future = out.dropna(subset=["forecast"])
    assert list(out.columns) == ["actual", "forecast", "lower", "upper"]
    assert len(future) == 7
    assert (future["lower"] <= future["forecast"]).all() and (future["forecast"] <= future["upper"]).all()
    assert future.index[0] == pd.Timestamp("2026-09-01"), future.index[0]
    recent = daily_revenue.tail(28).mean()
    assert 0.7 * recent < future["forecast"].mean() < 1.3 * recent, (future["forecast"].mean(), recent)
    print(f"PASS: daily forecast -> 7 days, mean {future['forecast'].mean():.0f} vs recent {recent:.0f}")


def test_daily_forecast_keeps_weekly_pattern():
    # The sample has a strong weekly cycle (Saturdays highest). A forecast
    # that flattens it would mean the seasonal adjustment isn't applied.
    future = forecast(daily_revenue, 14).dropna(subset=["forecast"])["forecast"]
    by_day = future.groupby(future.index.dayofweek).mean()
    assert by_day.idxmax() == 5, by_day  # Saturday
    assert by_day.max() / by_day.min() > 1.2, by_day
    print(f"PASS: weekly pattern kept -> Saturday/lowest ratio {by_day.max() / by_day.min():.2f}")


def test_monthly_and_weekly():
    months = pd.date_range("2020-01-01", periods=48, freq="MS")
    t = np.arange(48)
    monthly = pd.Series(1000 + 10 * t + 200 * np.sin(2 * np.pi * t / 12), index=months)
    out = forecast(monthly, 6).dropna(subset=["forecast"])
    assert len(out) == 6 and out.index[0] == pd.Timestamp("2024-01-01"), out.index
    weeks = pd.date_range("2024-01-07", periods=60, freq="W-SUN")
    weekly = pd.Series(np.linspace(500, 800, 60), index=weeks)
    out = forecast(weekly, 4).dropna(subset=["forecast"])
    assert out.index[0] == weeks[-1] + pd.Timedelta(days=7), out.index  # stays on Sundays
    print("PASS: monthly and weekly forecasts land on the right dates")


def test_day_first_dates():
    # Walmart-style weekly dates written day-first ("05-02-2010"). Read
    # month-first they look scattered and used to be rejected as irregular.
    weeks = pd.date_range("2010-02-05", periods=60, freq="7D")
    s = pd.Series(np.linspace(1e6, 1.5e6, 60), index=weeks.strftime("%d-%m-%Y"))
    out = forecast(s, 4).dropna(subset=["forecast"])
    assert out.index[0] == weeks[-1] + pd.Timedelta(days=7), out.index
    print(f"PASS: day-first dates -> weekly forecast from {out.index[0].date()}")


def test_unsuitable_data():
    titanic_like = pd.Series(np.random.default_rng(0).uniform(5, 80, 100), name="Fare")
    expect_forecast_error(titanic_like, "no date", "no date column")
    expect_forecast_error(daily_revenue.head(10), "at least 28 days", "too little history")
    yearly = pd.Series([100.0, 120, 130, 150], index=[2020, 2021, 2022, 2023])
    expect_forecast_error(yearly, "yearly", "yearly data")
    rng = np.random.default_rng(1)
    random_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.sort(rng.choice(2000, 40, replace=False)) * 3, "D")
    expect_forecast_error(pd.Series(rng.uniform(1, 9, 40), index=random_dates), "evenly spaced", "irregular dates")
    names = pd.Series(["a", "b", "c"], index=pd.date_range("2024-01-01", periods=3), name="product")
    expect_forecast_error(names, "numeric", "non-numeric column")
    gappy = daily_revenue.iloc[::2].iloc[:40]  # every other day -> 2-day spacing
    expect_forecast_error(gappy, "evenly spaced", "every-other-day data")


def test_long_horizon_is_capped():
    out = forecast(daily_revenue, 90)
    assert out["forecast"].notna().sum() == 14
    assert "limited to 14 days" in out.attrs["forecast"]["notes"][0]
    print("PASS: 90-day request capped to 14 with a note")


def test_duplicate_dates_ask_the_llm_to_aggregate():
    raw = daily.set_index("date")["revenue"]  # two regions per date
    try:
        forecast(raw)
    except ForecastError:
        raise AssertionError("duplicates are a code mistake -> ValueError, so the agent retries")
    except ValueError as e:
        assert "Aggregate first" in str(e)
        print("PASS: duplicate dates -> ValueError asking to aggregate (retryable)")
        return
    raise AssertionError("expected ValueError")


def test_sandbox_and_chart():
    result = run_safely("result = forecast(df.groupby('date')['revenue'].sum(), 7)", daily)
    kind, fig = auto_chart(result)
    assert kind == "figure" and [t.name for t in fig.data][-1] == "forecast"
    print("PASS: forecast() runs in the sandbox and renders as a forecast chart")


def test_agent_shows_forecast_error_without_retry():
    df = pd.read_csv("sample_sales.csv")  # only 10 days
    calls = []

    def fake_generate_code(question, profile_text, previous_code=None, previous_error=None):
        calls.append(question)
        return "result = forecast(df.groupby('date')['revenue'].sum(), 7)"

    with patch.object(agent, "generate_code", side_effect=fake_generate_code), \
            patch.object(agent, "explain_result", return_value="explained"):
        payload = agent.answer_question("Forecast revenue for next week", df, "profile")

    assert payload["error"] is None and payload["attempts"] == 1 and len(calls) == 1, payload
    assert payload["kind"] == "metric" and "at least 28 days" in payload["metric"], payload
    print(f"PASS: agent shows the reason, no retry -> {payload['metric']}")


def test_code_patterns_from_real_model_runs():
    """Regressions from testing with the real model: the code gpt-4o-mini
    writes once the prompt covers these cases, run through the sandbox.
    Before the fix it answered "per year" as missing data, passed 3 for
    "next 3 months" on daily data, and made up a 'date' column."""
    for rule in ('"next 3 months" is\n  90', "Never answer it as missing data",
                 "(the data has no date column)"):
        assert rule in SYSTEM_PROMPT, f"prompt lost the rule: {rule}"

    months = pd.date_range("1949-01-01", periods=144, freq="MS")
    flights = pd.DataFrame({"year": months.year, "month": months.strftime("%B"),
                            "passengers": np.arange(144) + 100})
    no_dates = pd.DataFrame({"price": np.linspace(5, 50, 100)})
    cases = [
        (flights, "result = forecast(df.groupby('year')['passengers'].sum())", "Can't forecast yearly"),
        (no_dates, "result = forecast(df['price'], 30)", "needs a date column"),
    ]
    for data, code, fragment in cases:
        try:
            run_safely(code, data)
        except ForecastError as e:
            assert fragment in str(e), e
            print(f"PASS: {code} -> {e}")
            continue
        raise AssertionError(f"{code}: expected ForecastError")

    out = run_safely("result = forecast(df.groupby('date')['revenue'].sum(), 90)", daily)
    assert out["forecast"].notna().sum() == 14 and out.attrs["forecast"]["notes"]
    print("PASS: 'next 3 months' on daily data -> 90 periods, capped to 14 with a note")


if __name__ == "__main__":
    test_daily_forecast_shape_and_range()
    test_daily_forecast_keeps_weekly_pattern()
    test_monthly_and_weekly()
    test_day_first_dates()
    test_unsuitable_data()
    test_long_horizon_is_capped()
    test_duplicate_dates_ask_the_llm_to_aggregate()
    test_sandbox_and_chart()
    test_agent_shows_forecast_error_without_retry()
    test_code_patterns_from_real_model_runs()
    print("\nAll forecasting tests passed.")
