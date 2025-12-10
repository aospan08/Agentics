from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
import wrds
from pydantic import BaseModel, Field

from agentics import AG
from ddgs import DDGS

# below lines for testing that the LLM is accessable 
# import os, agentics.core.llm_connections as llms
# import agentics, agentics.core.llm_connections as llms
# st.write("GEMINI_API_KEY", bool(os.getenv("GEMINI_API_KEY")))
# st.write("available_llms", llms.available_llms)
# print(agentics.__file__)

BASE = Path(__file__).resolve().parent
# print(BASE)
DEFAULT_WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")
DEFAULT_FOCUS_TOP_K = int(os.getenv("FOCUS_TOP_K", "10"))
ENABLE_NEWS_ENV = os.getenv("ENABLE_NEWS", "true").lower() == "true"

sector_etfs = {
    "Communication Services": "VOX",
    "Consumer Discretionary": "VCR",
    "Consumer Staples": "VDC",
    "Energy": "VDE",
    "Financial": "VFH",
    "Healthcare": "VHT",
    "Industrials": "VIS",
    "Materials": "VAW",
    "Real Estate": "VNQ",
    "Information Technology": "VGT",
    "Utilities": "VPU",
}


class IndicatorRequest(BaseModel):
    raw_request: Optional[str] = None
    company: Optional[str] = Field(None, description="Company name")
    ticker: str = Field(..., description="Company Ticker")
    sector: Literal[
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financial",
        "Healthcare",
        "Industrials",
        "Materials",
        "Real Estate",
        "Information Technology",
        "Utilities",
    ] = Field(..., description="Sector")
    anomaly_types: List[str] = Field(default_factory=list)
    timeframe_days: Optional[int] = 180
    output_style: Optional[str] = "markdown"


class MarketIndicatorSnapshot(BaseModel):
    date: Optional[datetime] = None
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Company Ticker")
    sector: Literal[
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financial",
        "Healthcare",
        "Industrials",
        "Materials",
        "Real Estate",
        "Information Technology",
        "Utilities",
    ] = Field(..., description="Sector")
    close_price: Optional[float] = None
    stock_return: Optional[float] = Field(None, description="Stock return in 1+r format")
    sector_return: Optional[float] = Field(None, description="Sector return in 1+r format")
    excess_return_pct: Optional[float] = Field(None, description="Daily excess return versus sector in percentages")
    cumul_ex_ret_pct: Optional[float] = Field(None, description="Cumulative excess return in percentages")
    stock_volatility: Optional[float] = Field(None, description="Stock 3 month volatility")
    sector_volatility: Optional[float] = Field(None, description="Sector 3 month volatility")
    board_event: Optional[str] = None
    analyst_rating_change: Optional[str] = None
    news: Optional[str] = None
    excess_return: Optional[float] = None
    daily_performance_label: Optional[str] = None
    macro_context: Optional[str] = None
    anomaly_score: Optional[float] = Field(None, description="Composite anomaly score used for ranking")


class MacroSignal(BaseModel):
    indicator: Optional[str]
    date: Optional[datetime]
    value: Optional[float] = None
    change_bps: Optional[float] = Field(None, description="Consecutive change in bps (used for fed fund rates)")
    change_yoy_pct: Optional[float] = Field(None, description="Year-over-year change in percentages (used for CPI)")


class BoardEventSource(BaseModel):
    directorname: Optional[str] = None
    rolename: Optional[str] = None
    description: Optional[str] = None
    announcementdate: Optional[str] = None


class BoardEventSummary(BaseModel):
    board_event: Optional[str] = Field(None, description="Board event description")
    date: Optional[str] = Field(None, description="Announcement date")


class AnalystRatingChange(BaseModel):
    date: Optional[datetime] = None
    analyst_rating_change: Optional[str] = Field(None, description="Analyst Rating Change")


class OverallStockHealth(BaseModel):
    window_start_date: Optional[str] = Field(None, description="Analysis period start date")
    window_end_date: Optional[str] = Field(None, description="Analysis period end date")
    company: Optional[str] = Field(None, description="Company Name")
    ticker: Optional[str] = Field(None, description="Ticker")
    sector: Optional[str] = Field(None, description="Stock's sector")
    close_price: Optional[float] = Field(None, description="Stock's latest close price")
    average_daily_excess_return: Optional[float] = Field(None, description="Stock's average daily excess return")
    cumulative_absolute_return: Optional[float] = Field(
        None, description="Stock's cumulative absolute return during the period covered"
    )
    cumulative_excess_return: Optional[float] = Field(
        None, description="Stock's cumulative excess return during the period covered relative to its sector"
    )
    stock_volatility: Optional[float] = Field(None, description="Stock's 3-month volatility in percentages")
    sector_volatility: Optional[float] = Field(None, description="Sector 3-month volatility in percentages")


class ReportDraft(BaseModel):
    user_focus: Optional[str] = Field(None, description="Anomaly focus areas by the user")
    overall_stock: Optional[OverallStockHealth] = Field(
        None, description="Information about the stock such as ticker, returns, sector, etc"
    )
    anomaly_notes: Optional[List[str]] = Field(None, description="Anomalous Dates and some related notes")
    macro_notes: Optional[List[str]] = Field(None, description="Overall macro news during the period covered")
    analyst_rating_note: Optional[List[str]] = Field(None, description="Overall analyst upgrades and downgrades news")
    board_event_note: Optional[List[str]] = Field(None, description="Overall board event news during the period covered")
    raw_highlights: Optional[str] = None
    markdown_summary: Optional[str] = None
    risk_rating: Optional[str] = None
    recommended_actions: Optional[str] = None


def ensure_required_fields(spec: IndicatorRequest) -> None:
    missing = []
    if not spec.ticker:
        missing.append("ticker")
    if not spec.sector:
        missing.append("sector")
    if missing:
        raise ValueError(f"Missing required fields from request: {', '.join(missing)}")


async def parse_indicator_request(user_request: str) -> IndicatorRequest:
    
    llm = AG.get_llm_provider("gemini")
    parser = AG(
        atype=IndicatorRequest,
        instructions=(
            "Extract company name, ticker, sector, anomaly keywords, timeframe, and desired tone from the user instruction. "
            "Return a markdown-friendly paraphrase inside raw_highlights if the user requests a specific style."
        ),
        llm=llm
    )
    parsed = await (parser << [user_request])
    if not parsed:
        raise ValueError(
            "LLM parser returned no result. Provide ticker/sector overrides or verify LLM configuration and API keys."
        )
    return parsed[0]


def connect_wrds(username: str) -> wrds.Connection:
    if not username:
        raise ValueError("Please provide a WRDS username (set WRDS_USERNAME in your .env).")
    return wrds.Connection(wrds_username=username)


def fetch_wrds_window(ticker: str, sector: str, start_date: datetime, end_date: datetime, db: wrds.Connection):
    vol_window = 63
    fetch_start = start_date - timedelta(days=90)
    sql = """
        SELECT datadate as date, tic AS ticker, conm AS company, gvkey, prccd AS close_price
        FROM comp_na_daily_all.secd 
        WHERE tic = %(tickers)s AND datadate BETWEEN %(start)s AND %(end)s
    """
    params = {"tickers": ticker, "start": fetch_start, "end": end_date}
    stock_df = db.raw_sql(sql, params=params)
    if stock_df.empty:
        raise ValueError(f"No WRDS price data returned for ticker {ticker}.")

    stock_df["stock_return"] = stock_df["close_price"] / stock_df["close_price"].shift(1)
    stock_df["stock_volatility"] = stock_df["stock_return"].rolling(window=vol_window, min_periods=30).std() * (252 ** 0.5)

    sector_ticker = sector_etfs[sector]
    params = {"tickers": sector_ticker, "start": fetch_start, "end": end_date}
    sector_df = db.raw_sql(sql, params=params)
    sector_df["sector_return"] = sector_df["close_price"] / sector_df["close_price"].shift(1)
    sector_df["sector_volatility"] = sector_df["sector_return"].rolling(window=vol_window, min_periods=30).std() * (252 ** 0.5)

    wrds_df = stock_df.merge(sector_df[["date", "sector_return", "sector_volatility"]], on="date")
    wrds_df = wrds_df.sort_values("date")
    wrds_df = wrds_df[pd.to_datetime(wrds_df["date"]).dt.date >= start_date.date()].reset_index(drop=True)
    wrds_df = wrds_df.assign(sector=sector)
    wrds_df["cumul_ex_ret_pct"] = (wrds_df["stock_return"].cumprod() - wrds_df["sector_return"].cumprod()) * 100
    wrds_df["excess_return_pct"] = (wrds_df["stock_return"] - wrds_df["sector_return"]) * 100

    board_sql = """
        SELECT directorname, rolename, description, announcementdate, effectivedate
        FROM boardex.na_board_dir_announcements board
        JOIN wrdsapps_link_crsp_comp_bdx.bdxcrspcomplink crsp
        ON board.companyid = crsp.companyid
        WHERE crsp.gvkey = %(gvkey)s AND board.announcementdate BETWEEN %(start)s AND %(end)s
    """

    gvkey = wrds_df["gvkey"].iloc[0]
    params = {"gvkey": gvkey, "start": start_date, "end": end_date}
    board_df = db.raw_sql(board_sql, params=params)
    wrds_df = wrds_df.drop(columns=["gvkey"])

    ibes_sql = """
        SELECT ticker, cname as company, anndats, analyst, ireccd, itext
        FROM tr_ibes.recddet 
        WHERE ticker = %(tickers)s AND ANNDATS BETWEEN %(start)s AND %(end)s
    """
    ibes_start_date = end_date - timedelta(days=365)
    params = {"tickers": ticker, "start": ibes_start_date, "end": end_date}
    ibes_df = db.raw_sql(ibes_sql, params=params)

    return wrds_df, board_df, ibes_df


async def summarize_board_events(board_df: pd.DataFrame):
    if board_df.empty:
        empty_ag = AG.from_states([], atype=BoardEventSummary)
        return pd.DataFrame(columns=["board_event", "date"]), empty_ag

    board_source = AG.from_dataframe(
        board_df[["directorname", "rolename", "description", "announcementdate"]],
        atype=BoardEventSource,
    )

    summaries = await (
        AG(
            atype=BoardEventSummary,
            instructions="Summarize the board event using the rolename, description, directorname information",
        )
        << board_source
    )

    events_df = summaries.to_dataframe()
    events_df["date"] = pd.to_datetime(events_df["date"]).dt.date
    return events_df, summaries


def summarize_rating_changes(ibes_df: pd.DataFrame, window_start: datetime):
    if ibes_df.empty:
        empty_df = pd.DataFrame(columns=["date", "analyst_rating_change"])
        empty_ag = AG.from_states([], atype=AnalystRatingChange)
        return empty_df, empty_ag

    ibes_df = ibes_df.sort_values(["analyst", "anndats"])
    ibes_df["prev_rating"] = ibes_df.groupby("analyst")["ireccd"].shift(1)
    ibes_df["prev_rating_text"] = ibes_df.groupby("analyst")["itext"].shift(1)
    rating_changes = ibes_df.loc[ibes_df["ireccd"] != ibes_df["prev_rating"]].copy()
    rating_changes = rating_changes.dropna(subset=["prev_rating"])
    rating_changes.sort_values("anndats", inplace=True)

    rating_changes = rating_changes.assign(
        analyst_rating_change=lambda df: df.apply(
            lambda r: f"{r['analyst']} changed recommendation: "
            f"{(r['prev_rating_text'] or '?').upper()} -> {(r['itext'] or '?').upper()}",
            axis=1,
        ),
    )
    ibes_changes_by_date = (
        rating_changes.groupby("anndats")["analyst_rating_change"]
        .agg("; ".join)
        .reset_index()
    )

    ibes_changes_by_date.rename(columns={"anndats": "date"}, inplace=True)
    ibes_changes_by_date["date"] = pd.to_datetime(ibes_changes_by_date["date"]).dt.date
    ibes_changes_by_date = ibes_changes_by_date.loc[
        (ibes_changes_by_date["date"] >= window_start.date())].copy()
    analyst_rating_change = AG.from_dataframe(ibes_changes_by_date, atype=AnalystRatingChange)
    return ibes_changes_by_date, analyst_rating_change


def compute_overall_stock_health(merged_df: pd.DataFrame) -> OverallStockHealth:
    merged_df = merged_df.sort_values("date")
    window_start_date = merged_df["date"].iloc[0]
    window_end_date = merged_df["date"].iloc[-1]
    company = merged_df["company"].iloc[0]
    ticker = merged_df["ticker"].iloc[0]
    sector = merged_df["sector"].iloc[0]
    close_price = merged_df["close_price"].iloc[-1]
    average_daily_excess_return = merged_df["excess_return_pct"].mean()
    cumulative_absolute_return = (merged_df["stock_return"].cumprod().iloc[-1] - 1) * 100
    cumulative_excess_return = merged_df["cumul_ex_ret_pct"].iloc[-1]
    stock_volatility = merged_df["stock_volatility"].iloc[-1] * 100
    sector_volatility = merged_df["sector_volatility"].iloc[-1] * 100

    def _fmt_date(value):
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        return str(value)

    return OverallStockHealth(
        window_start_date=_fmt_date(window_start_date),
        window_end_date=_fmt_date(window_end_date),
        company=company,
        ticker=ticker,
        sector=sector,
        close_price=close_price,
        average_daily_excess_return=average_daily_excess_return,
        cumulative_absolute_return=cumulative_absolute_return,
        cumulative_excess_return=cumulative_excess_return,
        stock_volatility=stock_volatility,
        sector_volatility=sector_volatility,
    )


def fetch_macro_context(start_date: datetime, end_date: datetime):
    fedfunds_path = BASE / "data" / "FEDFUNDS.csv"
    cpi_path = BASE / "data" / "CPIAUCSL.csv"
    fedfunds_df = pd.read_csv(fedfunds_path, parse_dates=[0])
    fedfunds_df["observation_date"] = pd.to_datetime(fedfunds_df["observation_date"])
    fedfunds_df["fedfunds_change"] = fedfunds_df["FEDFUNDS"].diff()

    fedfunds_window = fedfunds_df.loc[
        (fedfunds_df["observation_date"] >= start_date) &
        (fedfunds_df["observation_date"] <= end_date)
    ].copy()

    changes = fedfunds_window.loc[
        (fedfunds_window["fedfunds_change"].ne(0)) &
        (fedfunds_window["fedfunds_change"].notna()),
        ["observation_date", "FEDFUNDS", "fedfunds_change"],
    ]
    changes["fedfunds_change_bps"] = changes["fedfunds_change"] * 100
    changes.reset_index(drop=True, inplace=True)

    fedfundrate_df = (
        changes.rename(columns={"observation_date": "date", "FEDFUNDS": "value", "fedfunds_change_bps": "change_bps"})
        .assign(indicator="Fed Funds Rate")[["indicator", "date", "value", "change_bps"]]
    )
    fedfundrate_df["date"] = pd.to_datetime(fedfundrate_df["date"]).dt.date

    cpi_df = pd.read_csv(cpi_path, parse_dates=[0])
    cpi_df = cpi_df.sort_values("observation_date")
    cpi_df["cpi_yoy_pct"] = cpi_df["CPIAUCSL"].pct_change(12) * 100

    cpi_df = cpi_df.loc[
        (cpi_df["observation_date"] >= start_date) &
        (cpi_df["observation_date"] <= end_date)
    ].copy()

    cpi_df = (
        cpi_df.rename(columns={"observation_date": "date", "CPIAUCSL": "value", "cpi_yoy_pct": "change_yoy_pct"})
        .assign(indicator="CPI")[["indicator", "date", "value", "change_yoy_pct"]]
    )
    cpi_df["date"] = pd.to_datetime(cpi_df["date"]).dt.date

    return fedfundrate_df, cpi_df


async def compute_relative_health(state: MarketIndicatorSnapshot) -> MarketIndicatorSnapshot:
    if state.stock_return is not None and state.sector_return is not None:
        state.excess_return = state.stock_return - state.sector_return
        state.excess_return_pct = state.excess_return * 100
        if state.excess_return <= -0.02:
            state.daily_performance_label = "underperformance"
        elif state.excess_return >= 0.02:
            state.daily_performance_label = "outperformance"
        else:
            state.daily_performance_label = "within peer band"
    return state


def make_attach_macro_context(fed_macro_signal, cpi_macro_signal):
    fed_macro_by_date = {m.date: m for m in fed_macro_signal.states}
    cpi_macro_by_date = {m.date: m for m in cpi_macro_signal.states}

    async def attach_macro_context(state: MarketIndicatorSnapshot) -> MarketIndicatorSnapshot:
        macro_state = fed_macro_by_date.get(state.date)
        if macro_state:
            state.macro_context = f"{macro_state.indicator} changed by {macro_state.change_bps:.2f} bps to {macro_state.value}"
        macro_state = cpi_macro_by_date.get(state.date)
        if macro_state:
            state.macro_context = (
                f"{macro_state.indicator} indicates inflation move of {macro_state.change_yoy_pct:.2f}% as of {macro_state.date}"
            )
        return state

    return attach_macro_context


def summarize_anomaly(state: MarketIndicatorSnapshot) -> str:
    parts = [
        f"{state.date}: {state.daily_performance_label or 'neutral'} (excess return is {(state.excess_return_pct or 0):+.2f}% )",
    ]
    if state.anomaly_score is not None:
        parts.append(f"score={state.anomaly_score:.2f}")
    if state.macro_context:
        parts.append(f"macro={state.macro_context}")
    if state.board_event:
        parts.append(f"board={state.board_event}")
    if state.analyst_rating_change:
        parts.append(f"analyst={state.analyst_rating_change}")
    if state.news:
        parts.append(f"news={state.news}")
    return " | ".join(parts)


def build_report_drafts(
    states: List[MarketIndicatorSnapshot],
    board_event_summaries,
    analyst_rating_change,
    overall_stock_health,
    indicator_spec,
    fed_macro_signal,
    cpi_macro_signal,
) -> List[ReportDraft]:
    if not states:
        return []

    anomalies = [summarize_anomaly(s) for s in states]

    board_event_note: List[str] = []
    if len(board_event_summaries) > 0:
        board_event_note = [f"{s.date}: {s.board_event}" for s in board_event_summaries.states]

    analyst_rating_note: List[str] = []
    if len(analyst_rating_change) > 0:
        analyst_rating_note = [f"{s.date}: {s.analyst_rating_change}" for s in analyst_rating_change.states]

    fed_macro_note: List[str] = []
    cpi_macro_note: List[str] = []
    if len(fed_macro_signal) > 0:
        fed_macro_note = [
            f"{s.date}: {s.indicator} | value={s.value:.2f} | change_bps={s.change_bps if s.change_bps not in [None, ''] else 'n/a'}"
            for s in fed_macro_signal.states
        ]
    if len(cpi_macro_signal) > 0:
        cpi_macro_note = [
            f"{s.date}: {s.indicator} | value={s.value:.2f} | change_yoy_pct={s.change_yoy_pct if s.change_yoy_pct not in [None, ''] else 'n/a'}"
            for s in cpi_macro_signal.states
        ]

    macro_notes = fed_macro_note + cpi_macro_note
    user_focus = (
        ", ".join(indicator_spec.anomaly_types)
        if indicator_spec.anomaly_types
        else indicator_spec.raw_request
        or "Anomaly-focused market health review"
    )

    draft = ReportDraft(
        user_focus=user_focus,
        overall_stock=overall_stock_health,
        anomaly_notes=anomalies,
        macro_notes=macro_notes,
        analyst_rating_note=analyst_rating_note,
        board_event_note=board_event_note,
        raw_highlights=indicator_spec.raw_request or user_focus,
    )
    return [draft]


def states_to_dataframe(states: List[BaseModel]) -> pd.DataFrame:
    if not states:
        return pd.DataFrame()
    records = []
    for s in states:
        record = s.model_dump()
        if isinstance(record.get("date"), (datetime, date)):
            record["date"] = record["date"].isoformat()
        records.append(record)
    return pd.DataFrame(records)


def prepare_focus_window(market_health, focus_top_k: int):
    def _score_state(state: MarketIndicatorSnapshot) -> float:
        base = abs(state.excess_return_pct or 0)
        if state.board_event:
            base = base * 2
        if state.analyst_rating_change:
            base = base * 2
        if state.macro_context:
            base = base * 2
        state.anomaly_score = base
        return base

    async def select_focus_window(states: List[MarketIndicatorSnapshot]) -> List[MarketIndicatorSnapshot]:
        if not states:
            return []
        for state in states:
            _score_state(state)
        top_k = focus_top_k if focus_top_k > 0 else 10
        ranked = sorted(states, key=lambda s: s.anomaly_score or 0, reverse=True)
        return ranked[:top_k]

    return select_focus_window


def get_news_fetcher(enable_news: bool):
    if not enable_news or DDGS is None:
        async def no_news(state: MarketIndicatorSnapshot) -> MarketIndicatorSnapshot:
            return state
        return no_news

    async def get_news(state: MarketIndicatorSnapshot) -> MarketIndicatorSnapshot:
        state.news = str(
            DDGS().text(
                f"What happended to the {state.sector} and {state.ticker} on {state.date}",
                max_results=3,
            )
        )
        return state

    return get_news

def render_stock_health(health):
    st.subheader("Overall Stock Health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", health.ticker or "—")
    c2.metric("Sector", health.sector or "—")
    c3.metric("Close Price", f"{health.close_price:.2f}" if health.close_price else "—")
    c4.metric("Window", f"{health.window_start_date} - {health.window_end_date}")

    rows = [
        ("Average Daily Excess Return (%)", health.average_daily_excess_return),
        ("Cumulative Absolute Return (%)", health.cumulative_absolute_return),
        ("Cumulative Excess Return (%)", health.cumulative_excess_return),
        ("Stock Volatility (%)", health.stock_volatility),
        ("Sector Volatility (%)", health.sector_volatility),
    ]
    st.table(pd.DataFrame(
        [{"Metric": name, "Value": "—" if val is None else f"{val:.2f}"} for name, val in rows]
    ))

async def run_pipeline(
    user_request: str,
    wrds_username: str,
    timeframe_days: int,
    focus_top_k: int,
    enable_news: bool,
    ticker_override: Optional[str] = None,
    sector_override: Optional[str] = None,
):
    try:
        indicator_spec = await parse_indicator_request(user_request)
    except ValueError:
        if not ticker_override or not sector_override:
            raise
        indicator_spec = IndicatorRequest(
            raw_request=user_request,
            ticker=ticker_override.strip().upper(),
            sector=sector_override,
            anomaly_types=[],
            timeframe_days=timeframe_days,
        )

    indicator_spec.raw_request = indicator_spec.raw_request or user_request
    if ticker_override:
        indicator_spec.ticker = ticker_override.strip().upper()
    if sector_override:
        indicator_spec.sector = sector_override
    if timeframe_days:
        indicator_spec.timeframe_days = timeframe_days
    ensure_required_fields(indicator_spec)

    window_end = datetime.today()
    window_start = window_end - timedelta(days=indicator_spec.timeframe_days or 180)

    db = connect_wrds(wrds_username)
    try:
        wrds_df, board_df, ibes_df = fetch_wrds_window(
            ticker=indicator_spec.ticker,
            sector=indicator_spec.sector,
            start_date=window_start,
            end_date=window_end,
            db=db,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    board_events_df, board_event_summaries = await summarize_board_events(board_df)
    ibes_changes_by_date, analyst_rating_change = summarize_rating_changes(ibes_df, window_start)

    merged_market_health_df = wrds_df.copy()
    merged_market_health_df["date"] = pd.to_datetime(merged_market_health_df["date"]).dt.date
    if not board_events_df.empty:
        merged_market_health_df = merged_market_health_df.merge(board_events_df, on="date", how="left")
        merged_market_health_df["board_event"] = merged_market_health_df["board_event"].where(
            merged_market_health_df["board_event"].notna(), None
        )
    else:
        merged_market_health_df["board_event"] = None
    if not ibes_changes_by_date.empty:
        merged_market_health_df = merged_market_health_df.merge(ibes_changes_by_date, on="date", how="left")
        merged_market_health_df["analyst_rating_change"] = merged_market_health_df["analyst_rating_change"].where(
            merged_market_health_df["analyst_rating_change"].notna(), None
        )
    else:
        merged_market_health_df["analyst_rating_change"] = None

    if merged_market_health_df.empty:
        raise ValueError(
            f"No WRDS rows returned for {indicator_spec.ticker} between {window_start.date()} and {window_end.date()}. "
            "Check the ticker/sector combination or widen the timeframe."
        )

    overall_stock_health = compute_overall_stock_health(merged_market_health_df)

    fedfundrate_df, cpi_df = fetch_macro_context(window_start, window_end)
    fed_macro_signal = (
        AG.from_dataframe(fedfundrate_df, atype=MacroSignal) if not fedfundrate_df.empty else AG.from_states([], atype=MacroSignal)
    )
    cpi_macro_signal = (
        AG.from_dataframe(cpi_df, atype=MacroSignal) if not cpi_df.empty else AG.from_states([], atype=MacroSignal)
    )

    market_health = AG.from_dataframe(merged_market_health_df, atype=MarketIndicatorSnapshot)
    attach_macro_context = make_attach_macro_context(fed_macro_signal, cpi_macro_signal)

    market_health = await market_health.amap(compute_relative_health)
    market_health = await market_health.amap(attach_macro_context)

    select_focus_window = prepare_focus_window(market_health, focus_top_k)
    focus_window = await market_health.areduce(select_focus_window)

    news_fetcher = get_news_fetcher(enable_news)
    focus_window = await focus_window.amap(news_fetcher)

    report_drafts_list = build_report_drafts(
        focus_window.states,
        board_event_summaries,
        analyst_rating_change,
        overall_stock_health,
        indicator_spec,
        fed_macro_signal,
        cpi_macro_signal,
    )
    report_drafts = AG.from_states(report_drafts_list, atype=ReportDraft)
    report_drafts.instructions = (
        "Produce a markdown report covering anomalies, macro context, and user focus areas. "
        "Highlight board changes and analyst upgrades/downgrades over the window, prioritize high anomaly_score days, "
        "compare to sector (cumulative excess return and 3-month volatility), and recommend concrete next steps. "
        "Cite news snippets when available."
    )
    report_drafts.llm = AG.get_llm_provider()

    final_report = await report_drafts.self_transduction(
        ["user_focus", "overall_stock", "macro_notes", "analyst_rating_note", "board_event_note", "anomaly_notes", "raw_highlights"],
        ["markdown_summary", "risk_rating", "recommended_actions"],
    )

    if not final_report:
        raise ValueError("LLM did not return a report.")

    report_state = final_report[0]
    focus_window_df = states_to_dataframe(focus_window.states)
    macro_df = (
        pd.concat([fedfundrate_df, cpi_df], ignore_index=True)
        if not (fedfundrate_df.empty and cpi_df.empty)
        else pd.DataFrame()
    )

    return {
        "indicator_spec": indicator_spec,
        "overall_stock_health": overall_stock_health,
        "report": report_state,
        "focus_window_df": focus_window_df,
        "wrds_df": merged_market_health_df,
        "board_events_df": board_events_df,
        "ibes_changes_df": ibes_changes_by_date,
        "macro_df": macro_df,
        "news_available": enable_news and DDGS is not None,
    }


def main():
    st.set_page_config(page_title="Stock Health Anomaly Report", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("Stock Health Anomaly Report")
    st.caption("Powered by Agentics and WRDS")

    default_prompt = (
        "Focus on MSFT (company is Microsoft). Highlight any board changes, analyst upgrades/downgrades, extreme volatility spikes, "
        "and situations where its performance lags or outperforms its sector over the last 180 days. Include macro shocks from the Fed."
    )

    with st.form("controls"):
        user_request = st.text_area("Natural-language request", value=default_prompt, height=140)
        col_ticker, col_sector = st.columns(2)
        ticker_override = col_ticker.text_input("Ticker override (optional)", value="")
        sector_options =["", *list(sector_etfs.keys())]
        sector_override = col_sector.selectbox(
            "Sector override (optional)",
            options=sector_options,
            index=0,
            format_func=lambda val: "Auto-detect from request" if val == "" else val,
        )
        col1, col2 = st.columns(2)
        wrds_username = col1.text_input("WRDS username", value=DEFAULT_WRDS_USERNAME)
        # timeframe_days = int(col2.number_input("Timeframe (days)", min_value=30, max_value=365, value=90, step=5))
        timeframe_input = col2.text_input(
            "Timeframe (days, optional)",
            value="",
            placeholder="Auto-detect from request (default 180 if missing)",
        )
        col3, col4 = st.columns(2)
        focus_top_k = int(col3.number_input("Top anomaly days to keep", min_value=1, max_value=30, value=DEFAULT_FOCUS_TOP_K))
        enable_news = col4.checkbox("Enrich anomalies with DuckDuckGo news", value=ENABLE_NEWS_ENV and DDGS is not None)
        show_tables = st.checkbox("Show intermediate tables", value=False)
        submitted = st.form_submit_button("Generate report")

    if not submitted:
        return

    if not user_request.strip():
        st.error("Please enter a request to run the pipeline.")
        return
    
    timeframe_days = None
    if timeframe_input.strip():
        try:
            tf_val = int(timeframe_input)
            if 30 <= tf_val <= 365:
                timeframe_days = tf_val
            else:
                st.warning("Timeframe must be between 30 and 365; using auto-detect.")
        except ValueError:
            st.warning("Timeframe must be an integer; using auto-detect.")
    news_note = None
    if enable_news and DDGS is None:
        news_note = "ddgs is not installed; skipping news enrichment."
        enable_news = False

    with st.spinner("Running Agentics pipeline..."):
        try:
            result = asyncio.run(
                run_pipeline(
                    user_request=user_request,
                    wrds_username=wrds_username,
                    timeframe_days=timeframe_days,
                    focus_top_k=focus_top_k,
                    enable_news=enable_news,
                    ticker_override=ticker_override or None,
                    sector_override=sector_override or None,
                )
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)
            return

    report_state = result["report"]
    overall_stock_health = result["overall_stock_health"]

    st.subheader("Report")
    st.markdown(report_state.markdown_summary or "_No markdown summary returned._")

    st.subheader("Risk Rating")
    st.markdown(report_state.risk_rating or "_No risk rating returned._")

    st.subheader("Recommended Actions")
    st.markdown(report_state.recommended_actions or "_No recommended actions returned._")

    if news_note:
        st.info(news_note)
    elif enable_news and not result.get("news_available"):
        st.info("News enrichment disabled; set ENABLE_NEWS=true and install ddgs to include snippets.")

    render_stock_health(overall_stock_health)
    # st.subheader("Parsed Request")
    # st.json(result["indicator_spec"].model_dump())
    # st.subheader("Overall Stock Health")
    # st.json(overall_stock_health.model_dump())
    

    if show_tables:
        st.subheader("Top anomaly dates")
        st.dataframe(result["focus_window_df"])

        st.subheader("WRDS window (last 30 rows)")
        st.dataframe(result["wrds_df"].tail(30))

        st.subheader("Board events")
        st.dataframe(result["board_events_df"])

        st.subheader("Analyst rating changes")
        st.dataframe(result["ibes_changes_df"])

        st.subheader("Macro signals in window")
        st.dataframe(result["macro_df"])


if __name__ == "__main__":
    main()
