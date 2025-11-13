import json
import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Union

import typer

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.main import MessageBuffer, assemble_markdown_summary


app = typer.Typer(help="Batch runner for TradingAgents that reads a JSON configuration file.")


TickerConfig = Union[str, Dict[str, Any]]


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_tickers(ticker_entries: List[TickerConfig]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for entry in ticker_entries:
        if isinstance(entry, str):
            normalized.append({"ticker": entry})
        elif isinstance(entry, dict) and "ticker" in entry:
            normalized.append(entry)
        else:
            raise ValueError("Each ticker entry must be a string or an object containing a 'ticker' field.")
    return normalized


def _prepare_config(base_config: Dict[str, Any], per_target: Dict[str, Any]) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config.update(base_config)

    for key in ("llm_provider", "quick_think_llm", "deep_think_llm", "backend_url", "results_dir"):
        if key in per_target:
            config[key] = per_target[key]

    if "max_debate_rounds" not in per_target and "research_depth" in base_config:
        config["max_debate_rounds"] = base_config["research_depth"]
    if "max_risk_discuss_rounds" not in per_target and "research_depth" in base_config:
        config["max_risk_discuss_rounds"] = base_config["research_depth"]

    config.setdefault("results_dir", DEFAULT_CONFIG.get("results_dir", "./results"))
    return config


def _write_summary(
    selections: Dict[str, Any],
    config: Dict[str, Any],
    decision: str,
    final_state: Dict[str, Any],
) -> Path:
    message_buffer = MessageBuffer()
    for section in message_buffer.report_sections.keys():
        if section in final_state and final_state[section]:
            message_buffer.report_sections[section] = final_state[section]

    summary_content = assemble_markdown_summary(
        selections,
        config,
        decision,
        message_buffer,
        final_state,
    )

    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"] / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{selections['ticker']}_{timestamp}_summary.md"
    summary_path = results_dir / summary_filename
    summary_path.write_text(summary_content, encoding="utf-8")
    return summary_path


def _write_additional_outputs(
    final_state: Dict[str, Any],
    results_dir: Path,
    decision: str,
    ticker: str,
) -> None:
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    section_files = {
        "market_report": "market_report.md",
        "sentiment_report": "sentiment_report.md",
        "news_report": "news_report.md",
        "fundamentals_report": "fundamentals_report.md",
        "trader_investment_plan": "trader_investment_plan.md",
        "investment_plan": "investment_plan.md",
        "final_trade_decision": "final_trade_decision.md",
    }

    for section, filename in section_files.items():
        content = final_state.get(section)
        if content:
            (report_dir / filename).write_text(str(content), encoding="utf-8")

    message_log = results_dir / "message_tool.log"
    if not message_log.exists():
        message_log.touch()

    with message_log.open("a", encoding="utf-8") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(
            f"{timestamp} [Info] Batch run for {ticker} completed. Detailed step-by-step "
            "messages are unavailable in batch mode.\n"
        )
        if decision:
            log_file.write(f"{timestamp} [Decision] {decision}\n")


def _execute_single_target(task: Dict[str, Any]) -> None:
    ticker = task["ticker"]
    analysis_date = task["analysis_date"]
    run_config = task["run_config"]
    analysts = task["analysts"]
    research_depth = task["research_depth"]
    debug = task["debug"]
    task_index = task.get("task_index")
    task_total = task.get("task_total")

    console = typer.echo
    timestamp = datetime.datetime.now()
    banner = "=" * 80
    header_parts = [
        f"[{timestamp:%Y-%m-%d %H:%M:%S}]",
        f"Task {task_index}/{task_total}" if task_index and task_total else None,
        f"Analyzing {ticker}" ,
        f"Date {analysis_date}",
    ]
    header = " | ".join(part for part in header_parts if part)

    console("")
    console(typer.style(banner, fg=typer.colors.BRIGHT_BLUE))
    console(typer.style(header, fg=typer.colors.BRIGHT_CYAN, bold=True))
    console(typer.style(banner, fg=typer.colors.BRIGHT_BLUE))

    step_prefix = typer.style("[STEP]", fg=typer.colors.MAGENTA, bold=True)
    console(f"{step_prefix} Configuring graph with analysts: {', '.join(analysts)}")
    start_time = datetime.datetime.now()

    graph = TradingAgentsGraph(selected_analysts=analysts, config=run_config, debug=debug)
    final_state, decision = graph.propagate(ticker, analysis_date)

    duration = datetime.datetime.now() - start_time
    if decision:
        decision_text = typer.style(decision, fg=typer.colors.BRIGHT_GREEN, bold=True)
        console(f"{step_prefix} Final decision: {decision_text}")
    else:
        console(f"{step_prefix} Analysis completed without explicit decision.")

    selections = {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "analysts": analysts,
        "research_depth": research_depth,
        "llm_provider": run_config.get("llm_provider", "unknown"),
        "backend_url": run_config.get("backend_url", ""),
        "shallow_thinker": run_config.get("quick_think_llm"),
        "deep_thinker": run_config.get("deep_think_llm"),
    }

    summary_path = _write_summary(selections, run_config, decision, final_state)
    results_dir = Path(run_config["results_dir"]) / ticker / analysis_date
    _write_additional_outputs(final_state, results_dir, decision, ticker)
    console(
        f"{step_prefix} Summary saved: "
        + typer.style(str(summary_path), fg=typer.colors.YELLOW)
    )

    console(
        typer.style(
            f"[DONE] {ticker} completed in {duration.total_seconds():.2f}s",
            fg=typer.colors.BRIGHT_GREEN,
            bold=True,
        )
    )
    console(typer.style(banner, fg=typer.colors.BRIGHT_BLUE))


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to the batch configuration JSON."),
    debug: bool = typer.Option(False, help="Enable debug mode when constructing TradingAgentsGraph."),
) -> None:
    """Run TradingAgents analyses for multiple tickers defined in a JSON config."""

    config_data = _load_config(config_path)

    if "tickers" not in config_data:
        raise typer.BadParameter("Configuration file must include a 'tickers' field.")

    tickers = _normalize_tickers(config_data["tickers"])
    base_analysts = config_data.get("analysts", ["market", "social", "news", "fundamentals"])
    base_research_depth = config_data.get("research_depth", 1)

    base_config: Dict[str, Any] = {
        key: value
        for key, value in config_data.items()
        if key
        in {
            "llm_provider",
            "quick_think_llm",
            "deep_think_llm",
            "backend_url",
            "results_dir",
        }
    }
    base_config["research_depth"] = base_research_depth

    default_analysis_date = config_data.get("analysis_date")

    console = typer.echo
    console(f"Loaded configuration for {len(tickers)} ticker(s).")

    ctx = mp.get_context("spawn")

    for entry in tickers:
        ticker = entry["ticker"].upper()
        analysis_date = entry.get("analysis_date", default_analysis_date)
        if not analysis_date:
            raise typer.BadParameter(
                f"Analysis date missing for ticker {ticker}. Provide a global 'analysis_date' or per-ticker override."
            )

        analysts = entry.get("analysts", base_analysts)
        research_depth = entry.get("research_depth", base_research_depth)

        config_override = base_config.copy()
        config_override["research_depth"] = research_depth
        config_override.update({k: v for k, v in entry.items() if k in config_override})

        run_config = _prepare_config(base_config=config_override, per_target=entry)
        run_config["max_debate_rounds"] = research_depth
        run_config["max_risk_discuss_rounds"] = research_depth

        task = {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "analysts": analysts,
            "research_depth": research_depth,
            "run_config": run_config,
            "debug": debug,
        }

        process = ctx.Process(target=_execute_single_target, args=(task,))
        process.start()
        process.join()

        if process.exitcode != 0:
            raise typer.Exit(code=process.exitcode)

    console("\nAll analyses completed.")


if __name__ == "__main__":
    app()
