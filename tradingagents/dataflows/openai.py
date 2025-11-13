from typing import Dict

from openai import OpenAI

from .config import get_config


def _get_client_and_config() -> tuple[OpenAI, Dict]:
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])
    return client, config


def _supports_web_search(config: Dict) -> bool:
    provider = config.get("llm_provider", "openai")
    return provider.lower() != "openrouter"


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    try:
        return response.output[1].content[0].text
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("Unexpected response format from LLM provider") from exc


def get_stock_news_openai(query, start_date, end_date):
    client, config = _get_client_and_config()

    request_kwargs = dict(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    if _supports_web_search(config):
        request_kwargs["tools"] = [
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ]

    response = client.responses.create(**request_kwargs)

    return _extract_response_text(response)


def get_global_news_openai(curr_date, look_back_days=7, limit=5):
    client, config = _get_client_and_config()

    request_kwargs = dict(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    if _supports_web_search(config):
        request_kwargs["tools"] = [
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ]

    response = client.responses.create(**request_kwargs)

    return _extract_response_text(response)


def get_fundamentals_openai(ticker, curr_date):
    client, config = _get_client_and_config()

    request_kwargs = dict(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    if _supports_web_search(config):
        request_kwargs["tools"] = [
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ]

    response = client.responses.create(**request_kwargs)

    return _extract_response_text(response)