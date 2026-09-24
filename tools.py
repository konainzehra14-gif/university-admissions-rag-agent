"""
tools.py
--------
Individual tool functions the agent can call, plus their JSON-schema
definitions for Groq's tool-calling API.

Tools:
1. retrieve_knowledge_base
2. calculate_merit
3. web_search

The web_search tool uses Google News RSS.
No API key or credit card is required.
"""

import html
import re
import requests
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Tool 1: Knowledge-base retrieval
# ---------------------------------------------------------------------------
def make_retrieve_tool(rag_engine):
    """
    Wrap the existing RAGEngine retrieval so the agent can call it as a tool.
    """

    def retrieve_knowledge_base(query: str) -> str:
        try:
            chunks = rag_engine.retrieve(query)

            if not chunks:
                return "No relevant information found in the knowledge base."

            return "\n\n".join(
                f"[{c.get('source', 'kb')}] {c['text']}"
                for c in chunks
            )

        except Exception as e:
            return f"Knowledge-base search failed: {e}"

    return retrieve_knowledge_base


# ---------------------------------------------------------------------------
# Tool 2: Eligibility / merit calculator
# ---------------------------------------------------------------------------
def calculate_merit(
    matric_percent: float,
    inter_percent: float,
    test_percent: float,
    weight_matric: float = 0.10,
    weight_inter: float = 0.40,
    weight_test: float = 0.50,
) -> str:
    """
    Generic weighted-merit formula.

    Default weights:
        Matric = 10%
        Intermediate = 40%
        Entry Test = 50%

    These are illustrative defaults only.
    Actual university weightages may differ.
    """

    for name, val in [
        ("matric_percent", matric_percent),
        ("inter_percent", inter_percent),
        ("test_percent", test_percent),
    ]:
        if not (0 <= val <= 100):
            return f"Error: {name} must be between 0 and 100."

    merit = (
        matric_percent * weight_matric
        + inter_percent * weight_inter
        + test_percent * weight_test
    )

    return (
        f"Calculated merit score: {merit:.2f}%\n"
        f"Formula used: "
        f"matric×{weight_matric} + "
        f"inter×{weight_inter} + "
        f"test×{weight_test}.\n"
        f"Actual weightages differ by university — "
        f"confirm on the official admissions page."
    )


# ---------------------------------------------------------------------------
# Helper: strip HTML tags/entities out of a Google News RSS description.
# The raw <description> field is a chunk of HTML ("<a href=...>Title</a>
# &nbsp;&nbsp;<font ...>Source</font>"), which wastes tokens and confuses
# the model. We just want the plain text.
# ---------------------------------------------------------------------------
def _clean_html_snippet(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)     # strip tags
    text = html.unescape(text)                # &nbsp; -> ' ', &amp; -> '&', etc.
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Tool 3: Live Web Search using Google News RSS
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    Search current web/news information using Google News RSS.

    No API key required.
    No credit card required.

    Returns recent search results with:
        - title
        - source
        - publication date
        - URL
        - summary (plain text, HTML-stripped)
    """

    try:
        response = requests.get(
            "https://news.google.com/rss/search",
            params={
                "q": query,
                "hl": "en-PK",
                "gl": "PK",
                "ceid": "PK:en",
            },
            headers={
                "User-Agent": "Mozilla/5.0",
            },
            timeout=15,
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        items = root.findall(".//item")

        if not items:
            return (
                "No current web/news results were found "
                f"for: {query}"
            )

        results = []

        for item in items[:6]:

            title = _clean_html_snippet(
                item.findtext("title") or ""
            )

            link = (
                item.findtext("link") or ""
            ).strip()

            pub_date = (
                item.findtext("pubDate") or ""
            ).strip()

            description = _clean_html_snippet(
                item.findtext("description") or ""
            )

            source_element = item.find("source")

            source = ""

            if source_element is not None:
                source = (
                    source_element.text or ""
                ).strip()

            if not title or not link:
                continue

            results.append(
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Date: {pub_date}\n"
                f"URL: {link}\n"
                f"Summary: {description[:300]}"
            )

        if not results:
            return (
                "Google News returned data, but no usable "
                f"results could be extracted for: {query}"
            )

        return "\n\n".join(results)

    except requests.exceptions.Timeout:
        return "Web search failed: request timed out."

    except requests.exceptions.RequestException as e:
        return f"Web search failed: {e}"

    except ET.ParseError as e:
        return f"Web search failed: invalid RSS response: {e}"

    except Exception as e:
        return f"Web search failed: {e}"


# ---------------------------------------------------------------------------
# Groq tool definitions
# ---------------------------------------------------------------------------
TOOLS = [

    # -----------------------------------------------------------------------
    # Knowledge Base
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge_base",
            "description": (
                "Search the local admissions knowledge base for "
                "admissions requirements, eligibility rules, "
                "general policies, documents, and university information "
                "already stored in the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },

    # -----------------------------------------------------------------------
    # Merit Calculator
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "calculate_merit",
            "description": (
                "Calculate a weighted admission merit score from "
                "Matric percentage, Intermediate percentage, and "
                "entry test percentage. Use this when the user "
                "provides numeric marks or percentages and asks "
                "about merit or eligibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "matric_percent": {
                        "type": "number",
                        "description": "Matric percentage.",
                    },

                    "inter_percent": {
                        "type": "number",
                        "description": "Intermediate percentage.",
                    },

                    "test_percent": {
                        "type": "number",
                        "description": "Entry test percentage.",
                    },

                    "weight_matric": {
                        "type": "number",
                        "description": "Optional Matric weight. Default 0.10.",
                    },

                    "weight_inter": {
                        "type": "number",
                        "description": (
                            "Optional Intermediate weight. "
                            "Default 0.40."
                        ),
                    },

                    "weight_test": {
                        "type": "number",
                        "description": (
                            "Optional entry test weight. "
                            "Default 0.50."
                        ),
                    },
                },

                "required": [
                    "matric_percent",
                    "inter_percent",
                    "test_percent",
                ],
            },
        },
    },

    # -----------------------------------------------------------------------
    # Live Web Search
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search current web/news information using Google News RSS. "
                "Use this when the user asks for latest, current, 2026, "
                "today's, recent, updated, deadline, current fee, "
                "latest announcement, or other time-sensitive information. "
                "Call this AT MOST ONCE per question — if the first search "
                "doesn't have the answer, say so rather than searching again "
                "with slightly different wording."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A concise web search query."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Build tool functions
# ---------------------------------------------------------------------------
def build_tool_functions(rag_engine):
    """
    Return the name -> callable mapping used by Agent.
    """

    return {
        "retrieve_knowledge_base": make_retrieve_tool(
            rag_engine
        ),

        "calculate_merit": calculate_merit,

        "web_search": web_search,
    }