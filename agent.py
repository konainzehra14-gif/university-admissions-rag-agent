"""
agent.py
--------
A small tool-calling agent on top of Groq's chat completions API.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from tools import TOOLS, build_tool_functions


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL = "openai/gpt-oss-20b"

# 4 rounds, with the LAST round forced tool-free (tool_choice="none") so the
# model can never run out of rounds while still holding a pending tool call —
# it always produces a plain-text final answer by round 4 at the latest,
# using whatever it already found.
MAX_TOOL_ROUNDS = 4


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an agentic University Admissions Guide for Pakistan.

Available tools:
1. retrieve_knowledge_base - local admissions information
2. calculate_merit - calculate merit from marks/percentages
3. web_search - current/latest/time-sensitive information

Tool rules:
- Use web_search for latest, current, 2026, today, recent, updated,
  deadlines, current fees, or announcements.
- Use retrieve_knowledge_base for information covered by the local KB.
- Use calculate_merit when the user gives marks/percentages and asks
  about merit or eligibility.
- Call web_search AT MOST ONCE per question. Do not retry it with
  reworded queries even if the first result isn't a perfect match.
- Use the minimum tools needed.
- After getting sufficient information, give the final answer.

Answer rules:
- Base factual claims on tool results.
- Never invent fees, dates, eligibility rules, requirements, or policies.
- If a tool didn't return the exact answer, say plainly what you did find
  (if anything relevant), state clearly that you couldn't confirm the exact
  current detail, and point the student to the official university/HEC
  website — do NOT just say "I couldn't finalize an answer" with nothing
  else; always give the student something useful.
- For current information, prefer web_search results.
- For university-specific information, recommend the official university website.
- For HEC information, recommend the official HEC website.
- Be concise, clear, friendly, and professional.
- Your ENTIRE final answer must stay under 300 words total — this is a
  hard limit, not a suggestion.
- Use ONLY ONE structured format per answer: either a short table (max 5
  rows) OR a short bulleted list — never both a table AND a separate
  numbered "how to apply" section in the same answer.
- Prioritize the handful of facts a student actually needs to act (eligible
  %, test name, portal name, rough timing) over exhaustively covering every
  sub-topic. Point to the official website for anything beyond that.
- CRITICAL: only state a specific number (a fee amount, a cutoff
  percentage, an exact date) if that exact number appears in the tool
  results text you were given. If a tool result only mentions a topic in
  general terms (e.g. a news headline about "test dates released" without
  giving the actual date), describe it in general terms too — do not
  invent a plausible-sounding specific figure to fill the gap.
"""


class Agent:

    def __init__(self, groq_api_key: str = None, rag_engine=None):
        """
        Initialize the agent.

        groq_api_key:
            Groq API key. If not provided, GROQ_API_KEY
            is loaded from environment variables.

        rag_engine:
            Existing RAGEngine instance used by the
            retrieve_knowledge_base tool.
        """

        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required for the agent."
            )

        self.client = Groq(api_key=groq_api_key)

        # Build tool functions.
        self.tool_functions = build_tool_functions(rag_engine)

    def run(self, user_input: str, chat_history: list = None):
        """
        Run the agent.

        Returns:
            answer: final assistant response
            trace: list containing tool calls and their results
        """

        chat_history = chat_history or []

        # -------------------------------------------------------------------
        # Initial messages
        # -------------------------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Keep only recent conversation history.
        messages.extend(chat_history[-6:])

        # Current user question.
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        trace = []

        # Tools allowed only once per question (currently just web_search —
        # it's the slow, network-dependent one, and repeated reworded
        # searches were both wasting time and inviting the model to pad out
        # answers with unsupported specifics). Once called, we remove it
        # from the tool list entirely so the model CANNOT call it again,
        # rather than just asking it nicely in the prompt.
        ONCE_ONLY_TOOLS = {"web_search"}
        used_once_tools = set()

        # -------------------------------------------------------------------
        # Agent loop
        # -------------------------------------------------------------------
        for round_number in range(MAX_TOOL_ROUNDS):

            is_last_round = round_number == MAX_TOOL_ROUNDS - 1

            available_tools = [
                t for t in TOOLS
                if t["function"]["name"] not in used_once_tools
            ]

            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=available_tools,
                # Force a plain-text answer on the final round instead of
                # letting the model request yet another tool call and hit
                # the generic fallback message below.
                tool_choice="none" if is_last_round else "auto",
                temperature=0.2,
                # A final answer can be a full markdown table, which eats
                # far more tokens than a plain paragraph — 1600 gives real
                # headroom so answers don't get cut off mid-table. This
                # applies on every round since the model can decide to stop
                # calling tools and answer directly on any round, not just
                # the forced last one.
                max_tokens=1600,
            )

            choice = response.choices[0].message

            # ----------------------------------------------------------------
            # Final answer
            # ----------------------------------------------------------------
            if not getattr(choice, "tool_calls", None):
                return choice.content, trace

            # ----------------------------------------------------------------
            # Assistant requested tool(s)
            # ----------------------------------------------------------------
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        tc.model_dump()
                        for tc in choice.tool_calls
                    ],
                }
            )

            # ----------------------------------------------------------------
            # Execute requested tools
            # ----------------------------------------------------------------
            for tool_call in choice.tool_calls:

                name = tool_call.function.name

                # Parse arguments.
                try:
                    args = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                except json.JSONDecodeError:
                    args = {}

                # If this is a once-only tool (web_search) and it was
                # already used — either in an earlier round, or by another
                # parallel tool_call earlier in THIS SAME round's list —
                # skip actually running it again. This closes the gap where
                # a model requests two web_search calls in one response
                # before we get a chance to remove the tool for next round.
                already_used = (
                    name in ONCE_ONLY_TOOLS
                    and name in used_once_tools
                )

                if already_used:
                    result = (
                        "Skipped — web_search was already used once for "
                        "this question. Use the results already returned "
                        "above instead of searching again."
                    )
                else:
                    if name in ONCE_ONLY_TOOLS:
                        used_once_tools.add(name)

                    # Get Python function.
                    func = self.tool_functions.get(name)

                    # Execute tool.
                    if func is None:
                        result = f"Unknown tool: {name}"
                    else:
                        try:
                            result = func(**args)
                        except Exception as e:
                            result = (
                                f"Tool '{name}' failed: {e}"
                            )

                # Save tool call for debugging.
                trace.append(
                    {
                        "tool": name,
                        "args": args,
                        "result": result,
                        "skipped": already_used,
                    }
                )

                # ----------------------------------------------------------------
                # Limit tool result size.
                # ----------------------------------------------------------------
                result_text = str(result)

                if len(result_text) > 4000:
                    result_text = (
                        result_text[:4000]
                        + "\n[Result truncated]"
                    )

                # Send result back to the model.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

        # --------------------------------------------------------------------
        # Safety fallback — should be unreachable now since the last round
        # forces tool_choice="none", but kept just in case.
        # --------------------------------------------------------------------
        return (
            "I found some information but couldn't fully confirm the exact "
            "current details. Please check the official university or HEC "
            "website for the latest requirements.",
            trace,
        )