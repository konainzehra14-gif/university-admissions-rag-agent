"""
rag_engine.py
-------------
Core retrieval-augmented generation logic.

METHODOLOGY (hybrid BM25 + keyword boosting + two-stage re-ranking)
---------------------------------------------------------------------
Stage 1 (candidate retrieval):
    - Score every chunk with Okapi BM25 (rank_bm25) over the tokenized
      query — a keyword-frequency ranking function (the same family of
      algorithm used by Elasticsearch/Lucene), stronger than plain
      TF-IDF cosine similarity for short, keyword-heavy queries.
    - Add a keyword-boost bonus: if the query contains an exact domain
      acronym/term (e.g. "MDCAT", "IBCC", "ECAT") that also appears in a
      chunk's extracted keyword set, that chunk's score gets boosted.
    - Take the top CANDIDATE_POOL (e.g. 10) chunks by this hybrid score.

Stage 2 (re-ranking):
    - Within that candidate pool, re-rank using an exact-phrase-overlap
      score (shared bigrams between the query and each chunk) combined
      with the stage-1 hybrid score, then keep the final TOP_K (e.g. 4).
    - This two-stage design lets stage 1 cast a wider, more forgiving net
      (recall) while stage 2 sharpens precision on the finalists.

This avoids any dependency on PyTorch/neural embeddings, which can fail to
load on locked-down/managed Windows machines (antivirus/EDR blocking,
missing admin rights to install the Visual C++ runtime, etc.).

OUTPUT FORMAT
-------------
Every answer is asked (via SYSTEM_PROMPT) to come back with criteria-wise
markdown headings (Eligibility / Entry Test / Documents / Deadlines) and,
when a specific university is identifiable, a clickable link to that
university's official website — pulled from the trusted UNIVERSITY_LINKS
map below, never invented by the model. See build_official_links_block().
"""

import os
import re
import pickle

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from groq import Groq

load_dotenv()

INDEX_DIR = "index"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"  # fast Groq-hosted model; swap as needed

CANDIDATE_POOL = 20   # stage 1: how many candidates BM25+boost pulls in
TOP_K = 10              # stage 2: how many final chunks go to the LLM — bumped
                        # from 4 so generic (non-university-named) sections
                        # like "General Eligibility"/"Documents"/"Timeline"
                        # still make it into context even when the query
                        # names a specific university not mentioned in them.
KEYWORD_BOOST_WEIGHT = 1.5   # bonus per matched domain keyword (stage 1)
PHRASE_MATCH_WEIGHT = 2.0    # bonus per shared bigram (stage 2 re-rank)

# ---------------------------------------------------------------------------
# Trusted official-website map. Kept in code (not left to the model) so the
# chatbot never hallucinates a URL. Mirrors the list shown on the
# Universities page in app.py — keep the two in sync when adding entries.
# Keys are matched case-insensitively against both the university's full
# name and common short forms/acronyms that show up in the knowledge base
# and in user questions.
# ---------------------------------------------------------------------------
UNIVERSITY_LINKS = {
    "quaid-i-azam university": "https://qau.edu.pk",
    "qau": "https://qau.edu.pk",
    "lahore university of management sciences": "https://lums.edu.pk",
    "lums": "https://lums.edu.pk",
    "national university of sciences & technology": "https://nust.edu.pk",
    "national university of sciences and technology": "https://nust.edu.pk",
    "nust": "https://nust.edu.pk",
    "university of the punjab": "https://pu.edu.pk",
    "punjab university": "https://pu.edu.pk",
    "pu": "https://pu.edu.pk",
    "aga khan university": "https://aku.edu",
    "aku": "https://aku.edu",
    "institute of business administration": "https://iba.edu.pk",
    "iba": "https://iba.edu.pk",
    "fast national university": "https://nu.edu.pk",
    "fast-nu": "https://nu.edu.pk",
    "fast": "https://nu.edu.pk",
    "ghulam ishaq khan institute": "https://giki.edu.pk",
    "giki": "https://giki.edu.pk",
    "university of engineering and technology": "https://uet.edu.pk",
    "uet": "https://uet.edu.pk",
    "king edward medical university": "https://kemu.edu.pk",
    "kemu": "https://kemu.edu.pk",
    "dow university of health sciences": "https://duhs.edu.pk",
    "duhs": "https://duhs.edu.pk",
    "comsats university": "https://comsats.edu.pk",
    "comsats": "https://comsats.edu.pk",
}

SYSTEM_PROMPT = """You are the University Admissions Guide, a helpful chatbot for
students in Pakistan asking about university admissions — eligibility
criteria, entry tests, deadlines, required documents, merit lists, and
scholarships — using ONLY the provided context.

Rules:
- Answer using only the information in the CONTEXT section below.
- This is GENERIC, general-purpose guidance, not official information for
  any single university. Always remind the student to double-check exact
  dates, criteria, and merit formulas on the specific university's or HEC's
  official website, since these change every year and vary by institute.
- If the answer isn't in the context, say you don't have that information
  rather than guessing, and suggest the student check the relevant
  university's official admissions office or HEC/IBCC/NTS website.
- Be concise, clear, and encouraging — many students using this are first-time
  applicants who may feel overwhelmed by the process.

FORMAT — follow this every time the question is about a university's
admission process (skip a heading only if the context truly has nothing
for it):

## Eligibility
...

## Entry Test
...

## Documents
...

## Deadlines
...

At the very end, if an OFFICIAL LINKS section is provided below, add a line:
🔗 Official website: [<university name>](<url>) — always use the exact
markdown link given to you there. Never invent, guess, or modify a URL
yourself. If no official link is provided for the university being
discussed, skip this line instead of making one up.
"""


def tokenize(text):
    """Same simple tokenizer used at ingest time: lowercase, strip
    punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


KEYWORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def extract_query_keywords(query):
    return set(KEYWORD_RE.findall(query))


def bigrams(tokens):
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def find_official_links(text: str):
    """Case-insensitive whole-word-ish match of text against
    UNIVERSITY_LINKS, returning {display_name: url} for every hit —
    checked against both the retrieved context and the user's question,
    so a link surfaces even if the chunk uses a different short form than
    the query did."""
    lowered = text.lower()
    found = {}
    for key, url in UNIVERSITY_LINKS.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, lowered):
            display = key.upper() if len(key) <= 5 else key.title()
            found[display] = url
    return found


def build_official_links_block(query: str, retrieved_text: str) -> str:
    """Build the 'OFFICIAL LINKS' block appended to the system prompt's
    context, so the model only ever echoes a URL we already trust."""
    links = {}
    links.update(find_official_links(retrieved_text))
    links.update(find_official_links(query))  # query match wins on name casing
    if not links:
        return ""
    lines = [f"- [{name}]({url})" for name, url in links.items()]
    return "\n\nOFFICIAL LINKS (only use these, exactly as given):\n" + "\n".join(lines)


class RAGEngine:
    def __init__(self, index_dir: str = INDEX_DIR, groq_api_key: str = None):
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        tokens_path = os.path.join(index_dir, "tokens.pkl")
        keywords_path = os.path.join(index_dir, "keywords.pkl")

        for p in (chunks_path, metadata_path, tokens_path, keywords_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing '{p}'. Run `python ingest.py` first to build the index."
                )

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        with open(tokens_path, "rb") as f:
            self.tokens = pickle.load(f)
        with open(keywords_path, "rb") as f:
            self.keywords = pickle.load(f)

        self.bm25 = BM25Okapi(self.tokens)

        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY as an environment "
                "variable or in .streamlit/secrets.toml."
            )
        self.client = Groq(api_key=api_key)

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_tokens = tokenize(query)
        query_keywords = extract_query_keywords(query)
        query_bigrams = bigrams(query_tokens)

        # ---- Stage 1: BM25 + keyword-boost hybrid score ----
        bm25_scores = self.bm25.get_scores(query_tokens)
        hybrid_scores = []
        for idx, base_score in enumerate(bm25_scores):
            boost = KEYWORD_BOOST_WEIGHT * len(query_keywords & self.keywords[idx])
            hybrid_scores.append(base_score + boost)

        candidate_indices = sorted(
            range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True
        )[:CANDIDATE_POOL]
        candidate_indices = [i for i in candidate_indices if hybrid_scores[i] > 0]

        if not candidate_indices:
            return []

        # ---- Stage 2: re-rank candidates by exact phrase (bigram) overlap ----
        reranked = []
        for idx in candidate_indices:
            chunk_bigrams = bigrams(self.tokens[idx])
            phrase_bonus = PHRASE_MATCH_WEIGHT * len(query_bigrams & chunk_bigrams)
            final_score = hybrid_scores[idx] + phrase_bonus
            reranked.append((idx, final_score))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        top = reranked[:top_k]

        results = []
        for idx, score in top:
            results.append(
                {
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "section": self.metadata[idx].get("section", ""),
                    "score": float(score),
                }
            )
        return results

    def generate_answer(self, query: str, chat_history=None, top_k: int = TOP_K):
        """
        Generate an answer from retrieved knowledge-base context.

        For short follow-up questions, the latest previous user question is
        added to the retrieval query so questions such as "what about the
        test?" retain the university/topic context from the previous turn.
        """
        retrieval_query = query

        if chat_history:
            previous_user_messages = [
                m.get("content", "").strip()
                for m in chat_history
                if m.get("role") == "user" and m.get("content", "").strip()
            ]

            if previous_user_messages:
                previous_query = previous_user_messages[-1]
                q_lower = query.lower().strip()

                follow_up_markers = (
                    "what about",
                    "and what about",
                    "how about",
                    "what is the",
                    "what are the",
                    "when is the",
                    "when are the",
                    "which test",
                    "which one",
                    "what test",
                    "documents?",
                    "deadline?",
                    "fees?",
                    "scholarship?",
                )

                if (
                    len(query.split()) <= 8
                    or any(q_lower.startswith(marker) for marker in follow_up_markers)
                ):
                    retrieval_query = previous_query + " " + query

        retrieved = self.retrieve(retrieval_query, top_k=top_k)
        context_text = "\n\n---\n\n".join(
            f"[Source: {r['source']} | Section: {r['section']}]\n{r['text']}"
            for r in retrieved
        )

        links_block = build_official_links_block(query, context_text)
        system_content = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context_text + links_block

        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history[-6:])  # keep last few turns for context
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        answer = response.choices[0].message.content
        sources = sorted({r["source"] for r in retrieved})
        return answer, sources