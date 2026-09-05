import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from google import genai
from groq import Groq


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

JSON_FILE = Path("transcendence_services.json")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "transcendence_health"

EMBED_MODEL = "intfloat/multilingual-e5-small"

GEMINI_MODEL = "gemini-2.5-flash-lite"
GROQ_MODEL = "llama-3.1-8b-instant"

RRF_K = 60


# =========================================================
# VALIDATION
# =========================================================

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set.")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set.")


# =========================================================
# LOAD JSON
# =========================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list.")

    return data


records = load_json(JSON_FILE)

print(f"Loaded {len(records)} records.")


# =========================================================
# TEXT / METADATA
# =========================================================

def build_text(item):
    fields = [
        item.get("type"),
        item.get("service"),
        item.get("section"),
        item.get("title"),
        item.get("content", "")
    ]

    return "\n".join(
        str(value)
        for value in fields
        if value
    )


def build_metadata(item):
    return {
        key: value
        for key, value in item.items()
        if key != "content"
    }


texts = [build_text(item) for item in records]
metadata_list = [build_metadata(item) for item in records]


# =========================================================
# EMBEDDINGS
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded.")


# =========================================================
# QDRANT
# =========================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Qdrant client ready.")


# =========================================================
# BM25
# =========================================================

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")


# =========================================================
# LLM CLIENTS
# =========================================================

gemini = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

groq = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# =========================================================
# NORMALIZE
# =========================================================

def normalize(text):
    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        text.lower()
    )


def contains_any(text, terms):
    return any(term in text for term in terms)


# =========================================================
# AVAILABLE ROUTES
# =========================================================

VALID_ROUTES = {
    "services_overview",
    "service",
    "contact",
    "booking",
    "insurance",
    "unknown"
}

VALID_SERVICES = {
    "acupuncture",
    "chiropractic"
}

VALID_SECTIONS = {
    "overview",
    "effectiveness",
    "how_it_works",
    "candidate",
    "treatment_experience",
    "conditions",
    "home_care",
    "packages",
    "joints",
    "other_therapy",
    "insurance",
    "hours"
}


# =========================================================
# PYTHON ROUTER
# =========================================================

def route_query(query):

    q = normalize(query)

    # -----------------------------------------------------
    # SERVICES OVERVIEW
    # -----------------------------------------------------

    if contains_any(q, [
        "what services",
        "what all services",
        "what services do you have",
        "what do you offer",
        "what all do you offer",
        "services do you offer",
        "services available",
        "available services",
        "which services",
        "what can you help with"
    ]):
        return {
            "route": "services_overview",
            "service": None,
            "section": None,
            "clarification_needed": False,
            "clarification_question": ""
        }

    # -----------------------------------------------------
    # CONTACT
    # -----------------------------------------------------

    if contains_any(q, [
        "contact",
        "email",
        "phone",
        "telephone",
        "address",
        "location",
        "office hours",
        "hours",
        "open",
        "close",
        "reach you",
        "how can i reach"
    ]):
        return {
            "route": "contact",
            "service": None,
            "section": None,
            "clarification_needed": False,
            "clarification_question": ""
        }

    # -----------------------------------------------------
    # BOOKING
    # -----------------------------------------------------

    if contains_any(q, [
        "book",
        "booking",
        "appointment",
        "schedule",
        "reserve"
    ]):
        return {
            "route": "booking",
            "service": None,
            "section": None,
            "clarification_needed": False,
            "clarification_question": ""
        }

    # -----------------------------------------------------
    # INSURANCE
    # -----------------------------------------------------

    if contains_any(q, [
        "insurance",
        "coverage",
        "covered",
        "medicare",
        "anthem",
        "unitedhealthcare",
        "medical mutual",
        "hsa",
        "fsa",
        "superbill"
    ]):

        service = None

        if contains_any(q, [
            "chiropractic",
            "chiropractor",
            "manipulation"
        ]):
            service = "chiropractic"

        elif "acupuncture" in q:
            service = "acupuncture"

        return {
            "route": "insurance",
            "service": service,
            "section": "insurance",
            "clarification_needed": False,
            "clarification_question": ""
        }

    # -----------------------------------------------------
    # ACUPUNCTURE
    # -----------------------------------------------------

    if contains_any(q, [
        "acupuncture",
        "acupunctur"
    ]):

        section = None

        if contains_any(q, [
            "how does",
            "how it work"
        ]):
            section = "how_it_works"

        elif contains_any(q, [
            "how long",
            "duration",
            "expect",
            "painful",
            "needles",
            "treatment"
        ]):
            section = "treatment_experience"

        elif contains_any(q, [
            "treat",
            "condition",
            "migraine",
            "headache",
            "back pain",
            "allergy",
            "carpal",
            "vertigo",
            "fatigue"
        ]):
            section = "conditions"

        elif contains_any(q, [
            "candidate",
            "right for me"
        ]):
            section = "candidate"

        elif contains_any(q, [
            "home",
            "at home"
        ]):
            section = "home_care"

        elif contains_any(q, [
            "package",
            "discount"
        ]):
            section = "packages"

        return {
            "route": "service",
            "service": "acupuncture",
            "section": section,
            "clarification_needed": False,
            "clarification_question": ""
        }

    # -----------------------------------------------------
    # CHIROPRACTIC
    # -----------------------------------------------------

    if contains_any(q, [
        "chiropractic",
        "chiropractor",
        "manipulation",
        "adjustment"
    ]):

        section = None

        if contains_any(q, [
            "joint",
            "joints",
            "knee",
            "hip",
            "shoulder",
            "wrist",
            "elbow",
            "ankle",
            "foot",
            "spine",
            "tmj"
        ]):
            section = "joints"

        elif contains_any(q, [
            "cupping",
            "graston",
            "iastm",
            "manual therapy",
            "exercise",
            "kinesio",
            "taping",
            "dry needling"
        ]):
            section = "other_therapy"

        return {
            "route": "service",
            "service": "chiropractic",
            "section": section,
            "clarification_needed": False,
            "clarification_question": ""
        }

    return None


# =========================================================
# GEMINI ROUTER
# =========================================================

ROUTER_PROMPT = """
You are the routing assistant for the Transcendence Health website.

The available information contains:
- Services overview
- Acupuncture
- Chiropractic
- Contact information
- Booking information
- Insurance information

Available routes:

services_overview
service
contact
booking
insurance
unknown

Available services:
- acupuncture
- chiropractic

Possible sections:
- overview
- effectiveness
- how_it_works
- candidate
- treatment_experience
- conditions
- home_care
- packages
- joints
- other_therapy
- insurance
- hours

IMPORTANT RULES:

1. "What services do you offer?"
   MUST return:
   route = services_overview

2. "Tell me about acupuncture"
   MUST return:
   route = service
   service = acupuncture

3. "Tell me about chiropractic"
   MUST return:
   route = service
   service = chiropractic

4. "What can you do for back pain?"
   May return:
   route = service
   with a relevant service if the wording clearly points to one.
   Otherwise ask for clarification.

5. If the question is unrelated to Transcendence Health,
   return route = unknown.

6. If the user is vague but clearly asking about one of the
   available categories, ask a concise clarification question.

7. Never answer the user. Only classify.

8. ALWAYS return ALL fields below.

Return ONLY JSON:

{
  "route": "services_overview|service|contact|booking|insurance|unknown",
  "service": "acupuncture|chiropractic|null",
  "section": "overview|effectiveness|how_it_works|candidate|treatment_experience|conditions|home_care|packages|joints|other_therapy|insurance|hours|null",
  "clarification_needed": false,
  "clarification_question": ""
}
"""


def clean_json(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return text


def normalize_route(data):

    route = data.get("route")

    if route not in VALID_ROUTES:

        # Recover cases where Gemini only returned
        # service="acupuncture"
        if data.get("service") in VALID_SERVICES:
            route = "service"
        else:
            route = "unknown"

    service = data.get("service")

    if service not in VALID_SERVICES:
        service = None

    section = data.get("section")

    if section not in VALID_SECTIONS:
        section = None

    clarification_needed = bool(
        data.get("clarification_needed", False)
    )

    clarification_question = str(
        data.get(
            "clarification_question",
            ""
        )
    ).strip()

    # Services overview should never require clarification.
    if route == "services_overview":
        clarification_needed = False
        clarification_question = ""

    # Service route must have a service.
    if route == "service" and not service:
        route = "unknown"

    return {
        "route": route,
        "service": service,
        "section": section,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question
    }


def gemini_route(query, history):

    if not gemini:
        return None

    recent_history = "\n".join(
        history[-6:]
    )

    prompt = f"""
{ROUTER_PROMPT}

RECENT CONVERSATION:
{recent_history}

USER:
{query}
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw = clean_json(
        response.text
    )

    data = json.loads(raw)

    return normalize_route(data)


# =========================================================
# RESOLVE ROUTE
# =========================================================

def resolve_route(query, history):

    # First deterministic routing.
    python_route = route_query(query)

    if python_route:
        return python_route

    # Ambiguous → Gemini
    try:

        route = gemini_route(
            query,
            history
        )

        if route:
            return route

    except Exception as error:

        print(
            f"Gemini routing failed: {error}"
        )

    return {
        "route": "unknown",
        "service": None,
        "section": None,
        "clarification_needed": False,
        "clarification_question": ""
    }


# =========================================================
# FILTER CANDIDATES
# =========================================================

def filter_candidates(route):

    route_name = route["route"]
    service = route.get("service")
    section = route.get("section")

    candidates = []

    # ----------------------------------------------
    # SERVICES OVERVIEW
    # ----------------------------------------------

    if route_name == "services_overview":

        for i, metadata in enumerate(
            metadata_list
        ):

            if (
                metadata.get("type")
                == "service"
                and metadata.get("service")
            ):
                candidates.append(i)

        return candidates

    # ----------------------------------------------
    # CONTACT
    # ----------------------------------------------

    if route_name == "contact":

        return [
            i
            for i, metadata
            in enumerate(metadata_list)
            if metadata.get("type")
            == "contact"
        ]

    # ----------------------------------------------
    # BOOKING
    # ----------------------------------------------

    if route_name == "booking":

        return [
            i
            for i, metadata
            in enumerate(metadata_list)
            if (
                metadata.get("type")
                == "booking"
                or "booking"
                in metadata.get(
                    "actions",
                    []
                )
            )
        ]

    # ----------------------------------------------
    # INSURANCE
    # ----------------------------------------------

    if route_name == "insurance":

        for i, metadata in enumerate(
            metadata_list
        ):

            if metadata.get(
                "section"
            ) != "insurance":
                continue

            if service and metadata.get(
                "service"
            ) != service:
                continue

            candidates.append(i)

        return candidates

    # ----------------------------------------------
    # SERVICE
    # ----------------------------------------------

    if route_name == "service":

        for i, metadata in enumerate(
            metadata_list
        ):

            if metadata.get(
                "type"
            ) != "service":
                continue

            if metadata.get(
                "service"
            ) != service:
                continue

            if section and metadata.get(
                "section"
            ) != section:
                continue

            candidates.append(i)

        return candidates

    return []


# =========================================================
# DENSE SEARCH
# =========================================================

def dense_search(
    query,
    candidates
):

    if not candidates:
        return []

    query_vector = embeddings.embed_query(
        query
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=len(candidates),
        with_payload=True
    )

    allowed = set(candidates)

    results = []

    for rank, point in enumerate(
        response.points,
        start=1
    ):

        index = point.payload.get(
            "_record_index"
        )

        if index not in allowed:
            continue

        results.append({
            "record_index": index,
            "rank": rank,
            "score": point.score
        })

    return results


# =========================================================
# BM25 SEARCH
# =========================================================

def bm25_search(
    query,
    candidates
):

    if not candidates:
        return []

    query_tokens = normalize(
        query
    ).split()

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        candidates,
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        {
            "record_index": index,
            "rank": rank,
            "score": float(
                scores[index]
            )
        }
        for rank, index
        in enumerate(
            ranked,
            start=1
        )
    ]


# =========================================================
# RRF
# =========================================================

def rrf(
    dense_results,
    bm25_results
):

    fused = {}

    for result in dense_results:

        index = result[
            "record_index"
        ]

        fused.setdefault(
            index,
            {
                "record_index": index,
                "rrf_score": 0.0
            }
        )

        fused[index][
            "rrf_score"
        ] += (
            1 /
            (
                RRF_K
                + result["rank"]
            )
        )

    for result in bm25_results:

        index = result[
            "record_index"
        ]

        fused.setdefault(
            index,
            {
                "record_index": index,
                "rrf_score": 0.0
            }
        )

        fused[index][
            "rrf_score"
        ] += (
            1 /
            (
                RRF_K
                + result["rank"]
            )
        )

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )


# =========================================================
# RESULTS
# =========================================================

def build_results(
    fused
):

    results = []

    for result in fused:

        index = result[
            "record_index"
        ]

        results.append({
            "metadata":
                metadata_list[index],

            "text":
                records[index].get(
                    "content",
                    ""
                ),

            "rrf_score":
                result["rrf_score"]
        })

    return results


# =========================================================
# SERVICE OVERVIEW ANSWER
# =========================================================

def service_overview_answer():

    services = sorted({
        metadata.get("service")
        for metadata in metadata_list
        if (
            metadata.get("type")
            == "service"
            and metadata.get("service")
        )
    })

    readable = [
        service.replace(
            "_",
            " "
        ).title()
        for service in services
    ]

    return {
        "answer":
            "Transcendence Health currently offers "
            + " and ".join(readable)
            + " services.",
        "sources": [
            {
                "title":
                    metadata.get(
                        "title"
                    ),
                "url":
                    metadata.get(
                        "source_url"
                    )
            }
            for metadata
            in metadata_list
            if (
                metadata.get("type")
                == "service"
                and metadata.get(
                    "section"
                )
                == "overview"
            )
        ]
    }


# =========================================================
# LLM ANSWER PROMPTS
# =========================================================

ANSWER_PROMPT = """
You are the Transcendence Health website assistant.

Use ONLY the supplied Transcendence Health information.

Rules:
- Never invent information.
- Never use outside medical knowledge.
- Never diagnose.
- Never prescribe medication.
- Keep answers concise.
- Answer the actual user question.
- If multiple sources are relevant, combine them.
- If the sources do not answer the question, say:
  "I couldn't find that information in Transcendence Health's available information."
"""


def build_context(
    results
):

    blocks = []

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        blocks.append(
            f"""
SOURCE {i}
TITLE: {metadata.get('title')}
SERVICE: {metadata.get('service')}
SECTION: {metadata.get('section')}
URL: {metadata.get('source_url')}

CONTENT:
{result['text']}
"""
        )

    return "\n".join(blocks)


# =========================================================
# GEMINI ANSWER
# =========================================================

def gemini_answer(
    query,
    results,
    history
):

    if not gemini:
        raise RuntimeError(
            "Gemini API key not configured."
        )

    context = build_context(
        results
    )

    prompt = f"""
{ANSWER_PROMPT}

CONVERSATION:
{history[-6:]}

USER:
{query}

SOURCES:
{context}

Write only the final answer to the user.
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text.strip()


# =========================================================
# GROQ FALLBACK
# =========================================================

def groq_answer(
    query,
    results,
    history
):

    if not groq:
        raise RuntimeError(
            "Groq API key not configured."
        )

    context = build_context(
        results
    )

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=300,
        messages=[
            {
                "role":
                    "system",
                "content":
                    ANSWER_PROMPT
            },
            {
                "role":
                    "user",
                "content":
                    (
                        f"CONVERSATION:\n"
                        f"{history[-6:]}\n\n"
                        f"USER:\n"
                        f"{query}\n\n"
                        f"SOURCES:\n"
                        f"{context}"
                    )
            }
        ]
    )

    return response.choices[0].message.content.strip()


# =========================================================
# FINAL CHAT
# =========================================================

def chat(
    query,
    history
):

    route = resolve_route(
        query,
        history
    )

    print(
        f"\nROUTE: {route}"
    )

    # ---------------------------------------------
    # Clarification
    # ---------------------------------------------

    if route[
        "clarification_needed"
    ]:

        question = route[
            "clarification_question"
        ]

        return {
            "answer":
                question,
            "sources": []
        }

    # ---------------------------------------------
    # Unknown
    # ---------------------------------------------

    if route[
        "route"
    ] == "unknown":

        return {
            "answer":
                "I couldn't find that information in "
                "Transcendence Health's available information.",
            "sources": []
        }

    # ---------------------------------------------
    # Services overview
    # ---------------------------------------------

    if route[
        "route"
    ] == "services_overview":

        return service_overview_answer()

    # ---------------------------------------------
    # Candidates
    # ---------------------------------------------

    candidates = filter_candidates(
        route
    )

    print(
        f"Candidates: "
        f"{len(candidates)}"
    )

    if not candidates:

        return {
            "answer":
                "I couldn't find that information in "
                "Transcendence Health's available information.",
            "sources": []
        }

    # ---------------------------------------------
    # Dense + BM25
    # ---------------------------------------------

    dense_results = dense_search(
        query,
        candidates
    )

    bm25_results = bm25_search(
        query,
        candidates
    )

    fused = rrf(
        dense_results,
        bm25_results
    )

    results = build_results(
        fused
    )

    # ---------------------------------------------
    # If section was explicitly determined and only
    # one record exists, answer directly.
    # ---------------------------------------------

    if len(results) == 1:

        result = results[0]

        return {
            "answer":
                result["text"],

            "sources": [
                {
                    "title":
                        result[
                            "metadata"
                        ].get(
                            "title"
                        ),

                    "url":
                        result[
                            "metadata"
                        ].get(
                            "source_url"
                        )
                }
            ]
        }

    # ---------------------------------------------
    # Multiple records → synthesis
    # ---------------------------------------------

    try:

        answer = gemini_answer(
            query,
            results,
            history
        )

        return {
            "answer":
                answer,

            "sources": [
                {
                    "title":
                        r["metadata"].get(
                            "title"
                        ),
                    "url":
                        r["metadata"].get(
                            "source_url"
                        )
                }
                for r in results
            ]
        }

    except Exception as error:

        print(
            f"Gemini answer failed: "
            f"{error}"
        )

    # ---------------------------------------------
    # Groq fallback
    # ---------------------------------------------

    try:

        answer = groq_answer(
            query,
            results,
            history
        )

        return {
            "answer":
                answer,

            "sources": [
                {
                    "title":
                        r["metadata"].get(
                            "title"
                        ),
                    "url":
                        r["metadata"].get(
                            "source_url"
                        )
                }
                for r in results
            ]
        }

    except Exception as error:

        print(
            f"Groq answer failed: "
            f"{error}"
        )

    # ---------------------------------------------
    # Safe final fallback
    # ---------------------------------------------

    return {
        "answer":
            "I found relevant information, but I'm "
            "unable to generate the response right now. "
            "Please contact Transcendence Health directly.",

        "sources": []
    }


# =========================================================
# CONTINUOUS CHAT
# =========================================================

def run_chat():

    print("\n" + "=" * 90)
    print("TRANSCENDENCE HEALTH CHATBOT")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 90)

    history = []

    while True:

        try:

            query = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nChat ended."
            )

            break

        if not query:
            continue

        if query.lower() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nGoodbye!"
            )

            break

        try:

            result = chat(
                query,
                history
            )

            print(
                f"\nAssistant: "
                f"{result['answer']}"
            )

            if result[
                "sources"
            ]:

                print(
                    "\nSources:"
                )

                for source in result[
                    "sources"
                ]:

                    print(
                        f"- "
                        f"{source['title']}"
                    )

                    print(
                        f"  "
                        f"{source['url']}"
                    )

            history.append(
                f"User: {query}"
            )

            history.append(
                f"Assistant: "
                f"{result['answer']}"
            )

            # Keep conversation history bounded
            if len(history) > 12:
                history = history[-12:]

        except Exception as error:

            print(
                f"\nSystem error: "
                f"{error}"
            )

            # DO NOT terminate the chat.
            continue


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    run_chat()
