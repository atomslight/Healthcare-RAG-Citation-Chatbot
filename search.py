import json
import os
import re

from dotenv import load_dotenv
from google import genai
from qdrant_client import models

from ingestion_embedding import (
    COLLECTION_NAME,
    DENSE_MODEL,
    BM25_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    create_client,
    ensure_payload_indexes,
)


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_ROUTER_MODEL = (
    "gemini-2.5-flash-lite"
)

DENSE_LIMIT = 20
SPARSE_LIMIT = 20
FINAL_LIMIT = 10


# =========================================================
# QDRANT
# =========================================================

client = create_client()


# =========================================================
# ROUTER CLIENT
# =========================================================

gemini_router = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY
    else None
)


# =========================================================
# ROUTES
# =========================================================

VALID_ROUTES = {
    "services_overview",
    "service",
    "contact",
    "booking",
    "insurance",
    "unknown",
}


VALID_SERVICES = {
    "acupuncture",
    "chiropractic",
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
    "other_therapies",
    "insurance",
    "hours",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(text):

    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        text.lower(),
    )


def contains_any(
    text,
    terms,
):

    return any(
        term in text
        for term in terms
    )


# =========================================================
# PYTHON ROUTER
# =========================================================

def route_query(query):

    q = normalize(
        query
    )


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
        "what can you help with",
    ]):

        return {
            "route":
                "services_overview",
            "service":
                None,
            "section":
                None,
            "clarification_needed":
                False,
            "clarification_question":
                "",
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
        "how can i reach",
    ]):

        return {
            "route":
                "contact",
            "service":
                None,
            "section":
                None,
            "clarification_needed":
                False,
            "clarification_question":
                "",
        }


    # -----------------------------------------------------
    # BOOKING
    # -----------------------------------------------------

    if contains_any(q, [
        "book",
        "booking",
        "appointment",
        "schedule",
        "reserve",
    ]):

        return {
            "route":
                "booking",
            "service":
                None,
            "section":
                None,
            "clarification_needed":
                False,
            "clarification_question":
                "",
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
        "superbill",
    ]):

        service = None


        if contains_any(q, [
            "chiropractic",
            "chiropractor",
            "manipulation",
        ]):

            service = (
                "chiropractic"
            )

        elif (
            "acupuncture"
            in q
        ):

            service = (
                "acupuncture"
            )


        return {
            "route":
                "insurance",
            "service":
                service,
            "section":
                "insurance",
            "clarification_needed":
                False,
            "clarification_question":
                "",
        }


    # -----------------------------------------------------
    # ACUPUNCTURE
    # -----------------------------------------------------

    if contains_any(q, [
        "acupuncture",
        "acupunctur",
    ]):

        section = None


        if contains_any(q, [
            "how does",
            "how it work",
        ]):

            section = (
                "how_it_works"
            )

        elif contains_any(q, [
            "how long",
            "duration",
            "expect",
            "painful",
            "needles",
            "treatment",
        ]):

            section = (
                "treatment_experience"
            )

        elif contains_any(q, [
            "treat",
            "condition",
            "migraine",
            "headache",
            "back pain",
            "allergy",
            "carpal",
            "vertigo",
            "fatigue",
        ]):

            section = (
                "conditions"
            )

        elif contains_any(q, [
            "candidate",
            "right for me",
        ]):

            section = (
                "candidate"
            )

        elif contains_any(q, [
            "home",
            "at home",
        ]):

            section = (
                "home_care"
            )

        elif contains_any(q, [
            "package",
            "discount",
        ]):

            section = (
                "packages"
            )


        return {
            "route":
                "service",
            "service":
                "acupuncture",
            "section":
                section,
            "clarification_needed":
                False,
            "clarification_question":
                "",
        }


    # -----------------------------------------------------
    # CHIROPRACTIC
    # -----------------------------------------------------

    if contains_any(q, [
        "chiropractic",
        "chiropractor",
        "manipulation",
        "adjustment",
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
            "tmj",
        ]):

            section = (
                "joints"
            )

        elif contains_any(q, [
            "cupping",
            "graston",
            "iastm",
            "manual therapy",
            "exercise",
            "kinesio",
            "taping",
            "dry needling",
        ]):

            section = (
                "other_therapy"
            )


        return {
            "route":
                "service",
            "service":
                "chiropractic",
            "section":
                section,
            "clarification_needed":
                False,
            "clarification_question":
                "",
        }


    return None


# =========================================================
# GEMINI ROUTER
# =========================================================

ROUTER_PROMPT = """
You are the routing assistant for the Transcendence Health website.

Available information:

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

acupuncture
chiropractic

Possible sections:

overview
effectiveness
how_it_works
candidate
treatment_experience
conditions
home_care
packages
joints
other_therapy
other_therapies
insurance
hours

Rules:

1. If the user asks what services are offered,
   return services_overview.

2. If the user clearly asks about acupuncture,
   return service + acupuncture.

3. If the user clearly asks about chiropractic,
   return service + chiropractic.

4. If the user's symptom clearly points toward one
   of the available services, classify it accordingly.

5. If unrelated to Transcendence Health,
   return unknown.

6. If clearly related but ambiguous,
   set clarification_needed=true and ask a concise
   clarification question.

7. Never answer the user.
   Only classify.

Return ONLY JSON:

{
  "route": "services_overview|service|contact|booking|insurance|unknown",
  "service": "acupuncture|chiropractic|null",
  "section": "overview|effectiveness|how_it_works|candidate|treatment_experience|conditions|home_care|packages|joints|other_therapy|other_therapies|insurance|hours|null",
  "clarification_needed": false,
  "clarification_question": ""
}
"""


# =========================================================
# CLEAN JSON
# =========================================================

def clean_json(text):

    text = text.strip()


    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
        )

        text = text.replace(
            "```",
            "",
        )

        text = text.strip()


    return text


# =========================================================
# NORMALIZE ROUTE
# =========================================================

def normalize_route(
    data
):

    route = data.get(
        "route"
    )


    if route not in VALID_ROUTES:

        if (
            data.get("service")
            in VALID_SERVICES
        ):

            route = (
                "service"
            )

        else:

            route = (
                "unknown"
            )


    service = data.get(
        "service"
    )


    if service not in VALID_SERVICES:

        service = None


    section = data.get(
        "section"
    )


    if section not in VALID_SECTIONS:

        section = None


    clarification_needed = bool(
        data.get(
            "clarification_needed",
            False,
        )
    )


    clarification_question = str(
        data.get(
            "clarification_question",
            "",
        )
    ).strip()


    return {
        "route":
            route,
        "service":
            service,
        "section":
            section,
        "clarification_needed":
            clarification_needed,
        "clarification_question":
            clarification_question,
    }


# =========================================================
# GEMINI ROUTE
# =========================================================

def gemini_route(
    query,
    history,
):

    if not gemini_router:
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


    response = (
        gemini_router
        .models
        .generate_content(
            model=GEMINI_ROUTER_MODEL,
            contents=prompt,
        )
    )


    raw = clean_json(
        response.text
    )


    return normalize_route(
        json.loads(raw)
    )


# =========================================================
# RESOLVE ROUTE
# =========================================================

def resolve_route(
    query,
    history=None,
):

    if history is None:
        history = []


    python_route = route_query(
        query
    )


    if python_route:

        return python_route


    try:

        route = gemini_route(
            query,
            history,
        )


        if route:

            return route


    except Exception as error:

        print(
            f"\nGemini routing failed: "
            f"{error}"
        )


    return {
        "route":
            "unknown",
        "service":
            None,
        "section":
            None,
        "clarification_needed":
            False,
        "clarification_question":
            "",
    }


# =========================================================
# FILTER
# =========================================================

def build_filter(
    route
):

    route_name = route[
        "route"
    ]

    service = route.get(
        "service"
    )

    section = route.get(
        "section"
    )


    conditions = []


    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    if (
        route_name
        == "services_overview"
    ):

        conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="service"
                ),
            )
        )


    # -----------------------------------------------------
    # CONTACT
    # -----------------------------------------------------

    elif (
        route_name
        == "contact"
    ):

        conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="contact"
                ),
            )
        )


    # -----------------------------------------------------
    # BOOKING
    # -----------------------------------------------------

    elif (
        route_name
        == "booking"
    ):

        conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="booking"
                ),
            )
        )


    # -----------------------------------------------------
    # INSURANCE
    # -----------------------------------------------------

    elif (
        route_name
        == "insurance"
    ):

        conditions.append(
            models.FieldCondition(
                key="section",
                match=models.MatchValue(
                    value="insurance"
                ),
            )
        )


        if service:

            conditions.append(
                models.FieldCondition(
                    key="service",
                    match=models.MatchValue(
                        value=service
                    ),
                )
            )


    # -----------------------------------------------------
    # SERVICE
    # -----------------------------------------------------

    elif (
        route_name
        == "service"
    ):

        conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="service"
                ),
            )
        )


        if service:

            conditions.append(
                models.FieldCondition(
                    key="service",
                    match=models.MatchValue(
                        value=service
                    ),
                )
            )


        if section:

            conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchValue(
                        value=section
                    ),
                )
            )


    if not conditions:

        return None


    return models.Filter(
        must=conditions
    )


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    query,
    route,
):

    query_filter = (
        build_filter(
            route
        )
    )


    # -----------------------------------------------------
    # DENSE QUERY
    # -----------------------------------------------------

    dense_prefetch = (
        models.Prefetch(

            query=models.Document(
                text=f"query: {query}",
                model=DENSE_MODEL,
            ),

            using=DENSE_VECTOR_NAME,

            limit=DENSE_LIMIT,

            filter=query_filter,
        )
    )


    # -----------------------------------------------------
    # BM25 QUERY
    # -----------------------------------------------------

    bm25_prefetch = (
        models.Prefetch(

            query=models.Document(
                text=query,
                model=BM25_MODEL,
            ),

            using=SPARSE_VECTOR_NAME,

            limit=SPARSE_LIMIT,

            filter=query_filter,
        )
    )


    # -----------------------------------------------------
    # RRF FUSION
    # -----------------------------------------------------

    response = (
        client.query_points(

            collection_name=
                COLLECTION_NAME,

            prefetch=[
                dense_prefetch,
                bm25_prefetch,
            ],

            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            limit=FINAL_LIMIT,

            with_payload=True,

            with_vectors=False,
        )
    )


    # -----------------------------------------------------
    # FORMAT RESULTS
    # -----------------------------------------------------

    results = []


    for rank, point in enumerate(
        response.points,
        start=1,
    ):

        payload = (
            point.payload or {}
        )


        results.append({

            "rank":
                rank,

            "record_index":
                payload.get(
                    "_record_index"
                ),

            "score":
                point.score,

            "metadata":
                payload,

            "text":
                payload.get(
                    "content",
                    "",
                ),

        })


    return results


# =========================================================
# SEARCH API
# =========================================================

def search(
    query,
    history=None,
):

    if (
        not query
        or not query.strip()
    ):

        return {
            "route":
                None,

            "results":
                [],
        }


    if history is None:

        history = []


    # Make sure filter indexes exist.
    ensure_payload_indexes(
        client
    )


    route = resolve_route(
        query,
        history,
    )


    # -----------------------------------------------------
    # CLARIFICATION
    # -----------------------------------------------------

    if (
        route[
            "clarification_needed"
        ]
    ):

        return {
            "route":
                route,

            "results":
                [],
        }


    # -----------------------------------------------------
    # UNKNOWN
    # -----------------------------------------------------

    if (
        route["route"]
        == "unknown"
    ):

        return {
            "route":
                route,

            "results":
                [],
        }


    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------

    try:

        results = hybrid_search(
            query,
            route,
        )


    except Exception as error:

        print(
            f"\nHybrid search failed: "
            f"{error}"
        )

        return {
            "route":
                route,

            "results":
                [],
        }


    return {
        "route":
            route,

        "results":
            results,
    }


# =========================================================
# DIRECT SEARCH TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "TRANSCENDENCE HEALTH HYBRID SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        "\nType 'exit' to stop."
    )


    history = []


    while True:

        query = input(
            "\nQuery: "
        ).strip()


        if query.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            break


        result = search(
            query,
            history,
        )


        print(
            "\nROUTE:"
        )

        print(
            result["route"]
        )


        print(
            "\nRESULTS:"
        )


        for item in (
            result["results"]
        ):

            metadata = (
                item["metadata"]
            )


            print(
                f"\n{item['rank']}. "
                f"{metadata.get('title')}"
            )

            print(
                f"   Service: "
                f"{metadata.get('service')}"
            )

            print(
                f"   Section: "
                f"{metadata.get('section')}"
            )

            print(
                f"   Score: "
                f"{item['score']}"
            )
