import os

from dotenv import load_dotenv
from google import genai
from groq import Groq

from search import search


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GEMINI_MODEL = (
    "gemini-2.5-flash-lite"
)

GROQ_MODEL = (
    "llama-3.1-8b-instant"
)


# =========================================================
# CLIENTS
# =========================================================

gemini = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY
    else None
)


groq = (
    Groq(
        api_key=GROQ_API_KEY
    )
    if GROQ_API_KEY
    else None
)


# =========================================================
# ANSWER PROMPT
# =========================================================

ANSWER_PROMPT = """
You are the Transcendence Health website assistant.

Use ONLY the supplied Transcendence Health information.

Rules:

- Never invent information.
- Never use outside medical knowledge.
- Never diagnose.
- Never prescribe medication.
- Answer the user's actual question.
- Keep the answer concise, natural, and helpful.
- Combine relevant retrieved information naturally.
- Do not mention Qdrant.
- Do not mention embeddings.
- Do not mention BM25.
- Do not mention RRF.
- Do not mention retrieval or ranking.
- Do not expose internal system details.

If the supplied information does not answer the question,
say exactly:

"I couldn't find that information in Transcendence Health's available information."

Write only the final answer to the user.
"""


# =========================================================
# CONTEXT
# =========================================================

def build_context(
    results
):

    if not results:
        return ""


    blocks = []


    for number, result in enumerate(
        results,
        start=1,
    ):

        metadata = (
            result.get(
                "metadata",
                {}
            )
        )


        blocks.append(
            f"""
SOURCE {number}

TITLE:
{metadata.get('title', '')}

SERVICE:
{metadata.get('service', '')}

SECTION:
{metadata.get('section', '')}

CONTENT:
{result.get('text', '')}
"""
        )


    return "\n".join(
        blocks
    )


# =========================================================
# GEMINI ANSWER
# =========================================================

def gemini_answer(
    query,
    results,
    history,
):

    if not gemini:

        raise RuntimeError(
            "Gemini API key not configured."
        )


    context = build_context(
        results
    )


    conversation = "\n".join(
        history[-6:]
    )


    prompt = f"""
{ANSWER_PROMPT}

CONVERSATION:
{conversation}

USER:
{query}

TRANSCENDENCE HEALTH INFORMATION:
{context}

Write only the final answer to the user.
"""


    response = (
        gemini
        .models
        .generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    )


    return response.text.strip()


# =========================================================
# GROQ FALLBACK
# =========================================================

def groq_answer(
    query,
    results,
    history,
):

    if not groq:

        raise RuntimeError(
            "Groq API key not configured."
        )


    context = build_context(
        results
    )


    conversation = "\n".join(
        history[-6:]
    )


    response = (
        groq
        .chat
        .completions
        .create(

            model=GROQ_MODEL,

            temperature=0.1,

            max_tokens=300,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        ANSWER_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            f"CONVERSATION:\n"
                            f"{conversation}\n\n"

                            f"USER:\n"
                            f"{query}\n\n"

                            f"TRANSCENDENCE HEALTH "
                            f"INFORMATION:\n"
                            f"{context}"
                        ),
                },
            ],
        )
    )


    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    query,
    results,
    history,
):

    if not results:

        return (
            "I couldn't find that information in "
            "Transcendence Health's available information."
        )


    # -----------------------------------------------------
    # GEMINI PRIMARY
    # -----------------------------------------------------

    try:

        return gemini_answer(
            query,
            results,
            history,
        )


    except Exception as error:

        print(
            f"\nGemini generation failed: "
            f"{error}"
        )


    # -----------------------------------------------------
    # GROQ FALLBACK
    # -----------------------------------------------------

    try:

        return groq_answer(
            query,
            results,
            history,
        )


    except Exception as error:

        print(
            f"\nGroq generation failed: "
            f"{error}"
        )


    # -----------------------------------------------------
    # SAFE FALLBACK
    # -----------------------------------------------------

    return (
        "I found relevant information, but I'm "
        "unable to generate the response right now. "
        "Please contact Transcendence Health directly."
    )


# =========================================================
# CHAT
# =========================================================

def chat(
    query,
    history,
):

    retrieval = search(
        query,
        history,
    )


    route = retrieval.get(
        "route"
    )

    results = retrieval.get(
        "results",
        []
    )


    print(
        f"\nROUTE: {route}"
    )

    print(
        f"RESULTS FOUND: "
        f"{len(results)}"
    )


    # -----------------------------------------------------
    # UNKNOWN
    # -----------------------------------------------------

    if (
        route
        and route.get(
            "route"
        )
        == "unknown"
    ):

        return {
            "answer":
                "I couldn't find that information in "
                "Transcendence Health's available information.",

            "sources":
                [],
        }


    # -----------------------------------------------------
    # CLARIFICATION
    # -----------------------------------------------------

    if (
        route
        and route.get(
            "clarification_needed"
        )
    ):

        return {
            "answer":
                route.get(
                    "clarification_question",
                    "",
                ),

            "sources":
                [],
        }


    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    answer = generate_answer(
        query,
        results,
        history,
    )


    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    sources = []

    seen = set()


    for result in results:

        metadata = (
            result.get(
                "metadata",
                {}
            )
        )


        title = metadata.get(
            "title"
        )

        url = metadata.get(
            "source_url"
        )


        key = (
            title,
            url,
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        sources.append(
            {
                "title":
                    title,

                "url":
                    url,
            }
        )


    return {
        "answer":
            answer,

        "sources":
            sources,
    }


# =========================================================
# CHAT LOOP
# =========================================================

def run_chat():

    print(
        "\n" + "=" * 90
    )

    print(
        "TRANSCENDENCE HEALTH CHATBOT"
    )

    print(
        "Type 'exit', 'quit', or 'bye' to stop."
    )

    print(
        "=" * 90
    )


    history = []


    while True:

        try:

            query = input(
                "\nYou: "
            ).strip()


        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nChat ended."
            )

            break


        if not query:
            continue


        if query.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print(
                "\nGoodbye!"
            )

            break


        try:

            result = chat(
                query,
                history,
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


                for source in (
                    result[
                        "sources"
                    ]
                ):

                    print(
                        f"- "
                        f"{source['title']}"
                    )


                    if source["url"]:

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


            if len(history) > 12:

                history = history[-12:]


        except Exception as error:

            print(
                f"\nSystem error: "
                f"{error}"
            )

            continue


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    run_chat()
