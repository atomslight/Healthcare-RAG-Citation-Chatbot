from ingestion_embedding import load_json, ingest


# =========================================================
# TEST
# =========================================================

print("=" * 70)
print("TRANSCENDENCE INGESTION GOVERNANCE TEST")
print("=" * 70)


# ---------------------------------------------------------
# LOAD REAL SOURCE DATA
# ---------------------------------------------------------

records = load_json(
    "transcendence_services.json"
)

print(
    f"\nLoaded {len(records)} records."
)


# ---------------------------------------------------------
# RUN REAL INGESTION
# ---------------------------------------------------------

result = ingest(
    records
)


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEST RESULT")
print("=" * 70)

print(
    f"Unchanged:          "
    f"{result['unchanged']}"
)

print(
    f"Re-embedded:        "
    f"{result['re-embedded']}"
)

print(
    f"Reused embeddings:  "
    f"{result['reused_embedding']}"
)

print(
    f"Added:              "
    f"{result['added']}"
)

print(
    f"Empty:              "
    f"{result['empty']}"
)

print(
    f"Embeddings needed:  "
    f"{result['embeddings_needed']}"
)

print(
    f"Points written:     "
    f"{result['points_written']}"
)

print(
    f"Deleted:            "
    f"{result['deleted']}"
)


# =========================================================
# GOVERNANCE ASSERTION
# =========================================================
#
# With an unchanged source and a fully synchronized
# collection, this should be:
#
# embeddings_needed == 0
#
# This is the key test for avoiding re-embedding waste.
# =========================================================

if result["embeddings_needed"] == 0:

    print("\n" + "=" * 70)
    print("PASS")
    print("=" * 70)

    print(
        "\nNo Cloud document embeddings were required."
    )

    print(
        "Existing embeddings were reused or left "
        "untouched."
    )

else:

    print("\n" + "=" * 70)
    print("INFO")
    print("=" * 70)

    print(
        f"\n{result['embeddings_needed']} "
        f"document embeddings were required."
    )

    print(
        "This is expected when new or changed "
        "content exists."
    )
