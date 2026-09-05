import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

JSON_FILE = Path("transcendence_services.json")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "transcendence_health"

DENSE_MODEL = "intfloat/multilingual-e5-small"
BM25_MODEL = "qdrant/bm25"

# Your existing collection currently has an unnamed
# dense vector. We preserve that vector and add a
# named sparse vector for BM25.
DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "bm25"

DENSE_SIZE = 384

PIPELINE_VERSION = "transcendence-hybrid-v1"


# =========================================================
# VALIDATION
# =========================================================

if not QDRANT_URL:
    raise ValueError(
        "QDRANT_URL is not set in .env"
    )

if not QDRANT_API_KEY:
    raise ValueError(
        "QDRANT_API_KEY is not set in .env"
    )

if not JSON_FILE.exists():
    raise FileNotFoundError(
        f"JSON file not found: {JSON_FILE.resolve()}"
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "JSON root must be a list."
        )

    return data


# =========================================================
# TEXT PREPARATION
# =========================================================

def build_text(item: dict) -> str:

    fields = [
        item.get("type"),
        item.get("service"),
        item.get("section"),
        item.get("title"),
        item.get("content", ""),
    ]

    return "\n".join(
        str(value).strip()
        for value in fields
        if value
    )


def build_dense_text(text: str) -> str:
    """
    Exact text sent to the E5 dense embedding model.
    """
    return f"passage: {text}"


def text_hash(text: str) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# =========================================================
# STABLE SOURCE IDENTITY
# =========================================================

def build_source_key(item: dict) -> str:

    identity = {
        "type": item.get("type"),
        "service": item.get("service"),
        "section": item.get("section"),
        "title": item.get("title"),
    }

    return json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_point_id(item: dict) -> str:

    source_key = build_source_key(item)

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"transcendence-health:{source_key}",
        )
    )


# =========================================================
# PAYLOAD
# =========================================================

def build_payload(
    item: dict,
    record_index: int,
    text: str,
    dense_text: str,
) -> dict:

    payload = {
        key: value
        for key, value in item.items()
        if key != "content"
    }

    payload["content"] = item.get(
        "content",
        "",
    )

    payload["_record_index"] = record_index

    payload["_source_key"] = (
        build_source_key(item)
    )

    payload["_embedding_model"] = (
        DENSE_MODEL
    )

    payload["_bm25_model"] = (
        BM25_MODEL
    )

    payload["_pipeline_version"] = (
        PIPELINE_VERSION
    )

    payload["_embedding_text"] = (
        dense_text
    )

    payload["_embedding_text_hash"] = (
        text_hash(dense_text)
    )

    return payload


# =========================================================
# QDRANT CLIENT
# =========================================================

def create_client() -> QdrantClient:

    print(
        "Connecting to Qdrant Cloud..."
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
        timeout=60,
    )

    print(
        "Qdrant Cloud connection ready."
    )

    return client


# =========================================================
# COLLECTION CHECK
# =========================================================

def collection_exists(
    client,
    collection_name,
):

    collections = (
        client.get_collections()
        .collections
    )

    return any(
        collection.name == collection_name
        for collection in collections
    )


# =========================================================
# ENSURE COLLECTION
# =========================================================

def ensure_collection(
    client: QdrantClient,
):

    # -----------------------------------------------------
    # COLLECTION DOES NOT EXIST
    # -----------------------------------------------------

    if not collection_exists(
        client,
        COLLECTION_NAME,
    ):

        print(
            f"Creating collection: "
            f"{COLLECTION_NAME}"
        )

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=models.VectorParams(
                size=DENSE_SIZE,
                distance=models.Distance.COSINE,
            ),

            sparse_vectors_config={
                SPARSE_VECTOR_NAME:
                    models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
            },
        )

        print(
            "Collection created."
        )

        return


    # -----------------------------------------------------
    # EXISTING COLLECTION
    # -----------------------------------------------------

    print(
        f"Collection already exists: "
        f"{COLLECTION_NAME}"
    )


    collection = client.get_collection(
        COLLECTION_NAME
    )


    sparse_config = (
        collection.config.params
        .sparse_vectors
    )


    if (
        sparse_config is not None
        and SPARSE_VECTOR_NAME
        in sparse_config
    ):

        print(
            f"Sparse vector '{SPARSE_VECTOR_NAME}' "
            f"already exists."
        )

        return


    # -----------------------------------------------------
    # ADD BM25 VECTOR TO EXISTING COLLECTION
    # -----------------------------------------------------

    print(
        f"Adding sparse vector "
        f"'{SPARSE_VECTOR_NAME}' "
        f"to existing collection..."
    )


    client.create_vector_name(

        collection_name=COLLECTION_NAME,

        vector_name=SPARSE_VECTOR_NAME,

        vector_name_config=(
            models.SparseVectorNameConfig(
                sparse=models.SparseVectorConfig(
                    modifier=models.Modifier.IDF
                )
            )
        ),
    )


    print(
        "BM25 sparse vector schema added."
    )


# =========================================================
# PAYLOAD INDEXES
# =========================================================

def ensure_payload_indexes(
    client: QdrantClient,
):

    fields = [
        "type",
        "service",
        "section",
        "actions",
    ]


    collection = client.get_collection(
        COLLECTION_NAME
    )

    existing_schema = (
        collection.payload_schema
        or {}
    )


    for field in fields:

        if field in existing_schema:
            continue


        print(
            f"Creating payload index: "
            f"{field}"
        )


        client.create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name=field,

            field_schema=(
                models.PayloadSchemaType.KEYWORD
            ),

            wait=True,
        )


    print(
        "Payload indexes ready."
    )


# =========================================================
# LOAD EXISTING POINTS
# =========================================================

def load_existing_points(
    client: QdrantClient,
):

    existing = {}

    offset = None


    while True:

        points, next_offset = client.scroll(

            collection_name=COLLECTION_NAME,

            offset=offset,

            limit=100,

            with_payload=True,

            with_vectors=True,
        )


        for point in points:

            payload = point.payload or {}

            existing[str(point.id)] = {
                "id":
                    str(point.id),

                "payload":
                    payload,

                "vector":
                    point.vector,
            }


        if next_offset is None:
            break


        offset = next_offset


    return existing


# =========================================================
# GET DENSE VECTOR
# =========================================================

def get_dense_vector(
    vector_data
):

    if vector_data is None:
        return None


    # Multiple-vector representation
    if isinstance(
        vector_data,
        dict,
    ):

        return vector_data.get(
            DENSE_VECTOR_NAME
        )


    # Existing unnamed-only collection
    return vector_data


# =========================================================
# BUILD LOOKUPS
# =========================================================

def build_existing_maps(
    existing_points,
):

    by_id = {}
    by_index = {}
    by_hash = {}


    for point_id, point in (
        existing_points.items()
    ):

        payload = point["payload"]


        by_id[point_id] = point


        record_index = payload.get(
            "_record_index"
        )

        if record_index is not None:

            by_index[
                record_index
            ] = point


        stored_hash = payload.get(
            "_embedding_text_hash"
        )


        # Legacy fallback:
        # calculate hash from stored embedding text
        # if the explicit hash wasn't stored.
        if not stored_hash:

            stored_embedding_text = (
                payload.get(
                    "_embedding_text"
                )
            )

            if stored_embedding_text:

                stored_hash = text_hash(
                    stored_embedding_text
                )


        if stored_hash:

            by_hash.setdefault(
                stored_hash,
                [],
            ).append(
                point
            )


    return (
        by_id,
        by_index,
        by_hash,
    )


# =========================================================
# VALIDATE SOURCE
# =========================================================

def validate_source_keys(
    records,
):

    seen = {}


    for index, item in enumerate(
        records
    ):

        source_key = (
            build_source_key(item)
        )


        if source_key in seen:

            previous = seen[
                source_key
            ]

            raise ValueError(
                "Duplicate source identity detected: "
                f"records {previous} and {index} "
                f"produce the same source key."
            )


        seen[source_key] = index


# =========================================================
# INGEST
# =========================================================

def ingest(records):

    if not records:

        raise ValueError(
            "Refusing to sync an empty "
            "source dataset."
        )


    validate_source_keys(
        records
    )


    client = create_client()

    ensure_collection(
        client
    )

    ensure_payload_indexes(
        client
    )


    print(
        "\nReading existing points..."
    )


    existing_points = (
        load_existing_points(
            client
        )
    )


    print(
        f"Found {len(existing_points)} "
        f"existing points."
    )


    (
        existing_by_id,
        existing_by_index,
        existing_by_hash,
    ) = build_existing_maps(
        existing_points
    )


    # -----------------------------------------------------
    # COUNTERS
    # -----------------------------------------------------

    unchanged = 0
    bm25_only = 0
    re_embedded = 0
    reused_embedding = 0
    added = 0
    empty = 0


    incoming_ids = set()
    points_to_upsert = []
    vector_only_updates = []


    # IDs that should remain after sync.
    desired_ids = set()


    # -----------------------------------------------------
    # PROCESS RECORDS
    # -----------------------------------------------------

    for index, item in enumerate(
        records
    ):

        text = build_text(
            item
        )


        if not text.strip():

            print(
                f"[SKIP] Empty record: "
                f"{index}"
            )

            empty += 1

            continue


        dense_text = (
            build_dense_text(
                text
            )
        )

        current_hash = text_hash(
            dense_text
        )

        source_key = (
            build_source_key(item)
        )

        desired_id = (
            build_point_id(item)
        )

        desired_ids.add(
            desired_id
        )


        # =================================================
        # CASE 1: DESIRED DETERMINISTIC ID EXISTS
        # =================================================

        if desired_id in existing_by_id:

            existing = existing_by_id[
                desired_id
            ]

            payload = existing[
                "payload"
            ]

            vector = existing[
                "vector"
            ]

            stored_hash = payload.get(
                "_embedding_text_hash"
            )

            if not stored_hash:

                stored_text = payload.get(
                    "_embedding_text"
                )

                if stored_text:

                    stored_hash = (
                        text_hash(
                            stored_text
                        )
                    )


            stored_model = payload.get(
                "_embedding_model"
            )


            dense_vector = (
                get_dense_vector(
                    vector
                )
            )


            has_bm25 = (
                isinstance(
                    vector,
                    dict,
                )
                and SPARSE_VECTOR_NAME
                in vector
            )


            # -------------------------------------------------
            # COMPLETELY CURRENT
            # -------------------------------------------------

            if (
                stored_hash == current_hash
                and stored_model == DENSE_MODEL
                and dense_vector is not None
                and has_bm25
            ):

                print(
                    f"[UNCHANGED] "
                    f"Record {index}"
                )

                unchanged += 1

                continue


            # -------------------------------------------------
            # DENSE CURRENT, BM25 MISSING
            # -------------------------------------------------

            if (
                stored_hash == current_hash
                and stored_model == DENSE_MODEL
                and dense_vector is not None
                and not has_bm25
            ):

                print(
                    f"[BM25 ONLY] "
                    f"Record {index}"
                )

                vector_only_updates.append(

                    models.PointVectors(

                        id=desired_id,

                        vector={
                            SPARSE_VECTOR_NAME:
                                models.Document(
                                    text=text,
                                    model=BM25_MODEL,
                                )
                        },
                    )
                )


                client.set_payload(

                    collection_name=COLLECTION_NAME,

                    payload=build_payload(
                        item,
                        index,
                        text,
                        dense_text,
                    ),

                    points=models.PointIdsList(
                        points=[desired_id]
                    ),

                    wait=True,
                )


                bm25_only += 1

                continue


            # -------------------------------------------------
            # CONTENT CHANGED
            # -------------------------------------------------

            print(
                f"[RE-EMBED] "
                f"Record {index}: "
                f"content changed."
            )


            vector_only_updates.append(

                models.PointVectors(

                    id=desired_id,

                    vector={

                        DENSE_VECTOR_NAME:
                            models.Document(
                                text=dense_text,
                                model=DENSE_MODEL,
                            ),

                        SPARSE_VECTOR_NAME:
                            models.Document(
                                text=text,
                                model=BM25_MODEL,
                            ),
                    },
                )
            )


            client.set_payload(

                collection_name=COLLECTION_NAME,

                payload=build_payload(
                    item,
                    index,
                    text,
                    dense_text,
                ),

                points=models.PointIdsList(
                    points=[desired_id]
                ),

                wait=True,
            )


            re_embedded += 1

            continue


        # =================================================
        # CASE 2: OLD POINT WITH SAME RECORD INDEX
        # =================================================

        if index in existing_by_index:

            existing = existing_by_index[
                index
            ]

            old_id = existing[
                "id"
            ]

            payload = existing[
                "payload"
            ]

            old_vector = existing[
                "vector"
            ]

            old_dense = (
                get_dense_vector(
                    old_vector
                )
            )


            old_hash = payload.get(
                "_embedding_text_hash"
            )

            if not old_hash:

                old_text = payload.get(
                    "_embedding_text"
                )

                if old_text:

                    old_hash = (
                        text_hash(
                            old_text
                        )
                    )


            old_model = payload.get(
                "_embedding_model"
            )


            # -------------------------------------------------
            # SAME CONTENT:
            # REUSE DENSE, GENERATE ONLY BM25
            # -------------------------------------------------

            if (
                old_hash == current_hash
                and old_dense is not None
                and (
                    old_model is None
                    or old_model
                    == DENSE_MODEL
                )
            ):

                print(
                    f"[MIGRATE/REUSE] "
                    f"Record {index}"
                )


                points_to_upsert.append(

                    models.PointStruct(

                        id=desired_id,

                        vector={
                            DENSE_VECTOR_NAME:
                                old_dense,

                            SPARSE_VECTOR_NAME:
                                models.Document(
                                    text=text,
                                    model=BM25_MODEL,
                                ),
                        },

                        payload=build_payload(
                            item,
                            index,
                            text,
                            dense_text,
                        ),
                    )
                )


                reused_embedding += 1

                continue


            # -------------------------------------------------
            # LEGACY CONTENT CHANGED
            # -------------------------------------------------

            print(
                f"[RE-EMBED] "
                f"Record {index}: "
                f"legacy content changed."
            )


            points_to_upsert.append(

                models.PointStruct(

                    id=desired_id,

                    vector={

                        DENSE_VECTOR_NAME:
                            models.Document(
                                text=dense_text,
                                model=DENSE_MODEL,
                            ),

                        SPARSE_VECTOR_NAME:
                            models.Document(
                                text=text,
                                model=BM25_MODEL,
                            ),
                    },

                    payload=build_payload(
                        item,
                        index,
                        text,
                        dense_text,
                    ),
                )
            )


            re_embedded += 1

            continue


        # =================================================
        # CASE 3: SAME CONTENT UNDER ANOTHER ID
        # =================================================

        matches = (
            existing_by_hash.get(
                current_hash,
                []
            )
        )


        reusable = None

        for candidate in matches:

            candidate_dense = (
                get_dense_vector(
                    candidate["vector"]
                )
            )

            if candidate_dense is not None:

                reusable = candidate

                break


        if reusable is not None:

            print(
                f"[REUSE] "
                f"Record {index}: "
                f"same content already exists."
            )


            candidate_vector = (
                reusable["vector"]
            )


            candidate_dense = (
                get_dense_vector(
                    candidate_vector
                )
            )


            candidate_bm25 = None


            if isinstance(
                candidate_vector,
                dict,
            ):

                candidate_bm25 = (
                    candidate_vector.get(
                        SPARSE_VECTOR_NAME
                    )
                )


            if candidate_bm25 is not None:

                sparse_vector = (
                    candidate_bm25
                )

            else:

                sparse_vector = (
                    models.Document(
                        text=text,
                        model=BM25_MODEL,
                    )
                )


            points_to_upsert.append(

                models.PointStruct(

                    id=desired_id,

                    vector={
                        DENSE_VECTOR_NAME:
                            candidate_dense,

                        SPARSE_VECTOR_NAME:
                            sparse_vector,
                    },

                    payload=build_payload(
                        item,
                        index,
                        text,
                        dense_text,
                    ),
                )
            )


            reused_embedding += 1

            continue


        # =================================================
        # CASE 4: GENUINELY NEW
        # =================================================

        print(
            f"[NEW] "
            f"Record {index}: "
            f"dense + BM25 embedding required."
        )


        points_to_upsert.append(

            models.PointStruct(

                id=desired_id,

                vector={

                    DENSE_VECTOR_NAME:
                        models.Document(
                            text=dense_text,
                            model=DENSE_MODEL,
                        ),

                    SPARSE_VECTOR_NAME:
                        models.Document(
                            text=text,
                            model=BM25_MODEL,
                        ),
                },

                payload=build_payload(
                    item,
                    index,
                    text,
                    dense_text,
                ),
            )
        )


        added += 1


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INGESTION DECISION"
    )

    print(
        "=" * 70
    )

    print(
        f"Unchanged:          {unchanged}"
    )

    print(
        f"BM25 only:          {bm25_only}"
    )

    print(
        f"Re-embedded:        {re_embedded}"
    )

    print(
        f"Reused embedding:   {reused_embedding}"
    )

    print(
        f"New:                {added}"
    )

    print(
        f"Empty:              {empty}"
    )

    print(
        f"Dense inference:    "
        f"{re_embedded + added}"
    )

    print(
        f"Points to upsert:   "
        f"{len(points_to_upsert)}"
    )

    print(
        f"Vector-only updates:"
        f" {len(vector_only_updates)}"
    )


    # =====================================================
    # WRITE COMPLETE POINTS
    # =====================================================

    if points_to_upsert:

        print(
            "\nWriting complete point updates..."
        )

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=points_to_upsert,

            wait=True,
        )

        print(
            "Complete point updates finished."
        )


    # =====================================================
    # WRITE VECTOR-ONLY UPDATES
    # =====================================================

    if vector_only_updates:

        print(
            "\nWriting vector-only updates..."
        )

        client.update_vectors(

            collection_name=COLLECTION_NAME,

            points=vector_only_updates,

            wait=True,
        )

        print(
            "Vector-only updates finished."
        )


    # =====================================================
    # DELETE STALE POINTS
    # =====================================================

    stale_ids = [
        point_id
        for point_id in existing_points
        if point_id not in desired_ids
    ]


    if stale_ids:

        print(
            f"\nDeleting "
            f"{len(stale_ids)} stale points..."
        )

        client.delete(

            collection_name=COLLECTION_NAME,

            points_selector=(
                models.PointIdsList(
                    points=stale_ids
                )
            ),

            wait=True,
        )

        print(
            "Stale points deleted."
        )

    else:

        print(
            "\nNo stale points to delete."
        )


    # =====================================================
    # RETURN
    # =====================================================

    return {
        "unchanged": unchanged,
        "bm25_only": bm25_only,
        "re-embedded": re_embedded,
        "reused_embedding":
            reused_embedding,
        "new": added,
        "empty": empty,
        "embeddings_needed":
            re_embedded + added,
        "points_written":
            len(points_to_upsert),
        "vector_only_updates":
            len(vector_only_updates),
        "deleted":
            len(stale_ids),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TRANSCENDENCE HEALTH HYBRID INGESTION"
    )

    print(
        "=" * 70
    )


    records = load_json(
        JSON_FILE
    )


    print(
        f"Loaded {len(records)} records."
    )


    result = ingest(
        records
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 70
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nIngestion cancelled."
        )

    except Exception as error:

        print(
            f"\nINGESTION ERROR: {error}",
            file=sys.stderr,
        )

        raise
