import sys
import os
import io
import boto3
from typing import Iterable, Dict, Any, List, Generator, Tuple
from pypdf import PdfReader

from pdf_to_vector import convert_to_vectors

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), '../credentials')
    )
)

from pymilvus import MilvusClient, Collection

from credentials import ORACLE_S3_ACCESS_KEY, ORACLE_S3_SECRET_KEY, ORACLE_S3_ENDPOINT, ORACLE_REGION, ORACLE_INGEST_BUCKET, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

s3_client = boto3.client(
    "s3",
    aws_access_key_id=ORACLE_S3_ACCESS_KEY,
    aws_secret_access_key=ORACLE_S3_SECRET_KEY,
    endpoint_url=ORACLE_S3_ENDPOINT
)


client = MilvusClient("http://"+MILVUS_HOST+":"+MILVUS_PORT, db_name="default")

if not client.has_collection(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True
    )
    print(f"Collection '{COLLECTION_NAME}' created.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")


def _norm_etag(x: str) -> str:
    return str(x).strip('"').strip("'") if x is not None else x

def _batched(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _query_existing_etags(client: MilvusClient, collection: str, etags: List[str], etag_field="etag", batch_size=500) -> set:
    """Check existence in Milvus in IN-batches; returns a set of found etags."""
    found = set()
    for batch in _batched(etags, batch_size):
        in_list = ",".join([f'"{e}"' for e in batch])
        filt = f'{etag_field} in [{in_list}]'
        rows = client.query(
            collection_name=collection,
            filter=filt,
            output_fields=[etag_field],
            limit=len(batch)
        )
        for r in rows:
            val = r.get(etag_field)
            if val is not None:
                found.add(_norm_etag(val))
    return found

def iter_new_objects_by_etag(
    s3_client,
    bucket: str,
    prefix: str,
    milvus_client: MilvusClient,
    collection_name: str,
    etag_field: str = "etag",
    page_batch_check: int = 750,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream pages from S3/OCI and yield only objects whose ETag is NOT already in Milvus.
    Memory-friendly: handles a single page at a time.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix or "")


    milvus_client.load_collection(collection_name)

    for page in pages:
        contents = page.get("Contents", []) or []
        if not contents:
            continue

        etag_to_objs: Dict[str, List[Dict[str, Any]]] = {}
        for obj in contents:
            etg = _norm_etag(obj.get("ETag"))
            if not etg:
                continue
            etag_to_objs.setdefault(etg, []).append(obj)

        unique_etags = list(etag_to_objs.keys())

        existing = set()
        for sub in _batched(unique_etags, page_batch_check):
            existing |= _query_existing_etags(
                milvus_client, collection_name, sub, etag_field=etag_field, batch_size=page_batch_check
            )

        for etg, objs in etag_to_objs.items():
            if etg not in existing:
                for obj in objs:
                    yield obj



def _is_pdf_key(key: str) -> bool:
    return key.lower().endswith(".pdf")


def read_pdf_from_s3_bytes(s3_client, bucket: str, key: str, max_pages: int = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Download the object into memory (bytes) and extract text.
    Returns: (full_text, pages_list, meta)
    """
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()  # bytes
    reader = PdfReader(io.BytesIO(body))

    try:
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                pass
    except Exception:
        pass

    pages, limit = [], (max_pages or 10**9)
    for i, page in enumerate(reader.pages):
        if i >= limit:
            break
        txt = page.extract_text() or ""
        pages.append(txt)

    full_text = "\n\n".join(pages)
    meta = {
        "n_pages_total": len(reader.pages),
        "n_pages_read": len(pages),
        "content_length": resp.get("ContentLength"),
        "content_type": resp.get("ContentType"),
        "key": key,
        "bucket": bucket,
    }
    return full_text, pages, meta



new_objs_iter = iter_new_objects_by_etag(
    s3_client=s3_client,
    bucket=ORACLE_INGEST_BUCKET,
    prefix="",
    milvus_client=client,
    collection_name=COLLECTION_NAME,
    etag_field="Etag",
    page_batch_check=5
)

for obj in new_objs_iter:
    key  = obj["Key"]
    etag = obj["ETag"].strip('"')
    print(key," : ",etag)


    if not _is_pdf_key(key):
        continue

    try:
        text, pages, meta = read_pdf_from_s3_bytes(s3_client, ORACLE_INGEST_BUCKET, key)
        #print(f"[PDF] {key} | etag={etag} | pages_read={meta['n_pages_read']}/{meta['n_pages_total']}")

        convert_to_vectors(text = text, etag=etag, finalname=key)

    except Exception as e:
        print(f"[skip] {key} ({e})")


