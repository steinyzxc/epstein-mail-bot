import os
import io
import json
import base64
import logging
import random
from datetime import datetime, timedelta
from uuid import uuid4
import urllib.request

BOT_TOKEN = os.environ["BOT_TOKEN"]

# ---------------------------------------------------------------------------
# File pool infrastructure (optional — set env vars to enable)
# ---------------------------------------------------------------------------
_POOL_ENABLED = bool(os.environ.get("RANDOM_POOL_QUEUE_URL"))

if _POOL_ENABLED:
    import boto3

    _AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
    _AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
    _YMQ_ENDPOINT = os.environ.get(
        "YMQ_ENDPOINT", "https://message-queue.api.cloud.yandex.net"
    )
    _S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")
    _S3_BUCKET = os.environ["S3_BUCKET"]
    _RANDOM_POOL_QUEUE_URL = os.environ["RANDOM_POOL_QUEUE_URL"]
    _RANDOM_PHOTO_POOL_QUEUE_URL = os.environ["RANDOM_PHOTO_POOL_QUEUE_URL"]
    _REFILL_QUEUE_URL = os.environ["REFILL_QUEUE_URL"]

    _sqs = boto3.client(
        "sqs",
        endpoint_url=_YMQ_ENDPOINT,
        region_name="ru-central1",
        aws_access_key_id=_AWS_KEY,
        aws_secret_access_key=_AWS_SECRET,
    )
    _s3 = boto3.client(
        "s3",
        endpoint_url=_S3_ENDPOINT,
        region_name="ru-central1",
        aws_access_key_id=_AWS_KEY,
        aws_secret_access_key=_AWS_SECRET,
    )


def load_ids_from_file(filepath: str = None) -> dict[int, tuple]:
    """Load IDS_BY_DATASET from compact range format file.
    
    Format per line: dataset_id:range1,range2,...
    Where range is either 'start-end' or 'single_id'
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "ids.txt")
    
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            dataset_str, ranges_str = line.split(":", 1)
            dataset_id = int(dataset_str)
            
            ids = []
            for part in ranges_str.split(","):
                if "-" in part:
                    start, end = part.split("-", 1)
                    ids.extend(range(int(start), int(end) + 1))
                else:
                    ids.append(int(part))
            
            result[dataset_id] = tuple(ids)
    
    return result


IDS_BY_DATASET = load_ids_from_file()
SECRET_TOKEN = os.environ.get("TG_SECRET_TOKEN")  # optional webhook secret

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# All datasets for message commands
AVAILABLE_DATASETS = list(IDS_BY_DATASET.keys())
# Exclude dataset 9 for inline (too large)
INLINE_DATASETS = [k for k in IDS_BY_DATASET.keys() if k != 9]


def get_random_epstein_doc_url(dataset: int | None = None, inline: bool = False) -> tuple[str, str]:
    """Return (url, file_id) for a random Epstein document.

    Args:
        dataset: Specific dataset number to use. If None, picks random dataset.
        inline: If True, excludes dataset 9 from random selection.
    """
    pool = INLINE_DATASETS if inline else AVAILABLE_DATASETS
    if not pool:
        return ("https://www.justice.gov/epstein", "EFTA00000001")

    if dataset is not None and dataset in IDS_BY_DATASET:
        chosen_dataset = dataset
    else:
        chosen_dataset = random.choice(pool)

    num = random.choice(IDS_BY_DATASET[chosen_dataset])
    file_id = f"EFTA{num:08d}"
    url = f"https://www.justice.gov/epstein/files/DataSet%20{chosen_dataset}/{file_id}.pdf"
    return (url, file_id)


def download_pdf(url: str) -> bytes | None:
    """Download PDF from justice.gov with age verification cookie."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Cookie": "justiceGovAgeVerified=true",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return None


def pdf_first_page_to_png(pdf_bytes: bytes) -> bytes | None:
    """Convert first page of PDF to JPG image."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        # Render at 2x resolution for better quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("jpeg", jpg_quality=75)
        doc.close()
        return png_bytes
    except Exception as e:
        logger.error(f"Failed to convert PDF to JPG: {e}")
        return None


def tg(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Common name length patterns (first_name, last_name)
NAME_PATTERNS = [
    (4, 6), (5, 7), (6, 5), (7, 8), (5, 6),
    (6, 7), (4, 5), (5, 8), (6, 6), (7, 6),
    (8, 7), (6, 8), (5, 5), (7, 7), (4, 7),
]


def generate_censored_name() -> str:
    first_len, last_len = random.choice(NAME_PATTERNS)
    return f"{'█' * first_len} {'█' * last_len}"


def censor_capitalized_words(text: str) -> str:
    result = []
    sentence_start = True
    i = 0

    while i < len(text):
        if text[i] in ".!?\n":
            result.append(text[i])
            sentence_start = True
            i += 1
            continue

        if text[i].isspace():
            result.append(text[i])
            i += 1
            continue

        word_start = i
        while i < len(text) and (not text[i].isspace()) and text[i] not in ".!?\n":
            i += 1
        word = text[word_start:i]

        if word and word[0].isupper() and not sentence_start and len(word) > 1:
            result.append(word[0] + "█" * (len(word) - 1))
        else:
            result.append(word)

        sentence_start = False

    return "".join(result)


def generate_random_date() -> str:
    start_date = datetime(2006, 1, 1)
    end_date = datetime(2010, 12, 31)
    days_between = (end_date - start_date).days
    random_date = start_date + timedelta(days=random.randint(0, days_between))

    random_date = random_date.replace(
        hour=random.randint(6, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
    )

    # Portable formatting (no %-m / %-d dependency)
    weekday = random_date.strftime("%a")
    month = random_date.month
    day = random_date.day
    year = random_date.year
    time_part = random_date.strftime("%I:%M:%S %p")
    return f"{weekday} {month}/{day}/{year} {time_part}"


def format_epstein_message(user_message: str) -> str:
    censored_name = generate_censored_name()
    date = generate_random_date()
    censored_message = censor_capitalized_words(user_message)

    return (
        f"To: {censored_name}\n"
        f"From: Jeffrey Epstein <jeevacation@gmail.com>\n"
        f"{date}\n\n"
        f"{censored_message}\n\n"
        f"Sent from my iPad"
    )


def send_photo_to_chat(chat_id: int, png_bytes: bytes, caption: str) -> bool:
    """Send photo directly to a chat."""
    try:
        boundary = "----WebKitFormBoundary" + uuid4().hex[:16]
        body = io.BytesIO()

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        body.write(f"{chat_id}\r\n".encode())

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
        body.write(f"{caption}\r\n".encode())

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="parse_mode"\r\n\r\n')
        body.write(b"Markdown\r\n")

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="photo"; filename="page.png"\r\n')
        body.write(b"Content-Type: image/png\r\n\r\n")
        body.write(png_bytes)
        body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode())

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        req = urllib.request.Request(
            url,
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("ok", False)
    except Exception as e:
        logger.error(f"Failed to send photo: {e}")
        return False


# ---------------------------------------------------------------------------
# Pool helpers (consume from pre-filled queue, signal refill)
# ---------------------------------------------------------------------------

def _pool_receive(pool_name: str) -> dict | None:
    """Try to receive one ready preview from the pool queue.

    Returns parsed message body dict or None if pool is empty / unavailable.
    On success the message is deleted from the queue (acknowledged).
    """
    if not _POOL_ENABLED:
        return None

    queue_url = (
        _RANDOM_POOL_QUEUE_URL
        if pool_name == "random"
        else _RANDOM_PHOTO_POOL_QUEUE_URL
    )

    try:
        resp = _sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=0,  # no long polling — we need speed
        )
        messages = resp.get("Messages", [])
        if not messages:
            return None

        msg = messages[0]
        body = json.loads(msg["Body"])

        # Acknowledge immediately so no other consumer picks it up on retry
        _sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=msg["ReceiptHandle"],
        )
        return body
    except Exception as e:
        logger.error("_pool_receive(%s) failed: %s", pool_name, e)
        return None


def _pool_download_jpeg(s3_key: str) -> bytes | None:
    """Download a cached JPEG preview from Object Storage."""
    try:
        resp = _s3.get_object(Bucket=_S3_BUCKET, Key=s3_key)
        return resp["Body"].read()
    except Exception as e:
        logger.error("_pool_download_jpeg(%s) failed: %s", s3_key, e)
        return None


def _pool_cleanup_s3(s3_key: str):
    """Delete the preview from S3 after it has been sent."""
    try:
        _s3.delete_object(Bucket=_S3_BUCKET, Key=s3_key)
    except Exception as e:
        logger.warning("_pool_cleanup_s3(%s) failed: %s", s3_key, e)


def _pool_request_refill(pool_name: str):
    """Send a refill signal so the filler tops up the pool."""
    if not _POOL_ENABLED:
        return
    try:
        _sqs.send_message(
            QueueUrl=_REFILL_QUEUE_URL,
            MessageBody=json.dumps({"pool": pool_name}),
        )
    except Exception as e:
        logger.warning("_pool_request_refill(%s) failed: %s", pool_name, e)


# ---------------------------------------------------------------------------
# /random and /random_photo handler
# ---------------------------------------------------------------------------

def _handle_via_pool(chat_id: int, pool_name: str) -> bool:
    """Try to serve the request from the pre-filled pool.

    Returns True if the response was sent (success or link fallback).
    Returns False if pool is empty / unavailable — caller should use legacy path.
    """
    entry = _pool_receive(pool_name)
    if entry is None:
        return False

    s3_key = entry["s3_key"]
    file_id = entry["file_id"]
    original_url = entry["original_url"]
    caption = f"[{file_id}]({original_url})"

    jpeg_bytes = _pool_download_jpeg(s3_key)
    if jpeg_bytes is None:
        # S3 fetch failed — send link as fallback, still counts as handled
        tg("sendMessage", {
            "chat_id": chat_id,
            "text": caption,
            "parse_mode": "Markdown",
        })
        _pool_cleanup_s3(s3_key)
        return True

    if send_photo_to_chat(chat_id, jpeg_bytes, caption):
        _pool_cleanup_s3(s3_key)
        return True

    # Photo send failed — fall back to link
    tg("sendMessage", {
        "chat_id": chat_id,
        "text": caption,
        "parse_mode": "Markdown",
    })
    _pool_cleanup_s3(s3_key)
    return True


def _handle_legacy(chat_id: int, dataset: int | None, max_retries: int = 7):
    """Original on-the-fly download→convert→send path."""
    last_error = None
    last_doc_id = None
    last_doc_url = None

    for attempt in range(max_retries):
        doc_url, doc_id = get_random_epstein_doc_url(dataset=dataset)
        last_doc_id = doc_id
        last_doc_url = doc_url
        caption = f"[{doc_id}]({doc_url})"

        pdf_bytes = download_pdf(doc_url)
        if not pdf_bytes:
            last_error = "download"
            logger.warning(f"Retry {attempt + 1}/{max_retries}: failed to download {doc_id}")
            continue

        png_bytes = pdf_first_page_to_png(pdf_bytes)
        if not png_bytes:
            last_error = "convert"
            logger.warning(f"Retry {attempt + 1}/{max_retries}: failed to convert {doc_id}")
            continue

        if send_photo_to_chat(chat_id, png_bytes, caption):
            return
        else:
            tg("sendMessage", {
                "chat_id": chat_id,
                "text": caption,
                "parse_mode": "Markdown",
            })
            return

    caption = f"[{last_doc_id}]({last_doc_url})" if last_doc_url else "Файл не найден"
    error_msg = "загрузить" if last_error == "download" else "конвертировать"
    tg("sendMessage", {
        "chat_id": chat_id,
        "text": f"Не удалось {error_msg} PDF после {max_retries} попыток\n\n{caption}",
        "parse_mode": "Markdown",
    })


def handle_random_command(chat_id: int, dataset: int | None = None, max_retries: int = 7):
    """Handle /random command - send random document as image.

    Tries pre-filled pool first (fast path). Falls back to on-the-fly
    generation if pool is empty or disabled.

    Args:
        chat_id: Telegram chat ID to send the response to.
        dataset: Specific dataset number to use. If None, picks random dataset.
        max_retries: Maximum number of retry attempts for legacy path.
    """
    pool_name = "random_photo" if dataset == 2 else "random"

    # Fast path: grab a cached preview from the pool
    served = _handle_via_pool(chat_id, pool_name)

    # Signal the filler regardless — either we consumed one (needs replacement)
    # or the pool was empty (needs urgent filling).
    _pool_request_refill(pool_name)

    if served:
        return

    # Slow path: generate on-the-fly (pool was empty or disabled)
    logger.info("pool '%s' empty — falling back to legacy path", pool_name)
    _handle_legacy(chat_id, dataset, max_retries)


def handler(event, context):
    # Optional Telegram webhook secret check
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if SECRET_TOKEN:
        if headers.get("x-telegram-bot-api-secret-token") != SECRET_TOKEN:
            return {"statusCode": 401, "body": "unauthorized"}

    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    update = json.loads(body) if body else {}

    # Handle /random and /random_photo commands - send random document as image
    if "message" in update:
        msg = update["message"]
        text = msg.get("text", "")
        chat_id = msg["chat"]["id"]

        if text.startswith("/random_photo"):
            # /random_photo - specifically from dataset 2
            handle_random_command(chat_id, dataset=2)
            return {"statusCode": 200, "body": "ok"}

        if text.startswith("/random"):
            # /random - from any dataset
            handle_random_command(chat_id)
            return {"statusCode": 200, "body": "ok"}

        # Ignore other messages
        return {"statusCode": 200, "body": "ok"}

    # Handle inline queries
    if "inline_query" not in update:
        return {"statusCode": 200, "body": "ok"}

    iq = update["inline_query"]
    q = (iq.get("query") or "").strip()

    results = []

    # If user typed something, add the Epstein message option
    if q:
        formatted_message = format_epstein_message(q)
        results.append({
            "type": "article",
            "id": str(uuid4()),
            "title": "📧 Отправить как Epstein",
            "description": (q[:50] + "...") if len(q) > 50 else q,
            "input_message_content": {"message_text": formatted_message},
        })

    # Always add random document option (link only for speed)
    doc_url, doc_id = get_random_epstein_doc_url(inline=True)
    results.append({
        "type": "article",
        "id": str(uuid4()),
        "title": "📄 Рандомный файл Epstein",
        "description": doc_id,
        "input_message_content": {
            "message_text": f"[{doc_id}]({doc_url})",
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
    })

    if not results:
        return {"statusCode": 200, "body": "ok"}

    try:
        tg(
            "answerInlineQuery",
            {
                "inline_query_id": iq["id"],
                "results": results,
                "cache_time": 0,
                "is_personal": True,
            },
        )
    except Exception as e:
        logger.exception("answerInlineQuery failed: %s", e)

    return {"statusCode": 200, "body": "ok"}
