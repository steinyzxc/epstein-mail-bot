import asyncio
import base64
import json
import logging
import os
import random
from datetime import datetime, timedelta
from uuid import uuid4
import urllib.request

from telegram import (
    Bot,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InlineQueryResultCachedPhoto,
    InputTextMessageContent,
    LinkPreviewOptions,
)
from telegram.constants import ParseMode

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

MORE_RANDOM_BUTTONS = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("Еще файл", callback_data="more_random"),
        InlineKeyboardButton("Еще фото", callback_data="more_random_photo"),
    ],
])


def _escape_markdown(s: str) -> str:
    """Escape MarkdownV1 special chars so user content doesn't break parse_mode=Markdown."""
    for c in ("\\", "_", "*", "`", "[", "]"):
        s = s.replace(c, "\\" + c)
    return s


def _caption_from_user(from_user: dict | None) -> str:
    """Суффикс к caption при запросе по кнопке: ' from @username' или ' from FirstName'."""
    if not from_user:
        return ""
    username = from_user.get("username")
    if username:
        return " from @" + _escape_markdown(username)
    first = (from_user.get("first_name") or "").strip()
    if first:
        return " from " + _escape_markdown(first)
    return " from id:" + _escape_markdown(str(from_user.get("id", "")))


def get_random_epstein_doc_url(dataset: int | None = None, inline: bool = False) -> tuple[str, str]:
    """Return (url, file_id) for a random Epstein document."""
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
        logger.error("Failed to download PDF: %s", e)
        return None


def pdf_first_page_to_png(pdf_bytes: bytes) -> tuple[bytes, int] | None:
    """Convert first page of PDF to JPG image. Returns (jpeg_bytes, page_count) or None."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = len(doc)
        page = doc[0]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("jpeg", jpg_quality=75)
        doc.close()
        return png_bytes, pages
    except Exception as e:
        logger.error("Failed to convert PDF to JPG: %s", e)
        return None


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


# ---------------------------------------------------------------------------
# Pool helpers (consume from pre-filled queue, signal refill)
# ---------------------------------------------------------------------------

def _pool_receive(pool_name: str) -> dict | None:
    """Try to receive one ready preview from the pool queue."""
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
            WaitTimeSeconds=0,
        )
        messages = resp.get("Messages", [])
        if not messages:
            return None

        msg = messages[0]
        body = json.loads(msg["Body"])

        _sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=msg["ReceiptHandle"],
        )
        return body
    except Exception as e:
        logger.error("_pool_receive(%s) failed: %s", pool_name, e)
        return None


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
# /random and /random_photo handler (async, uses Bot)
# ---------------------------------------------------------------------------

async def _handle_via_pool(
    bot: Bot, chat_id: int, pool_name: str, from_user: dict | None = None
) -> bool:
    """Try to serve the request from the pre-filled pool."""
    entry = _pool_receive(pool_name)
    if entry is None or "tg_file_id" not in entry:
        return False

    tg_file_id = entry["tg_file_id"]
    file_id = entry["file_id"]
    original_url = entry["original_url"]
    pages = entry.get("pages")
    caption = f"[{file_id}]({original_url})"
    if pages is not None:
        caption += f" ({pages} p.)"
    caption += _caption_from_user(from_user)

    try:
        await bot.send_photo(
            chat_id=chat_id,
            photo=tg_file_id,
            caption=caption,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=MORE_RANDOM_BUTTONS,
        )
        return True
    except Exception as e:
        logger.error("sendPhoto by file_id failed: %s", e)

    await bot.send_message(
        chat_id=chat_id,
        text=caption,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=MORE_RANDOM_BUTTONS,
    )
    return True


async def _handle_legacy(
    bot: Bot,
    chat_id: int,
    dataset: int | None,
    max_retries: int = 7,
    from_user: dict | None = None,
):
    """Original on-the-fly download→convert→send path."""
    last_error = None
    last_doc_id = None
    last_doc_url = None

    for attempt in range(max_retries):
        doc_url, doc_id = get_random_epstein_doc_url(dataset=dataset)
        last_doc_id = doc_id
        last_doc_url = doc_url

        pdf_bytes = download_pdf(doc_url)
        if not pdf_bytes:
            last_error = "download"
            logger.warning("Retry %d/%d: failed to download %s", attempt + 1, max_retries, doc_id)
            continue

        result = pdf_first_page_to_png(pdf_bytes)
        if not result:
            last_error = "convert"
            logger.warning("Retry %d/%d: failed to convert %s", attempt + 1, max_retries, doc_id)
            continue
        png_bytes, pages = result

        caption = f"[{doc_id}]({doc_url})"
        if pages is not None:
            caption += f" ({pages} p.)"
        caption += _caption_from_user(from_user)

        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=InputFile(png_bytes, filename="page.png"),
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=MORE_RANDOM_BUTTONS,
            )
            return
        except Exception:
            pass
        await bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=MORE_RANDOM_BUTTONS,
        )
        return

    caption = f"[{last_doc_id}]({last_doc_url})" if last_doc_url else "Файл не найден"
    caption += _caption_from_user(from_user)
    error_msg = "загрузить" if last_error == "download" else "конвертировать"
    await bot.send_message(
        chat_id=chat_id,
        text=f"Не удалось {error_msg} PDF после {max_retries} попыток\n\n{caption}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=MORE_RANDOM_BUTTONS,
    )


async def handle_random_command_async(
    bot: Bot,
    chat_id: int,
    dataset: int | None = None,
    max_retries: int = 7,
    from_user: dict | None = None,
):
    """Handle /random command - send random document as image."""
    pool_name = "random_photo" if dataset == 2 else "random"

    served = await _handle_via_pool(bot, chat_id, pool_name, from_user)
    _pool_request_refill(pool_name)

    if served:
        return

    logger.info("pool '%s' empty — falling back to legacy path", pool_name)
    await _handle_legacy(bot, chat_id, dataset, max_retries, from_user)


async def process_update(update: dict) -> None:
    """Process a single webhook update using PTB Bot."""
    bot = Bot(BOT_TOKEN)

    if "message" in update:
        msg = update["message"]
        text = (msg.get("text") or "").strip()
        chat_id = msg["chat"]["id"]

        if text.startswith("/random_photo"):
            await handle_random_command_async(bot, chat_id, dataset=2)
            return
        if text.startswith("/random"):
            await handle_random_command_async(bot, chat_id)
            return
        return

    if "callback_query" in update:
        cq = update["callback_query"]
        data = cq.get("data")
        msg = cq.get("message")
        if data in ("more_random", "more_random_photo") and msg is not None:
            try:
                await bot.answer_callback_query(callback_query_id=cq["id"])
            except Exception:
                pass
            chat_id = msg["chat"]["id"]
            from_user = cq.get("from")
            if data == "more_random":
                await handle_random_command_async(bot, chat_id, from_user=from_user)
            else:
                await handle_random_command_async(bot, chat_id, dataset=2, from_user=from_user)
        return

    if "inline_query" not in update:
        return

    iq = update["inline_query"]
    q = (iq.get("query") or "").strip()
    results = []

    if q:
        formatted_message = format_epstein_message(q)
        results.append(
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="📧 Отправить как Epstein",
                description=(q[:50] + "...") if len(q) > 50 else q,
                input_message_content=InputTextMessageContent(message_text=formatted_message),
            )
        )

    entry = _pool_receive("random")
    if entry and entry.get("tg_file_id"):
        _pool_request_refill("random")
        inline_caption = f"[{entry['file_id']}]({entry['original_url']})"
        inline_pages = entry.get("pages")
        if inline_pages is not None:
            inline_caption += f" ({inline_pages} p.)"
        results.append(
            InlineQueryResultCachedPhoto(
                id=str(uuid4()),
                photo_file_id=entry["tg_file_id"],
                title="📄 Рандомный файл Epstein",
                description=entry["file_id"],
                caption=inline_caption,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Ещё файл", switch_inline_query_current_chat=""),
                ]]),
            )
        )
    else:
        doc_url, doc_id = get_random_epstein_doc_url(inline=True)
        results.append(
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="📄 Рандомный файл Epstein",
                description=doc_id,
                input_message_content=InputTextMessageContent(
                    message_text=f"[{doc_id}]({doc_url})",
                    parse_mode=ParseMode.MARKDOWN,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                ),
            )
        )

    if results:
        try:
            await bot.answer_inline_query(
                inline_query_id=iq["id"],
                results=results,
                cache_time=0,
                is_personal=True,
            )
        except Exception as e:
            logger.exception("answerInlineQuery failed: %s", e)


def handler(event, context):
    """Yandex Cloud Function entry: webhook POST body -> process update."""
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if SECRET_TOKEN:
        if headers.get("x-telegram-bot-api-secret-token") != SECRET_TOKEN:
            return {"statusCode": 401, "body": "unauthorized"}

    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    update = json.loads(body) if body else {}

    try:
        asyncio.run(process_update(update))
    except Exception as e:
        logger.exception("process_update failed: %s", e)

    return {"statusCode": 200, "body": "ok"}
