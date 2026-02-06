# epstein-mail-bot

Telegram bot that serves random documents from the Jeffrey Epstein public court filings (justice.gov). Deployed as a Yandex Cloud Function behind a Telegram webhook.

## Features

- `/random` — sends a random document page as a photo (any dataset)
- `/random_photo` — sends a random document from dataset 2 specifically
- **Inline mode** — type the bot's name in any chat:
  - Empty query: returns a random document photo
  - Text query: formats the text as a fake Epstein email with censored names

## Architecture

```
Telegram webhook
    → Yandex Cloud Function (this bot)
        → tries SQS pool (pre-rendered photos, instant)
        → falls back to on-the-fly: download PDF → render page 1 → send photo
        → signals refill queue after each pool consume
```

The bot reads from two SQS queues (`random`, `random_photo`) that are pre-filled by the companion [epstein-file-pool-filler](../epstein-file-pool-filler). If the pool is empty or disabled, it falls back to downloading and rendering the PDF on the fly.

## Queue message format

```json
{
  "tg_file_id": "AgACAgIAA...",
  "original_url": "https://www.justice.gov/[redacted]/files/DataSet%202/EFTA00003200.pdf",
  "file_id": "EFTA00003200",
  "dataset": 2,
  "pages": 42
}
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `BOT_TOKEN` | yes | Telegram bot token |
| `TG_SECRET_TOKEN` | no | Webhook secret for request validation |
| `AWS_ACCESS_KEY_ID` | if pool enabled | YMQ credentials |
| `AWS_SECRET_ACCESS_KEY` | if pool enabled | YMQ credentials |
| `YMQ_ENDPOINT` | no | Defaults to `https://message-queue.api.cloud.yandex.net` |
| `RANDOM_POOL_QUEUE_URL` | no | SQS queue URL for random doc pool (enables pool mode) |
| `RANDOM_PHOTO_POOL_QUEUE_URL` | if pool enabled | SQS queue URL for random photo pool |
| `REFILL_QUEUE_URL` | if pool enabled | SQS queue URL for refill signals |

## Dataset file

`ids.txt` — compact range format mapping dataset numbers to document IDs:

```
2:3159-3857
5:8409-8457
```

Each line: `dataset_id:range1,range2,...` where range is `start-end` or a single ID.

## Deployment

Automatic via GitHub Actions on push to `main`. Deploys to Yandex Cloud Functions (`epstein-bot`, python314, 256MB, 7s timeout). Secrets pulled from Yandex Lockbox.

### GitHub Secrets required

- `YC_SA_KEY_JSON` — service account key JSON
- `YC_FOLDER_ID` — Yandex Cloud folder ID
- `LOCKBOX_SECRET_ID` — Lockbox secret containing all env vars

## Dependencies

- `PyMuPDF` — PDF rendering
- `boto3` — SQS (Yandex Message Queue)
