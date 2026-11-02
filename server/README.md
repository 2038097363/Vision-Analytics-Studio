# Vision Analytics Studio cloud service scaffold

This directory contains the first runnable account service plus the PostgreSQL schema for later synchronization work.

- `schema.sql`: PostgreSQL ownership, session, sync metadata, object reference, audit, and deletion tables.
- `app.py`: FastAPI registration, login, opaque access tokens, rotating refresh tokens, replay-family revocation, logout, and profile endpoints.
- `test_auth.py`: security-focused account flow tests using a disposable SQLite database. SQLite is a development/test substitute only.
- `requirements.txt`: pinned small-server dependencies. Production installs the psycopg driver and uses PostgreSQL.
- `../contracts/cloud-api.openapi.yaml`: versioned HTTP API contract.
- Image bytes must use object storage after explicit user opt-in. They must never be inserted into PostgreSQL.
- The desktop SQLite database remains a per-device cache and photo index.

Run locally with development verification enabled:

```powershell
$env:VAS_DATABASE_URL = "sqlite:///./server/vision-analytics-dev.db"
$env:VAS_DEV_AUTO_VERIFY = "true"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

`VAS_DEV_AUTO_VERIFY` must stay false in production. The next server milestone adds verification/reset email tokens, migrations, shared rate limiting, secret injection, audit events, and deployment health checks. Never ship the desktop development bypass in a store build.
