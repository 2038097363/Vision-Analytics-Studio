"""Small account service for Vision Analytics Studio.

The desktop app never sends device paths or image bytes to these auth endpoints.
Production uses PostgreSQL; SQLite is supported only for local tests and UI work.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Generator

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import DateTime, ForeignKey, LargeBinary, String, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("VAS_DATABASE_URL", "sqlite:///./server/vision-analytics-dev.db")
CREATE_SCHEMA = os.getenv("VAS_CREATE_SCHEMA", "false").lower() == "true" or DATABASE_URL.startswith("sqlite")
DEV_AUTO_VERIFY = os.getenv("VAS_DEV_AUTO_VERIFY", "false").lower() == "true"
ACCESS_TTL = timedelta(minutes=15)
REFRESH_TTL = timedelta(days=30)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1, hash_len=32, salt_len=16)
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash("not-a-real-password-value")
HASH_SEMAPHORE = threading.BoundedSemaphore(2)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email_normalized: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(80))
    password_hash: Mapped[str] = mapped_column(String(255))
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    platform: Mapped[str] = mapped_column(String(40), default="windows")
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class RefreshSession(Base):
    __tablename__ = "refresh_sessions"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    family_id: Mapped[uuid.UUID]
    token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True, index=True)
    access_token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True, index=True)
    access_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, seconds: int) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - seconds:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True


limiter = SlidingWindowLimiter()


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=15, max_length=128)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("invalid email")
        return normalized

    @field_validator("display_name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("display name is required")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)
    device_name: str = Field(min_length=1, max_length=120)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh_token: str = Field(min_length=32, max_length=1024)


class SessionResponse(BaseModel):
    access_token: str
    access_expires_in: int = 900
    refresh_token: str
    refresh_expires_in: int = 2_592_000
    device_id: uuid.UUID
    user_id: uuid.UUID
    display_name: str


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str


def database() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


Db = Annotated[Session, Depends(database)]


def token_hash(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def expired(value: datetime, now: datetime) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= now


def safe_verify(encoded: str, password: str) -> bool:
    with HASH_SEMAPHORE:
        try:
            return PASSWORD_HASHER.verify(encoded, password)
        except (VerifyMismatchError, InvalidHash):
            return False


def client_key(request: Request, email: str) -> str:
    peer = request.client.host if request.client else "unknown"
    return f"{peer}:{email.strip().casefold()}"


def issue_session(db: Session, user: User, device: Device, family_id: uuid.UUID | None = None) -> SessionResponse:
    now = datetime.now(timezone.utc)
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(48)
    session = RefreshSession(
        user_id=user.id,
        device_id=device.id,
        family_id=family_id or uuid.uuid4(),
        token_hash=token_hash(refresh_token),
        access_token_hash=token_hash(access_token),
        access_expires_at=now + ACCESS_TTL,
        expires_at=now + REFRESH_TTL,
    )
    db.add(session)
    device.last_seen_at = now
    db.commit()
    return SessionResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        device_id=device.id,
        user_id=user.id,
        display_name=user.display_name,
    )


def bearer_token(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="authentication_required")
    return authorization[7:]


def authenticated_session(token: Annotated[str, Depends(bearer_token)], db: Db) -> tuple[RefreshSession, User]:
    now = datetime.now(timezone.utc)
    session = db.scalar(select(RefreshSession).where(RefreshSession.access_token_hash == token_hash(token)))
    if session is None or session.revoked_at is not None or expired(session.access_expires_at, now):
        raise HTTPException(status_code=401, detail="authentication_required")
    user = db.get(User, session.user_id)
    if user is None or user.status != "active":
        raise HTTPException(status_code=401, detail="authentication_required")
    return session, user


@asynccontextmanager
async def lifespan(_: FastAPI):
    if CREATE_SCHEMA:
        Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Vision Analytics Studio Cloud API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        value.strip()
        for value in os.getenv(
            "VAS_ALLOWED_ORIGINS",
            "http://localhost:1420,http://127.0.0.1:1420,http://tauri.localhost,https://tauri.localhost",
        ).split(",")
        if value.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.exception_handler(HTTPException)
async def http_problem(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    return JSONResponse(
        status_code=exc.status_code,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": "请求无法完成",
            "status": exc.status_code,
            "code": str(exc.detail),
            "request_id": request_id,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/auth/register", status_code=202)
def register(payload: RegisterRequest, request: Request, db: Db) -> dict[str, str]:
    if not limiter.allow(f"register:{client_key(request, payload.email)}", 4, 3600):
        raise HTTPException(status_code=429, detail="rate_limited")
    existing = db.scalar(select(User).where(User.email_normalized == payload.email))
    with HASH_SEMAPHORE:
        encoded = PASSWORD_HASHER.hash(payload.password)
    if existing is None:
        now = datetime.now(timezone.utc)
        db.add(User(
            email_normalized=payload.email,
            display_name=payload.display_name,
            password_hash=encoded,
            status="active" if DEV_AUTO_VERIFY else "pending",
            email_verified_at=now if DEV_AUTO_VERIFY else None,
        ))
        db.commit()
    return {"status": "accepted"}


@app.post("/v1/auth/login", response_model=SessionResponse)
def login(payload: LoginRequest, request: Request, db: Db) -> SessionResponse:
    email = payload.email.strip().casefold()
    if not limiter.allow(f"login:{client_key(request, email)}", 8, 900):
        raise HTTPException(status_code=429, detail="rate_limited")
    user = db.scalar(select(User).where(User.email_normalized == email))
    valid = safe_verify(user.password_hash if user else DUMMY_PASSWORD_HASH, payload.password)
    if user is None or not valid or user.status != "active":
        raise HTTPException(status_code=401, detail="invalid_credentials")
    device = Device(user_id=user.id, display_name=payload.device_name.strip(), platform="windows")
    db.add(device)
    db.flush()
    return issue_session(db, user, device)


@app.post("/v1/auth/refresh", response_model=SessionResponse)
def refresh(payload: RefreshRequest, db: Db) -> SessionResponse:
    now = datetime.now(timezone.utc)
    current = db.scalar(select(RefreshSession).where(RefreshSession.token_hash == token_hash(payload.refresh_token)))
    if current is None:
        raise HTTPException(status_code=401, detail="invalid_session")
    if current.rotated_at is not None:
        db.execute(update(RefreshSession).where(RefreshSession.family_id == current.family_id).values(revoked_at=now))
        db.commit()
        raise HTTPException(status_code=401, detail="session_reuse_detected")
    if current.revoked_at is not None or expired(current.expires_at, now):
        raise HTTPException(status_code=401, detail="invalid_session")
    user = db.get(User, current.user_id)
    device = db.get(Device, current.device_id)
    if user is None or device is None or user.status != "active" or device.revoked_at is not None:
        raise HTTPException(status_code=401, detail="invalid_session")
    current.rotated_at = now
    db.flush()
    return issue_session(db, user, device, current.family_id)


@app.post("/v1/auth/logout", status_code=204)
def logout(auth: Annotated[tuple[RefreshSession, User], Depends(authenticated_session)], db: Db) -> Response:
    session, _ = auth
    now = datetime.now(timezone.utc)
    db.execute(update(RefreshSession).where(RefreshSession.family_id == session.family_id).values(revoked_at=now))
    db.commit()
    return Response(status_code=204)


@app.get("/v1/me", response_model=UserResponse)
def me(auth: Annotated[tuple[RefreshSession, User], Depends(authenticated_session)]) -> UserResponse:
    _, user = auth
    return UserResponse(id=user.id, email=user.email_normalized, display_name=user.display_name)
