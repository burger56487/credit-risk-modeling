"""Research scoring API: fixed artefact, strict input, bounded body, token.

The service accepts only the predefined numeric application fields. There is no
endpoint for training, uploading artefacts or choosing a model path, and labels
are rejected as unknown fields. Responses deliberately do not echo raw
application values, and every response is marked as a research simulation.

The token check and the body limit are minimum protections, not production
security: there is no rate limiting, transport encryption, token rotation or
audit logging here.
"""
import hmac
import os
import re
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.serving.artifact import load_bundle


class ApplicationInput(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    application_id: str = Field(min_length=1, max_length=64)

    amt_income_total: float | None
    amt_credit: float | None
    amt_annuity: float | None
    days_birth: float | None
    days_employed: float | None
    ext_source_1: float | None
    ext_source_2: float | None
    ext_source_3: float | None
    bureau_cnt: int = Field(ge=0)
    bureau_debt_total: float | None

    @field_validator("application_id")
    @classmethod
    def validate_identifier(cls, value):
        if value != value.strip() or not value:
            raise ValueError("申请标识不能空白或带首尾空格。")
        return value


class BatchInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    applications: list[ApplicationInput] = Field(
        min_length=1, max_length=100
    )

    @model_validator(mode="after")
    def check_identifiers(self):
        if self.request_id != self.request_id.strip():
            raise ValueError("请求标识不能带首尾空格。")

        identifiers = [item.application_id for item in self.applications]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("同一请求中的申请标识不能重复。")

        return self


class ScoreItem(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    application_id: str
    predicted_bad_probability: float = Field(ge=0, le=1)
    score_raw: float
    score_display: float
    unknown_selected_bins: int = Field(ge=0)
    input_review_required: bool
    simulated_approval: bool | None


class BatchOutput(BaseModel):
    request_id: str
    artifact_id: str
    model_version: str
    strategy_version: str
    purpose: Literal["研究仿真，非真实授信决定"]
    results: list[ScoreItem]


class BodyLimitMiddleware:
    """Limit the scoring body size before it is parsed into models."""

    def __init__(self, app, max_bytes: int = 65536):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/score":
            await self.app(scope, receive, send)
            return

        body = bytearray()

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            chunk = message.get("body", b"")

            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse(
                    {"detail": "请求体超过当前允许大小。"}, status_code=413
                )
                await response(scope, receive, send)
                return

            body.extend(chunk)

            if not message.get("more_body", False):
                break

        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    artifact_directory,
    trusted_manifest_sha256: str,
    api_token: str,
) -> FastAPI:
    if (
        not isinstance(api_token, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{32,128}", api_token) is None
    ):
        raise ValueError("接口令牌必须满足当前长度和字符要求。")

    # A failed load stops the application from being created at all; the service
    # never starts in a half-ready state.
    bundle = load_bundle(artifact_directory, trusted_manifest_sha256)

    app = FastAPI(
        title="信用风险研究评分",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=65536)

    bearer = HTTPBearer(auto_error=False)

    def require_token(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(bearer)
        ],
    ):
        supplied = "" if credentials is None else credentials.credentials

        if not hmac.compare_digest(
            supplied.encode("utf-8"), api_token.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="认证失败。",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(request, exception):
        # Default validation errors can quote the offending input; this API does
        # not echo application fields back to the caller.
        return JSONResponse(
            {"detail": "请求不符合字段、类型、样本数量或标识约定。"},
            status_code=422,
        )

    @app.get("/health")
    def health():
        return {"status": "就绪"}

    @app.post(
        "/v1/score",
        response_model=BatchOutput,
        dependencies=[Depends(require_token)],
    )
    def score(request: BatchInput):
        rows = [
            item.model_dump(exclude={"application_id"})
            for item in request.applications
        ]
        frame = pd.DataFrame(rows)

        try:
            result = bundle.predict_frame(frame)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422, detail="输入未通过评分数值校验。"
            ) from None
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="评分执行失败，请联系内部维护人员。",
            ) from None

        items = [
            ScoreItem(
                application_id=application.application_id,
                **row,
            )
            for application, row in zip(
                request.applications, result.to_dict("records")
            )
        ]

        return BatchOutput(
            request_id=request.request_id,
            artifact_id=bundle.release.artifact_id,
            model_version=bundle.release.model_version,
            strategy_version=bundle.release.strategy_version,
            purpose="研究仿真，非真实授信决定",
            results=items,
        )

    return app


def app_from_environment():
    """Service entry point; secrets are read at start-up, never printed."""
    required = (
        "RISK_ARTIFACT_DIR",
        "RISK_MANIFEST_SHA256",
        "RISK_API_TOKEN",
    )

    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("启动所需的工件、摘要或令牌配置不完整。")

    return create_app(
        os.environ["RISK_ARTIFACT_DIR"],
        os.environ["RISK_MANIFEST_SHA256"],
        os.environ["RISK_API_TOKEN"],
    )
