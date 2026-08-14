"""FastAPI application factory for local read-only evidence serving."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHttpException

from junctionlens.api.models import (
    ApiErrorDetail,
    ApiErrorResponse,
    ArtifactDetail,
    ArtifactPage,
    DecisionDetail,
    HealthResponse,
    MetricTablePage,
    RunPage,
    ServiceConfig,
)
from junctionlens.api.repository import EvidenceReadError, EvidenceRepository

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}


class ApiProblem(RuntimeError):
    """A stable client-visible API failure."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _problem(status_code: int, code: str, message: str) -> JSONResponse:
    body = ApiErrorResponse(
        schema_version="junctionlens.api-error.v1",
        error=ApiErrorDetail(code=code, message=message),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _read_error(error: EvidenceReadError) -> ApiProblem:
    return ApiProblem(409, "API_REGISTRY_INVALID", str(error))


def create_app(config: ServiceConfig) -> FastAPI:
    """Create one API instance bound to an already registered artifact root."""
    repository = EvidenceRepository(config)
    app = FastAPI(
        title="JunctionLens local evidence API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(_request: Request, error: ApiProblem) -> JSONResponse:
        return _problem(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return _problem(422, "API_REQUEST_INVALID", "request did not satisfy the API contract")

    @app.exception_handler(StarletteHttpException)
    async def http_handler(_request: Request, error: StarletteHttpException) -> JSONResponse:
        if error.status_code == 404:
            return _problem(404, "API_ROUTE_NOT_FOUND", "API route does not exist")
        if error.status_code == 405:
            return _problem(405, "API_METHOD_NOT_ALLOWED", "API route is read-only")
        return _problem(error.status_code, "API_HTTP_ERROR", "request could not be completed")

    @app.exception_handler(Exception)
    async def internal_handler(_request: Request, _error: Exception) -> JSONResponse:
        return _problem(500, "API_INTERNAL_ERROR", "internal evidence service error")

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            artifact_count, run_count = repository.counts()
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return HealthResponse(
            schema_version="junctionlens.api-health.v1",
            state="READY",
            artifact_count=artifact_count,
            run_count=run_count,
        )

    @app.get("/api/v1/runs", response_model=RunPage)
    def runs(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> RunPage:
        try:
            items, page = repository.list_runs(offset, limit)
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return RunPage(schema_version="junctionlens.api-run-page.v1", page=page, items=items)

    @app.get("/api/v1/artifacts", response_model=ArtifactPage)
    def artifacts(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        kind: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    ) -> ArtifactPage:
        try:
            items, page = repository.list_artifacts(offset, limit, kind=kind)
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return ArtifactPage(
            schema_version="junctionlens.api-artifact-page.v1",
            page=page,
            items=items,
        )

    @app.get("/api/v1/artifacts/{manifest_sha256}", response_model=ArtifactDetail)
    def artifact(
        manifest_sha256: Annotated[str, ApiPath(pattern=_SHA256_PATTERN)],
    ) -> ArtifactDetail:
        try:
            return repository.artifact(manifest_sha256)
        except KeyError as error:
            raise ApiProblem(404, "API_ARTIFACT_NOT_FOUND", "artifact is not registered") from error
        except EvidenceReadError as error:
            raise _read_error(error) from error

    @app.get("/api/v1/artifacts/{manifest_sha256}/content")
    def artifact_content(
        manifest_sha256: Annotated[str, ApiPath(pattern=_SHA256_PATTERN)],
    ) -> StreamingResponse:
        try:
            artifact_value = repository.artifact(manifest_sha256)
            payload = repository.open_payload(manifest_sha256)
        except KeyError as error:
            raise ApiProblem(404, "API_ARTIFACT_NOT_FOUND", "artifact is not registered") from error
        except EvidenceReadError as error:
            raise _read_error(error) from error
        headers = {
            "Content-Disposition": f'attachment; filename="{manifest_sha256}"',
            "Content-Length": str(artifact_value.payload_byte_size),
        }
        return StreamingResponse(
            payload.chunks(),
            media_type=artifact_value.media_type,
            headers=headers,
        )

    @app.get("/api/v1/metrics/{manifest_sha256}", response_model=MetricTablePage)
    def metrics(
        manifest_sha256: Annotated[str, ApiPath(pattern=_SHA256_PATTERN)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> MetricTablePage:
        try:
            columns, rows, page = repository.metric_rows(manifest_sha256, offset, limit)
        except KeyError as error:
            raise ApiProblem(404, "API_ARTIFACT_NOT_FOUND", "artifact is not registered") from error
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return MetricTablePage(
            schema_version="junctionlens.api-metric-table.v1",
            manifest_sha256=manifest_sha256,
            columns=columns,
            page=page,
            rows=rows,
        )

    @app.get("/api/v1/decisions", response_model=ArtifactPage)
    def decisions(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ArtifactPage:
        try:
            items, page = repository.list_artifacts(offset, limit, kind="release_decision")
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return ArtifactPage(
            schema_version="junctionlens.api-artifact-page.v1",
            page=page,
            items=items,
        )

    @app.get("/api/v1/decisions/{manifest_sha256}", response_model=DecisionDetail)
    def decision(
        manifest_sha256: Annotated[str, ApiPath(pattern=_SHA256_PATTERN)],
    ) -> DecisionDetail:
        try:
            value = repository.decision(manifest_sha256)
        except KeyError as error:
            raise ApiProblem(404, "API_ARTIFACT_NOT_FOUND", "artifact is not registered") from error
        except EvidenceReadError as error:
            raise _read_error(error) from error
        return DecisionDetail(
            schema_version="junctionlens.api-decision.v1",
            manifest_sha256=manifest_sha256,
            decision=value,
        )

    @app.get("/api/v1/images/{manifest_sha256}")
    def image(
        manifest_sha256: Annotated[str, ApiPath(pattern=_SHA256_PATTERN)],
    ) -> StreamingResponse:
        try:
            artifact_value = repository.artifact(manifest_sha256)
            if artifact_value.media_type not in repository.image_media_types():
                raise ApiProblem(
                    415,
                    "API_IMAGE_TYPE_UNSUPPORTED",
                    "artifact is not a supported raster image",
                )
            payload = repository.open_payload(
                manifest_sha256,
                limit=config.max_image_bytes,
            )
        except KeyError as error:
            raise ApiProblem(404, "API_ARTIFACT_NOT_FOUND", "artifact is not registered") from error
        except EvidenceReadError as error:
            raise _read_error(error) from error
        headers = {
            "Content-Disposition": "inline",
            "Content-Length": str(artifact_value.payload_byte_size),
            "X-JunctionLens-License": artifact_value.license_id,
        }
        return StreamingResponse(
            payload.chunks(),
            media_type=artifact_value.media_type,
            headers=headers,
        )

    return app


__all__ = ["ApiProblem", "create_app"]
