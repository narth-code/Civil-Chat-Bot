"""Routing audit routes: windowed history with metrics, run detail, and human evaluations."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import audit as audit_service
from app.schemas import Evaluation, EvaluationCreate, RoutingDetail, RoutingList, RoutingRunTrace
from app.services.audit import AuditService

router = APIRouter(prefix="/api/routing-runs", tags=["routing"])


@router.get("", response_model=RoutingList)
async def list_runs(
    days: int = Query(default=7),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    audit: AuditService = Depends(audit_service),
) -> RoutingList:
    if days not in (7, 30, 90):
        raise HTTPException(status_code=422, detail="days must be 7, 30, or 90")
    return await audit.routing_list(days, offset, limit)


@router.get("/{run_id}/trace", response_model=RoutingRunTrace)
async def get_trace(run_id: UUID, audit: AuditService = Depends(audit_service)) -> RoutingRunTrace:
    return await audit.routing_trace(run_id)


@router.get("/{run_id}", response_model=RoutingDetail)
async def get_run(run_id: UUID, audit: AuditService = Depends(audit_service)) -> RoutingDetail:
    return await audit.routing_detail(run_id)


@router.post("/{run_id}/evaluations", response_model=Evaluation)
async def create_evaluation(
    run_id: UUID, payload: EvaluationCreate, audit: AuditService = Depends(audit_service)
) -> Evaluation:
    return await audit.add_evaluation(run_id, payload)
