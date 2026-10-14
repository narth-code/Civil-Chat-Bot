"""Consumer regressions against the ask seam and isolated migrated PostgreSQL test database."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.db.session import create_resources
from app.main import create_app

pytestmark = pytest.mark.asyncio
DECISION = {"route": "retrieval", "reason_code": "code_lookup", "rationale": "Civil reference request."}
CITATION = {
    "type": "provision",
    "document_id": "environmental",
    "identifier": "2.3",
    "title": None,
    "pages": "19-20",
    "quote": "Exact source text",
    "url": None,
}


class Agent:
    def __init__(self, responses=None, pause=False):
        self.responses = list(responses or [])
        self.calls = []
        self.scopes = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not pause:
            self.release.set()

    async def __call__(self, question, history, *, scope=None):
        self.calls.append((question, history))
        self.started.set()
        await self.release.wait()
        self.scopes.append(scope)
        response = self.responses.pop(0) if self.responses else {}
        if isinstance(response, Exception):
            raise response
        return {
            "answer": "A cited response.",
            "source_used": "kb",
            "citations": [],
            "clarification": None,
            "review": None,
            "trace": ["routed"],
            "retry_count": 0,
            "city": "Austin",
            "decision": DECISION,
            **response,
        }


@asynccontextmanager
async def client_for(agent):
    settings = Settings(postgres_db="civil_chatbot_test", ollama_url="http://127.0.0.1:1")
    assert settings.postgres_db == "civil_chatbot_test"
    app = create_app(settings, ask=agent)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            client.test_conversations = set()
            try:
                yield client
            finally:
                engine, _ = create_resources(settings.database_url)
                async with engine.begin() as db:
                    for cid in client.test_conversations:
                        params = {"cid": cid}
                        await db.execute(
                            text(
                                "DELETE FROM routing_evaluations WHERE run_id IN (SELECT id FROM routing_runs WHERE conversation_id=:cid)"
                            ),
                            params,
                        )
                        await db.execute(
                            text(
                                "DELETE FROM run_trace_events WHERE run_id IN (SELECT id FROM routing_runs WHERE conversation_id=:cid)"
                            ),
                            params,
                        )
                        await db.execute(
                            text("DELETE FROM routing_run_context_messages WHERE conversation_id=:cid"), params
                        )
                        await db.execute(
                            text(
                                "DELETE FROM routing_run_sources WHERE run_id IN (SELECT id FROM routing_runs WHERE conversation_id=:cid)"
                            ),
                            params,
                        )
                        await db.execute(
                            text(
                                "DELETE FROM messages WHERE conversation_id=:cid AND origin IN ('router_status', 'assistant_answer')"
                            ),
                            params,
                        )
                        await db.execute(text("DELETE FROM routing_runs WHERE conversation_id=:cid"), params)
                        await db.execute(text("DELETE FROM messages WHERE conversation_id=:cid"), params)
                        await db.execute(text("DELETE FROM conversations WHERE id=:cid"), params)
                await engine.dispose()


async def conversation(client):
    response = await client.post("/api/conversations", json={})
    assert response.status_code in (200, 201), response.text
    cid = response.json()["id"]
    client.test_conversations.add(cid)
    return cid


def question(content, message_id=None, scope=None):
    return {"message_id": str(message_id or uuid4()), "content": content, "selected_manual_ids": scope or []}


async def send(client, cid, payload):
    return await client.post(f"/api/conversations/{cid}/messages", json=payload)


async def test_answer_audit_citations_review_trace_city_and_outcome_counts():
    agent = Agent(
        [
            {
                "citations": [CITATION],
                "review": {"valid": True, "confidence": 0.8, "reasoning": "Supported"},
                "trace": ["route", "retrieve", "review"],
            }
        ]
    )
    async with client_for(agent) as client:
        cid = await conversation(client)
        response = await send(client, cid, question("Which design criteria apply?", scope=["environmental"]))
        assert response.status_code == 200, response.text
        data = response.json()
        run = data["routing_run"]
        assert run["status"] == "classified"
        assert run["outcome"] == "kb_answer"
        assert run["citations"] == [CITATION]
        assert run["review"]["confidence"] == 0.8
        assert run["trace"] == ["route", "retrieve", "review"]
        assert run["city"] == "Austin"
        assert run["model_options"]["temperature"] == 0
        assert run["model_options"]["reviewer_model"]
        assert run["policy_version"].startswith("agent-")
        assert [item["origin"] for item in data["messages"]] == ["user", "assistant_answer"]
        assert data["messages"][1]["content"] == "A cited response."
        detail = (await client.get(f"/api/routing-runs/{run['id']}")).json()
        assert detail["routing_run"] == run
        assert detail["policy"]["system_prompt"].startswith('"""Agentic RAG graph')
        assert "EvidenceGrade" in detail["policy"]["response_schema"]["schemas"]
        metrics = (await client.get("/api/routing-runs?days=7")).json()["metrics"]
        assert metrics["outcome_counts"]["kb_answer"] >= 1
        assert set(metrics["outcome_counts"]) == {
            "kb_answer",
            "web_answer",
            "clarification",
            "blocked",
            "insufficient",
            "direct",
            "error",
        }


async def test_selected_scope_reaches_agent_and_empty_scope_searches_all_manuals():
    agent = Agent()
    async with client_for(agent) as client:
        cid = await conversation(client)
        selected = question("What is the drainage provision?", scope=["drainage"])
        first = await send(client, cid, selected)
        assert first.status_code == 200, first.text
        assert first.json()["messages"][0]["selected_manual_ids"] == ["drainage"]
        assert first.json()["routing_run"]["selected_manual_ids"] == ["drainage"]
        unrestricted = question("Which manuals discuss drainage?")
        second = await send(client, cid, unrestricted)
        assert second.status_code == 200, second.text
        assert second.json()["routing_run"]["selected_manual_ids"] == []
        assert agent.scopes == [["drainage"], None]


async def test_agent_error_result_stays_failed_not_blocked_and_replays_original():
    agent = Agent(
        [
            {
                "source_used": "error",
                "answer": "The assistant could not process this message.",
                "decision": None,
                "citations": [],
                "trace": ["router failed"],
            }
        ]
    )
    async with client_for(agent) as client:
        cid = await conversation(client)
        payload = question("Drainage criteria?")
        response = await send(client, cid, payload)
        assert response.status_code == 502, response.text
        result = response.json()
        assert result["routing_run"]["status"] == "failed"
        assert result["routing_run"]["decision"] is None
        assert result["routing_run"]["outcome"] == "error"
        assert result["messages"][1]["origin"] == "assistant_answer"
        assert result["messages"][1]["content"] == "The assistant could not process this message."
        replay = await send(client, cid, payload)
        assert replay.status_code == 502
        assert replay.json()["run_id"] == result["run_id"]
        assert len(agent.calls) == 1


async def test_two_conversations_in_flight_keep_histories_separate_and_busy_is_rejected():
    agent = Agent(pause=True)
    async with client_for(agent) as client:
        a, b = await conversation(client), await conversation(client)
        qa, qb = question("Austin rainfall?"), question("Fire sprinkler spacing?")
        first = asyncio.create_task(send(client, a, qa))
        await asyncio.wait_for(agent.started.wait(), 10)
        assert (await send(client, a, qa)).status_code == 409
        assert (await send(client, a, question("second pending"))).status_code == 409
        second = asyncio.create_task(send(client, b, qb))
        for _ in range(100):
            if len(agent.calls) == 2:
                break
            await asyncio.sleep(0.01)
        agent.release.set()
        ra, rb = await asyncio.gather(first, second)
        assert ra.status_code == rb.status_code == 200
        assert sorted(agent.calls) == sorted([(qa["content"], []), (qb["content"], [])])
        assert ra.json()["routing_run"]["context_message_ids"] == [qa["message_id"]]
        assert rb.json()["routing_run"]["context_message_ids"] == [qb["message_id"]]


async def test_completed_replay_does_not_call_agent_and_conflicting_identity_fails():
    agent = Agent()
    async with client_for(agent) as client:
        cid = await conversation(client)
        payload = question("Austin rainfall?", scope=["environmental"])
        first = await send(client, cid, payload)
        assert first.status_code == 200, first.text
        assert (await send(client, cid, payload)).json() == first.json()
        assert len(agent.calls) == 1
        for altered in ({**payload, "content": "different"}, {**payload, "selected_manual_ids": []}):
            assert (await send(client, cid, altered)).status_code == 409
        other = await conversation(client)
        assert (await send(client, other, payload)).status_code == 409


async def test_clarification_answer_is_passed_as_assistant_history_on_followup():
    agent = Agent(
        [
            {"source_used": "clarification", "answer": "Which city?", "clarification": "Which city?"},
            {"source_used": "direct", "answer": "Thanks, Austin."},
        ]
    )
    async with client_for(agent) as client:
        cid = await conversation(client)
        first = question("What applies to my project?")
        a = await send(client, cid, first)
        assert a.status_code == 200, a.text
        second = question("Austin")
        b = await send(client, cid, second)
        assert b.status_code == 200, b.text
        assert agent.calls[1] == (
            "Austin",
            [{"role": "user", "content": first["content"]}, {"role": "assistant", "content": "Which city?"}],
        )
        assert b.json()["routing_run"]["context_message_ids"] == [
            first["message_id"],
            a.json()["messages"][1]["id"],
            second["message_id"],
        ]


async def test_overbudget_history_is_preserved_without_agent_call():
    agent = Agent()
    async with client_for(agent) as client:
        cid = await conversation(client)
        payload = question("Austin " + "rainfall context " * 1300)
        response = await send(client, cid, payload)
        assert response.status_code == 413, response.text
        data = response.json()
        assert data["code"] == "context_limit"
        assert data["routing_run"]["status"] == "failed"
        assert data["routing_run"]["model_calls"] == 0
        assert data["routing_run"]["context_message_ids"] == [payload["message_id"]]
        assert agent.calls == []
        detail = (await client.get(f"/api/routing-runs/{data['run_id']}")).json()
        assert [item["content"] for item in detail["context_messages"]] == [payload["content"]]


async def test_prior_history_is_not_truncated_when_followup_exceeds_limit():
    agent = Agent()
    async with client_for(agent) as client:
        cid = await conversation(client)
        first = question("Austin drainage context")
        saved = await send(client, cid, first)
        assert saved.status_code == 200
        large = question("rainfall " * 1500)
        response = await send(client, cid, large)
        assert response.status_code == 413
        run = response.json()["routing_run"]
        assert run["context_message_ids"] == [
            first["message_id"],
            saved.json()["messages"][1]["id"],
            large["message_id"],
        ]
        assert len(agent.calls) == 1
        detail = (await client.get(f"/api/routing-runs/{run['id']}")).json()
        assert [item["content"] for item in detail["context_messages"]] == [
            first["content"],
            "A cited response.",
            large["content"],
        ]


@pytest.mark.parametrize(
    "source_used,outcome,decision",
    [
        ("web", "web_answer", DECISION),
        ("insufficient_evidence", "insufficient", DECISION),
        ("blocked", "blocked", {"route": "invalid", "reason_code": "off_topic", "rationale": "Outside scope."}),
        ("direct", "direct", {"route": "conversation", "reason_code": "greeting", "rationale": "Greeting."}),
    ],
)
async def test_all_terminal_agent_outcomes_are_classified(source_used, outcome, decision):
    agent = Agent([{"source_used": source_used, "decision": decision}])
    async with client_for(agent) as client:
        cid = await conversation(client)
        response = await send(client, cid, question("Research question"))
        assert response.status_code == 200, response.text
        assert response.json()["routing_run"]["outcome"] == outcome
        assert response.json()["routing_run"]["status"] == "classified"


@pytest.mark.parametrize(
    "exception,code,status",
    [
        (ConnectionError("secret"), "provider_unavailable", 503),
        (TimeoutError("secret"), "provider_timeout", 504),
        (ValueError("secret"), "provider_invalid_output", 502),
    ],
)
async def test_agent_exception_becomes_typed_persisted_failure(exception, code, status):
    agent = Agent([exception])
    async with client_for(agent) as client:
        cid = await conversation(client)
        response = await send(client, cid, question("Drainage criteria?"))
        assert response.status_code == status, response.text
        data = response.json()
        assert data["code"] == code
        assert "secret" not in response.text
        assert data["routing_run"]["status"] == "failed"
        assert data["messages"][1]["origin"] == "assistant_answer"
        assert data["messages"][1]["content"].startswith("FAILED: The assistant could not complete")


async def test_latest_human_label_changes_agreement_not_saved_decision():
    agent = Agent()
    async with client_for(agent) as client:
        cid = await conversation(client)
        sent = await send(client, cid, question("Drainage criteria?"))
        assert sent.status_code == 200, sent.text
        run = sent.json()["routing_run"]
        before = (await client.get("/api/routing-runs?days=7")).json()["metrics"]["evaluation"]
        first = await client.post(
            f"/api/routing-runs/{run['id']}/evaluations", json={"expected_route": "conversation", "comment": "Initial"}
        )
        second = await client.post(
            f"/api/routing-runs/{run['id']}/evaluations", json={"expected_route": "retrieval", "comment": "Corrected"}
        )
        assert first.status_code in (200, 201) and second.status_code in (200, 201)
        detail = (await client.get(f"/api/routing-runs/{run['id']}")).json()
        assert detail["routing_run"]["decision"] == DECISION
        assert detail["routing_run"]["latest_evaluation"]["id"] == second.json()["id"]
        after = (await client.get("/api/routing-runs?days=7")).json()["metrics"]["evaluation"]
        assert after["reviewed_count"] == before["reviewed_count"] + 1
        assert after["agreement_count"] == before["agreement_count"] + 1


async def test_import_replay_is_atomic_and_legacy_assistant_excluded_from_history():
    agent = Agent()
    async with client_for(agent) as client:
        now = datetime.now(timezone.utc).isoformat()
        cid = str(uuid4())
        client.test_conversations.add(cid)
        m1, m2 = str(uuid4()), str(uuid4())
        imported = {
            "conversations": [
                {
                    "id": cid,
                    "title": "Historical",
                    "created_at": now,
                    "messages": [
                        {
                            "id": m1,
                            "role": "user",
                            "content": "Original  quote\n",
                            "created_at": now,
                            "selected_manual_ids": ["environmental"],
                        },
                        {
                            "id": m2,
                            "role": "assistant",
                            "content": "Unverified demo",
                            "created_at": now,
                            "selected_manual_ids": [],
                        },
                    ],
                }
            ]
        }
        assert (await client.post("/api/conversations/import", json=imported)).status_code == 200
        assert (await client.post("/api/conversations/import", json=imported)).status_code == 200
        detail = (await client.get(f"/api/conversations/{cid}")).json()
        assert [item["origin"] for item in detail["messages"]] == ["user", "legacy_demo"]
        conflict = json.loads(json.dumps(imported))
        conflict["conversations"][0]["messages"].reverse()
        assert (await client.post("/api/conversations/import", json=conflict)).status_code == 409
        assert (await client.get(f"/api/conversations/{cid}")).json() == detail
        assert (await send(client, cid, question("And in Austin?"))).status_code == 200
        assert agent.calls[0][1] == [{"role": "user", "content": "Original  quote\n"}]


async def test_trace_events_are_persisted_in_order_and_exposed():
    events = [
        {
            "node": "route_question",
            "kind": "llm_call",
            "started_at": "2026-10-09T12:00:00+00:00",
            "duration_ms": 4,
            "summary": "route model call",
            "data": {"purpose": "route"},
        },
        {
            "node": "retrieve_kb",
            "kind": "retrieval",
            "started_at": "2026-10-09T12:00:01+00:00",
            "duration_ms": 7,
            "summary": "Retrieve KB -> 1 chunks",
            "data": {"chunks": [{"chunk_id": "chunk-1", "rrf_score": 0.02}]},
        },
    ]
    async with client_for(Agent([{"events": events}])) as client:
        cid = await conversation(client)
        sent = await send(client, cid, question("Trace this request"))
        assert sent.status_code == 200, sent.text
        run_id = sent.json()["routing_run"]["id"]
        trace = await client.get(f"/api/routing-runs/{run_id}/trace")
        assert trace.status_code == 200, trace.text
        payload = trace.json()
        assert payload["run_id"] == run_id
        assert [(event["seq"], event["kind"]) for event in payload["events"]] == [(0, "llm_call"), (1, "retrieval")]
        assert payload["events"][1]["data"]["chunks"][0]["rrf_score"] == 0.02


async def test_traced_failing_node_attaches_event_to_exception():
    from app.rag.trace import traced

    def boom(state):
        raise RuntimeError("broken node")

    with pytest.raises(RuntimeError) as raised:
        await traced("boom", boom)({"events": []})
    assert raised.value.events[0]["node"] == "boom"
    assert raised.value.events[0]["kind"] == "node"
