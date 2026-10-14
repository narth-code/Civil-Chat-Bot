import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCw, Save, X } from "lucide-react";
import {
  api,
  type Route,
  type RoutingDetail,
  type RoutingList,
  type RoutingRun,
  type RoutingTraceEvent,
} from "./api";
import { Citations, outcomeKeys, outcomeLabels } from "./Citations";
import { manuals } from "./manuals";

type Range = 7 | 30 | 90;
const pageSize = 25;

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatMillis(value: number | null) {
  return value === null ? "—" : `${Math.round(value)} ms`;
}

function routeLabel(run: RoutingRun) {
  return run.decision
    ? `${run.decision.route} · ${run.decision.reason_code}`
    : run.error_code
      ? `Failed · ${run.error_code}`
      : "Pending";
}

export default function RoutingPanel({ active }: { active: boolean }) {
  const [range, setRange] = useState<Range>(30);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<RoutingList | null>(null);
  const [selected, setSelected] = useState<RoutingDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [expectedRoute, setExpectedRoute] = useState<Route>("retrieval");
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);
  const [trace, setTrace] = useState<RoutingTraceEvent[] | null>(null);
  const [traceUnavailable, setTraceUnavailable] = useState(false);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void api
      .listRoutingRuns(range, offset, pageSize, controller.signal)
      .then((result) => {
        setData(result);
        setSelected((current) =>
          current &&
          !result.items.some((item) => item.id === current.routing_run.id)
            ? null
            : current,
        );
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError")
          return;
        setData(null);
        setSelected(null);
        setError(
          reason instanceof Error
            ? reason.message
            : "Live routing could not be loaded.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [active, range, offset, refreshToken]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    void api
      .getRoutingRun(selected.routing_run.id, controller.signal)
      .then((detail) => {
        setSelected(detail);
        setExpectedRoute(
          detail.routing_run.latest_evaluation?.expected_route ??
            detail.routing_run.decision?.route ??
            "retrieval",
        );
        setComment(detail.routing_run.latest_evaluation?.comment ?? "");
      })
      .catch((reason) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError"))
          setError(
            reason instanceof Error
              ? reason.message
              : "Routing run details could not be loaded.",
          );
      });
    return () => controller.abort();
  }, [selected?.routing_run.id]);
  useEffect(() => {
    if (!selected) {
      setTrace(null);
      setTraceUnavailable(false);
      return;
    }
    const controller = new AbortController();
    setTrace(null);
    setTraceUnavailable(false);
    void api
      .getRoutingTrace(selected.routing_run.id, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setTrace(result.events);
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError")
          return;
        if (!controller.signal.aborted) setTraceUnavailable(true);
      });
    return () => controller.abort();
  }, [selected?.routing_run.id]);

  const openRun = (run: RoutingRun) => {
    setSelected({
      routing_run: run,
      context_messages: [],
      policy: {
        version: run.policy_version,
        system_prompt: "",
        response_schema: {},
      },
      evaluations: [],
    });
  };

  const saveEvaluation = async () => {
    if (!selected || saving) return;
    setSaving(true);
    try {
      const evaluation = await api.createEvaluation(
        selected.routing_run.id,
        expectedRoute,
        comment,
      );
      setSelected((current) =>
        current
          ? {
              ...current,
              routing_run: {
                ...current.routing_run,
                latest_evaluation: evaluation,
              },
              evaluations: [...current.evaluations, evaluation],
            }
          : current,
      );
      setData((current) =>
        current
          ? {
              ...current,
              items: current.items.map((run) =>
                run.id === selected.routing_run.id
                  ? { ...run, latest_evaluation: evaluation }
                  : run,
              ),
              metrics: {
                ...current.metrics,
                evaluation: {
                  ...current.metrics.evaluation,
                  reviewed_count:
                    current.metrics.evaluation.reviewed_count +
                    (selected.routing_run.latest_evaluation ? 0 : 1),
                },
              },
            }
          : current,
      );
      setRefreshToken((current) => current + 1);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Evaluation could not be saved.",
      );
    } finally {
      setSaving(false);
    }
  };

  const manualScope = (ids: string[]) =>
    ids.length === 0
      ? "No manual scope selected"
      : ids
          .map((id) => manuals.find((manual) => manual.id === id)?.label ?? id)
          .join(", ");
  const canGoBack = offset > 0;
  const canGoForward = Boolean(data && offset + data.items.length < data.total);

  return (
    <section
      id="admin-routing"
      className="routing-panel"
      aria-labelledby="admin-routing-title"
    >
      <div className="routing-header">
        <div>
          <p className="admin-eyebrow">LIVE ROUTING · LOCAL DATABASE</p>
          <h2 id="admin-routing-title">Live routing</h2>
          <p>
            Saved routing decisions, research outcomes, and human evaluations.
            Answer review is monitoring, not a measure of engineering
            correctness.
          </p>
        </div>
        <div className="routing-controls">
          <label>
            UTC window
            <select
              value={range}
              onChange={(event) => {
                setRange(Number(event.target.value) as Range);
                setOffset(0);
              }}
            >
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
          <button
            type="button"
            onClick={() => setRefreshToken((current) => current + 1)}
            disabled={loading}
          >
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </div>
      {error && (
        <div className="routing-state routing-state-error" role="alert">
          {error}
        </div>
      )}
      {loading && !data && (
        <div className="routing-state" role="status">
          Loading live routing…
        </div>
      )}
      {data && (
        <>
          <div className="routing-metrics">
            <Metric label="Total runs" value={String(data.metrics.total)} />
            <Metric
              label="Classified"
              value={String(data.metrics.classified)}
            />
            <Metric
              label="Pending / failed"
              value={`${data.metrics.pending} / ${data.metrics.failed}`}
            />
            <Metric
              label="Median / P95"
              value={`${formatMillis(data.metrics.latency.median_ms)} / ${formatMillis(data.metrics.latency.p95_ms)}`}
              detail={`${data.metrics.latency.sample_count} model attempt${data.metrics.latency.sample_count === 1 ? "" : "s"}`}
            />
            <Metric
              label="Human agreement"
              value={
                data.metrics.evaluation.agreement_rate === null
                  ? "Not evaluated"
                  : `${Math.round(data.metrics.evaluation.agreement_rate * 100)}%`
              }
              detail={`${data.metrics.evaluation.reviewed_count} reviewed`}
            />
          </div>
          <div className="routing-breakdown">
            <span>Retrieval: {data.metrics.route_counts.retrieval}</span>
            <span>Conversation: {data.metrics.route_counts.conversation}</span>
            <span>Invalid: {data.metrics.route_counts.invalid}</span>
            {outcomeKeys.map((outcome) => (
              <span key={outcome}>
                {outcomeLabels[outcome]}: {data.metrics.outcome_counts[outcome]}
              </span>
            ))}
            {Object.entries(data.metrics.invalid_reason_counts).map(
              ([reason, count]) => (
                <span key={reason}>
                  {reason}: {count}
                </span>
              ),
            )}
          </div>
          {data.items.length === 0 ? (
            <div className="routing-state">
              No live routing runs fall within this UTC window.
            </div>
          ) : (
            <div className="routing-table-wrap">
              <table className="routing-table">
                <thead>
                  <tr>
                    <th>Timestamp (UTC)</th>
                    <th>Question</th>
                    <th>Route / failure</th>
                    <th>Outcome</th>
                    <th>Model</th>
                    <th>Latency</th>
                    <th>Human label</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((run) => (
                    <tr key={run.id}>
                      <td>{formatDate(run.created_at)}</td>
                      <td>
                        <button
                          type="button"
                          className="routing-run-link"
                          onClick={() => openRun(run)}
                        >
                          {run.question}
                        </button>
                      </td>
                      <td>{routeLabel(run)}</td>
                      <td>
                        {run.outcome
                          ? outcomeLabels[run.outcome]
                          : run.status === "failed"
                            ? "Error"
                            : "Pending"}
                      </td>
                      <td>
                        {run.model}
                        {run.model_digest ? (
                          <small>{run.model_digest}</small>
                        ) : null}
                      </td>
                      <td>{formatMillis(run.routing_ms)}</td>
                      <td>
                        {run.latest_evaluation
                          ? run.latest_evaluation.expected_route
                          : "Not evaluated"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="routing-pagination">
            <span>
              {data.total} total run{data.total === 1 ? "" : "s"}
            </span>
            <div>
              <button
                type="button"
                disabled={!canGoBack || loading}
                onClick={() =>
                  setOffset((current) => Math.max(0, current - pageSize))
                }
              >
                <ChevronLeft size={15} /> Previous
              </button>
              <button
                type="button"
                disabled={!canGoForward || loading}
                onClick={() => setOffset((current) => current + pageSize)}
              >
                Next <ChevronRight size={15} />
              </button>
            </div>
          </div>
        </>
      )}
      {selected && (
        <aside className="routing-detail" aria-label="Routing run detail">
          <div className="routing-detail-top">
            <div>
              <p className="admin-eyebrow">RUN DETAIL</p>
              <h3>{routeLabel(selected.routing_run)}</h3>
            </div>
            <button
              type="button"
              aria-label="Close routing detail"
              onClick={() => setSelected(null)}
            >
              <X size={18} />
            </button>
          </div>
          <dl className="routing-properties">
            <div>
              <dt>Model</dt>
              <dd>{selected.routing_run.model}</dd>
            </div>
            <div>
              <dt>Digest</dt>
              <dd>{selected.routing_run.model_digest ?? "Unavailable"}</dd>
            </div>
            <div>
              <dt>Policy</dt>
              <dd>{selected.policy.version}</dd>
            </div>
            <div>
              <dt>Options</dt>
              <dd>
                <pre>
                  {JSON.stringify(selected.routing_run.model_options, null, 2)}
                </pre>
              </dd>
            </div>
            <div>
              <dt>Selected scope</dt>
              <dd>{manualScope(selected.routing_run.selected_manual_ids)}</dd>
            </div>
            <div>
              <dt>Outcome</dt>
              <dd>
                {selected.routing_run.outcome
                  ? outcomeLabels[selected.routing_run.outcome]
                  : selected.routing_run.status === "failed"
                    ? "Error"
                    : "Pending"}
              </dd>
            </div>
            <div>
              <dt>City</dt>
              <dd>{selected.routing_run.city ?? "Not identified"}</dd>
            </div>
          </dl>
          <Citations citations={selected.routing_run.citations} />
          <h4>Agent trace</h4>
          {selected.routing_run.trace.length ? (
            <ol className="routing-trace">
              {selected.routing_run.trace.map((step, index) => (
                <li key={index}>{step}</li>
              ))}
            </ol>
          ) : (
            <p className="routing-empty-detail">No trace recorded.</p>
          )}
          <h4>Answer review (monitoring)</h4>
          {selected.routing_run.review ? (
            "error" in selected.routing_run.review ? (
              <p className="routing-review">
                {selected.routing_run.review.error}
              </p>
            ) : (
              <dl className="routing-review">
                <div>
                  <dt>Valid</dt>
                  <dd>{selected.routing_run.review.valid ? "Yes" : "No"}</dd>
                </div>
                <div>
                  <dt>Confidence</dt>
                  <dd>{selected.routing_run.review.confidence}</dd>
                </div>
                <div>
                  <dt>Reasoning</dt>
                  <dd>{selected.routing_run.review.reasoning}</dd>
                </div>
              </dl>
            )
          ) : (
            <p className="routing-empty-detail">No answer review recorded.</p>
          )}
          <h4>Trace</h4>
          {traceUnavailable ? (
            <p className="routing-empty-detail">Trace not available.</p>
          ) : trace === null ? (
            <p className="routing-empty-detail">Loading trace…</p>
          ) : trace.length === 0 ? (
            <p className="routing-empty-detail">
              No structured trace recorded.
            </p>
          ) : (
            <ol className="routing-event-timeline">
              {trace.map((event) => (
                <li key={event.seq}>
                  <details>
                    <summary>
                      <span>#{event.seq}</span>
                      <strong>{event.node}</strong>
                      <span
                        className={`routing-kind routing-kind-${event.kind}`}
                      >
                        {event.kind}
                      </span>
                      <span>{formatMillis(event.duration_ms)}</span>
                      <span>{event.summary}</span>
                    </summary>
                    <TraceEventData event={event} />
                  </details>
                </li>
              ))}
            </ol>
          )}
          <h4>Exact submitted transcript</h4>
          {selected.context_messages.length === 0 ? (
            <p className="routing-empty-detail">Loading transcript…</p>
          ) : (
            <ol className="routing-transcript">
              {selected.context_messages.map((message) => (
                <li key={message.id}>
                  <strong>
                    {message.role === "user" ? "User" : "Assistant"}
                  </strong>
                  <span>{message.content}</span>
                </li>
              ))}
            </ol>
          )}
          <h4>Policy</h4>
          <pre className="routing-policy">
            {selected.policy.system_prompt || "Loading policy…"}
          </pre>
          <h4>Human evaluation</h4>
          <div className="routing-evaluation">
            <label>
              Expected route
              <select
                value={expectedRoute}
                onChange={(event) =>
                  setExpectedRoute(event.target.value as Route)
                }
              >
                <option value="retrieval">retrieval</option>
                <option value="conversation">conversation</option>
                <option value="invalid">invalid</option>
              </select>
            </label>
            <label>
              Comment
              <textarea
                value={comment}
                maxLength={5000}
                onChange={(event) => setComment(event.target.value)}
              />
            </label>
            <button
              type="button"
              onClick={() => void saveEvaluation()}
              disabled={saving}
            >
              <Save size={15} /> {saving ? "Saving…" : "Save evaluation"}
            </button>
          </div>
          {selected.evaluations.length > 0 && (
            <ol className="routing-evaluations">
              {selected.evaluations.map((evaluation) => (
                <li key={evaluation.id}>
                  <strong>{evaluation.expected_route}</strong>
                  <span>{evaluation.comment || "No comment"}</span>
                  <small>{formatDate(evaluation.created_at)}</small>
                </li>
              ))}
            </ol>
          )}
        </aside>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <article>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}
function TraceEventData({ event }: { event: RoutingTraceEvent }) {
  const data = event.data;
  if (event.kind === "retrieval") {
    const chunks = Array.isArray(data.chunks) ? data.chunks : [];
    return (
      <div className="routing-trace-data">
        <p>Query: {String(data.query ?? "")}</p>
        <table>
          <thead>
            <tr>
              <th>Document</th>
              <th>Provision</th>
              <th>Pages</th>
              <th>Rank scores</th>
            </tr>
          </thead>
          <tbody>
            {chunks.map((chunk, index) => {
              const row = chunk as Record<string, unknown>;
              return (
                <tr key={String(row.chunk_id ?? index)}>
                  <td>{String(row.document_id ?? "—")}</td>
                  <td>{String(row.provision_identifier ?? "—")}</td>
                  <td>{String(row.pages ?? "—")}</td>
                  <td>{formatTraceScores(row)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }
  if (event.kind === "web_search") {
    const results = Array.isArray(data.results) ? data.results : [];
    return (
      <div className="routing-trace-data">
        <p>Query sent: {String(data.query_sent ?? "")}</p>
        <ul>
          {results.map((result, index) => {
            const row = result as Record<string, unknown>;
            const url = typeof row.url === "string" ? row.url : "";
            return <li key={url || index}>{url || "No URL"}</li>;
          })}
        </ul>
      </div>
    );
  }
  if (event.kind === "llm_call") {
    return (
      <div className="routing-trace-data">
        <strong>Prompt</strong>
        <pre>{typeof data.prompt === "string" ? data.prompt : ""}</pre>
        <strong>Output</strong>
        <pre>{typeof data.output === "string" ? data.output : ""}</pre>
      </div>
    );
  }
  return (
    <pre className="routing-trace-data">{JSON.stringify(data, null, 2)}</pre>
  );
}

function formatTraceScores(chunk: Record<string, unknown>) {
  return ["rank", "fts_rank", "vector_rank", "rrf_score"]
    .filter((key) => chunk[key] !== undefined && chunk[key] !== null)
    .map((key) => key + ": " + String(chunk[key]))
    .join(" · ");
}
