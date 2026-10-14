import { useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  BookOpen,
  ChevronRight,
  Clock3,
  Database,
  FileSearch,
  HelpCircle,
  LayoutDashboard,
  Layers3,
  TrendingUp,
  Users,
} from "lucide-react";
import {
  demoActivity,
  demoPeriodLabel,
  manualIds,
  unansweredReasonLabels,
  type DemoDay,
  type ManualId,
} from "./analytics";
import { manuals } from "./manuals";
import RoutingPanel from "./RoutingPanel";
import "./admin.css";

type Range = 7 | 30 | 90;
type Totals = Record<ManualId, number>;

const number = new Intl.NumberFormat("en-US");
const dateFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};
const percentile = (values: number[], value: number) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length
    ? sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * value) - 1)]
    : 0;
};
const sumManual = (
  days: DemoDay[],
  key: "manualPassages" | "manualReferences",
) =>
  days.reduce<Totals>(
    (acc, day) => {
      manualIds.forEach((id) => {
        acc[id] += day[key][id];
      });
      return acc;
    },
    { transportation: 0, environmental: 0, fire: 0 },
  );

export default function AdminWorkspace({ active }: { active: boolean }) {
  const [range, setRange] = useState<Range>(30);
  const days = useMemo(() => demoActivity.slice(-range), [range]);
  const summary = useMemo(() => {
    const responseSamples = days.flatMap((day) => day.responseSamples);
    const retrievalLatencies = days.flatMap((day) => day.retrievalLatencies);
    const total = (field: keyof DemoDay) =>
      days.reduce((value, day) => value + Number(day[field] || 0), 0);
    const activeEngineers = new Set(days.flatMap((day) => day.engineerIds))
      .size;
    return {
      questions: total("questions"),
      activeEngineers,
      unanswered: total("unanswered"),
      responseMedian: median(responseSamples),
      requests: total("retrievalRequests"),
      retrievalMedian: median(retrievalLatencies),
      retrievalP95: percentile(retrievalLatencies, 0.95),
      sources: total("sourceReturns"),
      noMatches: total("noMatches"),
      references: sumManual(days, "manualReferences"),
      passages: sumManual(days, "manualPassages"),
      reasons: unansweredReasonLabels.map((reason) => ({
        label: reason,
        value: days.reduce((n, day) => n + day.unansweredReasons[reason], 0),
      })),
    };
  }, [days]);
  const maxQuestions = Math.max(...days.map((day) => day.questions));
  const chartPoints = days
    .map((day, index) => {
      const x = days.length === 1 ? 50 : (index / (days.length - 1)) * 100;
      const y = 92 - (day.questions / maxQuestions) * 76;
      return `${x},${y}`;
    })
    .join(" ");
  const periodStart = dateFormat.format(new Date(days[0].date + "T12:00:00Z"));
  const periodEnd = dateFormat.format(
    new Date(days.at(-1)!.date + "T12:00:00Z"),
  );
  const sourceRate = summary.requests
    ? Math.round((summary.sources / summary.requests) * 100)
    : 0;
  const nav = (id: string) =>
    document
      .getElementById(id)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="admin-workspace">
      <aside className="admin-sidebar" aria-label="Analytics navigation">
        <div className="admin-sidebar-brand">
          <span className="admin-brand-mark">
            <Database size={18} />
          </span>
          <span>Reference Ops</span>
        </div>
        <p className="admin-sidebar-caption">
          LOCAL ROUTING + DEMONSTRATION DATA
        </p>
        <nav className="admin-nav">
          <a
            href="#admin-overview"
            onClick={(event) => {
              event.preventDefault();
              nav("admin-overview");
            }}
          >
            <LayoutDashboard size={17} /> Overview
          </a>
          <a
            href="#admin-routing"
            onClick={(event) => {
              event.preventDefault();
              nav("admin-routing");
            }}
          >
            <Activity size={17} /> Live routing
          </a>
          <a
            href="#admin-retrieval"
            onClick={(event) => {
              event.preventDefault();
              nav("admin-retrieval");
            }}
          >
            <FileSearch size={17} /> Retrieval performance
          </a>
          <a
            href="#admin-activity"
            onClick={(event) => {
              event.preventDefault();
              nav("admin-activity");
            }}
          >
            <Activity size={17} /> Recent activity
          </a>
        </nav>
        <div className="admin-sidebar-foot">
          Live routing is local and auditable.
          <br />
          Demo retrieval period: {demoPeriodLabel}
        </div>
      </aside>

      <div className="admin-main">
        <header className="admin-header">
          <div>
            <p className="admin-eyebrow">
              ANALYTICS WORKSPACE · DEMONSTRATION DATA
            </p>
            <h1>Reference operations</h1>
            <p>
              Code-reference workspace activity for a fixed demonstration
              period.
            </p>
          </div>
          <label className="admin-range">
            <span>Date range</span>
            <select
              value={range}
              onChange={(event) =>
                setRange(Number(event.target.value) as Range)
              }
              aria-label="Demonstration date range"
            >
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
        </header>
        <div className="admin-period">
          <Clock3 size={15} />
          Showing {periodStart} – {periodEnd}, 2026 <span>·</span> within fixed
          demo period {demoPeriodLabel}
        </div>

        <RoutingPanel active={active} />

        <section
          id="admin-overview"
          className="admin-section"
          aria-labelledby="admin-overview-title"
        >
          <div className="admin-section-heading">
            <div>
              <p className="admin-eyebrow">OVERVIEW · DEMONSTRATION DATA</p>
              <h2 id="admin-overview-title">Activity summary</h2>
            </div>
          </div>
          <div className="admin-kpis">
            <Metric
              icon={<HelpCircle />}
              label="Question volume"
              value={number.format(summary.questions)}
              detail="Questions in selected period"
            />
            <Metric
              icon={<Users />}
              label="Active engineers"
              value={number.format(summary.activeEngineers)}
              detail="Distinct engineers in selected period"
            />
            <Metric
              icon={<Clock3 />}
              label="Median response time"
              value={`${summary.responseMedian} ms`}
              detail="Answered-question responses"
            />
            <Metric
              icon={<Activity />}
              label="Unanswered questions"
              value={number.format(summary.unanswered)}
              detail="Needs follow-up or no match"
              tone="warning"
            />
          </div>
          <div className="admin-grid admin-overview-grid">
            <article className="admin-card admin-trend-card">
              <div className="admin-card-top">
                <div>
                  <h3>Usage trend</h3>
                  <p>Questions submitted by day</p>
                </div>
                <TrendingUp size={19} />
              </div>
              <div className="admin-chart-wrap">
                <svg
                  className="admin-line-chart"
                  viewBox="0 0 100 100"
                  role="img"
                  aria-label={`Question volume ranges from ${Math.min(...days.map((d) => d.questions))} to ${maxQuestions} per day across the selected period.`}
                  preserveAspectRatio="none"
                >
                  <line x1="0" y1="92" x2="100" y2="92" />
                  <line x1="0" y1="54" x2="100" y2="54" />
                  <polyline points={chartPoints} />
                </svg>
              </div>
              <div className="admin-chart-labels">
                <span>{periodStart}</span>
                <span>{periodEnd}</span>
              </div>
            </article>
            <article className="admin-card">
              <div className="admin-card-top">
                <div>
                  <h3>Commonly referenced manuals</h3>
                  <p>Manual selections attached to questions</p>
                </div>
                <BookOpen size={19} />
              </div>
              <ManualBars values={summary.references} unit="references" />
            </article>
            <article className="admin-card">
              <div className="admin-card-top">
                <div>
                  <h3>Unanswered question breakdown</h3>
                  <p>Reason recorded in the demo activity</p>
                </div>
                <HelpCircle size={19} />
              </div>
              <div className="admin-breakdown">
                {summary.reasons.map((reason) => (
                  <div key={reason.label}>
                    <div>
                      <span>{reason.label}</span>
                      <strong>{reason.value}</strong>
                    </div>
                    <div className="admin-progress">
                      <i
                        style={{
                          width: `${summary.unanswered ? (reason.value / summary.unanswered) * 100 : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section
          id="admin-retrieval"
          className="admin-section"
          aria-labelledby="admin-retrieval-title"
        >
          <div className="admin-section-heading">
            <div>
              <p className="admin-eyebrow">
                RETRIEVAL PERFORMANCE · DEMONSTRATION DATA
              </p>
              <h2 id="admin-retrieval-title">Source retrieval signals</h2>
            </div>
            <span className="admin-system-label">
              <Layers3 size={15} />
              Demonstration retrieval data
            </span>
          </div>
          <div className="admin-kpis admin-retrieval-kpis">
            <Metric
              icon={<Database />}
              label="Total retrieval requests"
              value={number.format(summary.requests)}
              detail="Retrieval attempts"
            />
            <Metric
              icon={<Clock3 />}
              label="Median retrieval latency"
              value={`${summary.retrievalMedian} ms`}
              detail={`P95: ${summary.retrievalP95} ms`}
            />
            <Metric
              icon={<FileSearch />}
              label="Requests with source passages"
              value={`${sourceRate}%`}
              detail={`${number.format(summary.sources)} requests returned passages`}
              tone="success"
            />
            <Metric
              icon={<HelpCircle />}
              label="No-match requests"
              value={number.format(summary.noMatches)}
              detail="No passages returned"
              tone="warning"
            />
          </div>
          <div className="admin-grid admin-retrieval-grid">
            <article className="admin-card">
              <div className="admin-card-top">
                <div>
                  <h3>Retrieved passages by manual</h3>
                  <p>Passages returned across retrieval requests</p>
                </div>
                <FileSearch size={19} />
              </div>
              <ManualBars values={summary.passages} unit="passages" />
            </article>
            <aside className="admin-retrieval-note">
              <strong>Interpretation note</strong>
              <p>
                Retrieved passages indicate that source material was returned.
                They do not prove answer correctness.
              </p>
            </aside>
          </div>
        </section>

        <section
          id="admin-activity"
          className="admin-section"
          aria-labelledby="admin-activity-title"
        >
          <div className="admin-section-heading">
            <div>
              <p className="admin-eyebrow">
                RECENT ACTIVITY · DEMONSTRATION DATA
              </p>
              <h2 id="admin-activity-title">Recent daily activity</h2>
            </div>
            <span className="admin-muted">
              Latest five days in selected range
            </span>
          </div>
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th scope="col">Activity date</th>
                  <th scope="col">Question volume</th>
                  <th scope="col">Retrieval status</th>
                  <th scope="col">Source passages</th>
                  <th scope="col">Median latency</th>
                  <th scope="col">Summary</th>
                </tr>
              </thead>
              <tbody>
                {[...days]
                  .reverse()
                  .slice(0, 5)
                  .map((day) => {
                    const status = day.noMatches
                      ? "Review no matches"
                      : "Source passages returned";
                    return (
                      <tr key={day.date}>
                        <td>
                          <strong>
                            {dateFormat.format(
                              new Date(day.date + "T12:00:00Z"),
                            )}
                            , 2026
                          </strong>
                          <span>Demonstration activity</span>
                        </td>
                        <td>{day.questions} questions</td>
                        <td>
                          <span
                            className={`admin-badge ${day.noMatches ? "admin-badge-warning" : "admin-badge-success"}`}
                          >
                            {status}
                          </span>
                        </td>
                        <td>
                          {number.format(
                            Object.values(day.manualPassages).reduce(
                              (a, b) => a + b,
                              0,
                            ),
                          )}{" "}
                          passages
                        </td>
                        <td>{median(day.retrievalLatencies)} ms</td>
                        <td>
                          <a
                            className="admin-row-link"
                            href="#admin-retrieval"
                            aria-label="View retrieval summary for the selected period"
                            onClick={(event) => {
                              event.preventDefault();
                              nav("admin-retrieval");
                            }}
                          >
                            <ChevronRight size={18} />
                          </a>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone?: "success" | "warning";
}) {
  return (
    <article className={`admin-metric ${tone ? `admin-metric-${tone}` : ""}`}>
      <span className="admin-metric-icon">{icon}</span>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}
function ManualBars({ values, unit }: { values: Totals; unit: string }) {
  const max = Math.max(...Object.values(values));
  return (
    <div className="admin-bars">
      {manuals.map((manual) => {
        const value = values[manual.id as ManualId];
        const hasDemoData = Number.isFinite(value);
        return (
          <div className="admin-bar-row" key={manual.id}>
            <div>
              <a href={manual.url} target="_blank" rel="noreferrer">
                {manual.label}
              </a>
              <strong>
                {hasDemoData
                  ? number.format(value) + " " + unit
                  : "No demo data"}
              </strong>
            </div>
            <div className="admin-progress">
              <i
                style={{
                  width:
                    hasDemoData && max
                      ? String((value / max) * 100) + "%"
                      : "0%",
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
