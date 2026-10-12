export type Route = "retrieval" | "conversation" | "invalid";
export type ReasonCode =
  | "code_lookup"
  | "greeting"
  | "clarification"
  | "workspace_help"
  | "unsafe_request"
  | "prompt_injection"
  | "off_topic";

export type RouteDecision = {
  route: Route;
  reason_code: ReasonCode;
  rationale: string;
};
export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};
export type ApiMessage = {
  id: string;
  conversation_id: string;
  sequence: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  origin: "user" | "router_status" | "legacy_demo" | "assistant_answer";
  selected_manual_ids: ManualId[];
  routing_run_id: string | null;
};
export type ManualId = string;
export type Evaluation = {
  id: string;
  run_id: string;
  expected_route: Route;
  comment: string;
  created_at: string;
};
export type Outcome =
  | "kb_answer"
  | "web_answer"
  | "clarification"
  | "blocked"
  | "insufficient"
  | "direct"
  | "error";
export type Citation =
  | {
      type: "provision";
      document_id?: string;
      identifier?: string;
      title?: string | null;
      pages?: string | null;
      quote?: string;
    }
  | { type: "web"; title?: string | null; url?: string };
export type AnswerReview =
  { valid: boolean; confidence: number; reasoning: string } | { error: string };
export type RoutingRun = {
  id: string;
  conversation_id: string;
  user_message_id: string;
  question: string;
  status: "pending" | "classified" | "failed";
  decision: RouteDecision | null;
  outcome: Outcome | null;
  citations: Citation[];
  review: AnswerReview | null;
  trace: string[];
  city: string | null;
  error_code: string | null;
  created_at: string;
  completed_at: string | null;
  model: string;
  model_digest: string | null;
  policy_version: string;
  model_options: Record<string, unknown>;
  elapsed_ms: number | null;
  routing_ms: number | null;
  model_calls: number;
  context_message_ids: string[];
  selected_manual_ids: ManualId[];
  selected_source_version_ids: string[];
  latest_evaluation: Evaluation | null;
};
export type ConversationDetail = {
  conversation: ConversationSummary;
  messages: ApiMessage[];
  pending_run: RoutingRun | null;
};
export type MessageResult = { messages: ApiMessage[]; routing_run: RoutingRun };
export type ApiErrorPayload = {
  code: string;
  message: string;
  conversation_id?: string | null;
  message_id?: string | null;
  run_id?: string | null;
  messages?: ApiMessage[] | null;
  routing_run?: RoutingRun | null;
};
export type RoutingMetrics = {
  total: number;
  pending: number;
  failed: number;
  classified: number;
  route_counts: Record<Route, number>;
  outcome_counts: Record<Outcome, number>;
  invalid_reason_counts: Record<string, number>;
  latency: {
    sample_count: number;
    median_ms: number | null;
    p95_ms: number | null;
  };
  evaluation: {
    reviewed_count: number;
    agreement_count: number;
    agreement_rate: number | null;
  };
};
export type RoutingList = {
  window: { days: 7 | 30 | 90; start: string; end: string };
  metrics: RoutingMetrics;
  items: RoutingRun[];
  total: number;
  offset: number;
  limit: number;
};
export type RoutingDetail = {
  routing_run: RoutingRun;
  context_messages: ApiMessage[];
  policy: {
    version: string;
    system_prompt: string;
    response_schema: Record<string, unknown>;
  };
  evaluations: Evaluation[];
};
export type RoutingTraceEvent = {
  seq: number;
  node: string;
  kind: "node" | "llm_call" | "retrieval" | "web_search" | "decision";
  started_at: string;
  duration_ms: number;
  summary: string;
  data: Record<string, unknown>;
};
export type RoutingTrace = { run_id: string; events: RoutingTraceEvent[] };
export type Manual = {
  id: ManualId;
  label: string;
  title: string;
  filename: string;
  jurisdiction: string;
  url: string;
  current_source_version_id?: string | null;
  edition?: string | null;
};
export type ImportedConversation = {
  id: string;
  title: string;
  created_at: string;
  messages: Array<{
    id: string;
    role: "user" | "assistant";
    content: string;
    created_at: string;
    selected_manual_ids: ManualId[];
  }>;
};

export class ApiError extends Error {
  readonly status: number;
  readonly payload: ApiErrorPayload | null;
  constructor(
    status: number,
    message: string,
    payload: ApiErrorPayload | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      signal,
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new ApiError(0, "The local API is unavailable.");
  }
  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    const structured = isErrorPayload(payload) ? payload : null;
    throw new ApiError(
      response.status,
      structured?.message ?? `Request failed (${response.status}).`,
      structured,
    );
  }
  return payload as T;
}

function isErrorPayload(value: unknown): value is ApiErrorPayload {
  return Boolean(
    value && typeof value === "object" && "code" in value && "message" in value,
  );
}

export const api = {
  listConversations: (signal?: AbortSignal) =>
    request<{ conversations: ConversationSummary[] }>(
      "/conversations",
      {},
      signal,
    ),
  createConversation: (signal?: AbortSignal) =>
    request<ConversationSummary>(
      "/conversations",
      { method: "POST", body: "{}" },
      signal,
    ),
  getConversation: (id: string, signal?: AbortSignal) =>
    request<ConversationDetail>(
      `/conversations/${encodeURIComponent(id)}`,
      {},
      signal,
    ),
  importConversations: (conversations: ImportedConversation[]) =>
    request<{ imported_conversation_ids: string[] }>("/conversations/import", {
      method: "POST",
      body: JSON.stringify({ conversations }),
    }),
  sendMessage: (
    conversationId: string,
    body: {
      message_id: string;
      content: string;
      selected_manual_ids: ManualId[];
    },
  ) =>
    request<MessageResult>(
      `/conversations/${encodeURIComponent(conversationId)}/messages`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  listRoutingRuns: (
    days: 7 | 30 | 90,
    offset: number,
    limit: number,
    signal?: AbortSignal,
  ) =>
    request<RoutingList>(
      `/routing-runs?days=${days}&offset=${offset}&limit=${limit}`,
      {},
      signal,
    ),
  getRoutingRun: (id: string, signal?: AbortSignal) =>
    request<RoutingDetail>(
      `/routing-runs/${encodeURIComponent(id)}`,
      {},
      signal,
    ),
  getRoutingTrace: (id: string, signal?: AbortSignal) =>
    request<RoutingTrace>(
      `/routing-runs/${encodeURIComponent(id)}/trace`,
      {},
      signal,
    ),
  createEvaluation: (runId: string, expected_route: Route, comment: string) =>
    request<Evaluation>(
      `/routing-runs/${encodeURIComponent(runId)}/evaluations`,
      { method: "POST", body: JSON.stringify({ expected_route, comment }) },
    ),
  manuals: (signal?: AbortSignal) =>
    request<{ manuals: Manual[] }>("/manuals", {}, signal),
};
