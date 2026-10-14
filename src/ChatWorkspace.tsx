import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  Clipboard,
  FileText,
  Library,
  MessageSquare,
  Plus,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import {
  api,
  ApiError,
  type ApiMessage,
  type ConversationDetail,
  type ConversationSummary,
  type ManualId,
  type RoutingRun,
} from "./api";
import { Citations, outcomeLabels } from "./Citations";
import { importLegacyConversations } from "./legacyImport";
import { manuals, type Manual } from "./manuals";
import { sample, sampleId, type SampleMessage } from "./sampleConversation";
import "./chat.css";

type DisplayMessage = ApiMessage | SampleMessage;

const promptCards = [
  "Which drainage criteria apply to this Austin site?",
  "What do the manuals say about design rainfall?",
  "Which fire-protection provisions should I review?",
];

function mergeMessages(current: ApiMessage[], incoming: ApiMessage[]) {
  const byId = new Map(current.map((message) => [message.id, message]));
  for (const message of incoming) byId.set(message.id, message);
  return [...byId.values()].sort(
    (left, right) =>
      left.sequence - right.sequence || left.id.localeCompare(right.id),
  );
}
function mergeConversations(
  current: ConversationSummary[],
  incoming: ConversationSummary[],
) {
  const byId = new Map(
    current.map((conversation) => [conversation.id, conversation]),
  );
  for (const conversation of incoming) byId.set(conversation.id, conversation);
  return [...byId.values()].sort((left, right) =>
    right.updated_at.localeCompare(left.updated_at),
  );
}

export default function ChatWorkspace() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [details, setDetails] = useState<Record<string, ConversationDetail>>(
    {},
  );
  const [activeId, setActiveId] = useState(sampleId);
  const [selectedManualIds, setSelectedManualIds] = useState<ManualId[]>([]);
  const [sourceManual, setSourceManual] = useState<Manual | null>(
    manuals[0] ?? null,
  );
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [pendingConversationIds, setPendingConversationIds] = useState<
    Set<string>
  >(new Set());
  const [runs, setRuns] = useState<Record<string, RoutingRun>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [migrationError, setMigrationError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const activeIdRef = useRef(activeId);
  const creatingConversations = useRef(
    new Map<string, Promise<ConversationSummary | null>>(),
  );
  const bottomRef = useRef<HTMLDivElement>(null);
  const runsRef = useRef(runs);
  runsRef.current = runs;

  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const imported = await importLegacyConversations(
          localStorage,
          api,
          () => cancelled,
        );
        if (cancelled) return;
        if (imported) {
          setConversations((current) => mergeConversations(current, imported));
          setNotice(
            "Saved browser conversations were imported into the local backend.",
          );
        } else {
          const result = await api.listConversations();
          if (!cancelled)
            setConversations((current) =>
              mergeConversations(current, result.conversations),
            );
        }
      } catch (error) {
        if (cancelled) return;
        setMigrationError(
          error instanceof Error
            ? error.message
            : "Saved browser conversations could not be imported.",
        );
        try {
          const result = await api.listConversations();
          if (!cancelled)
            setConversations((current) =>
              mergeConversations(current, result.conversations),
            );
        } catch {
          /* backend unavailable: the error above remains visible and browser data is untouched */
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (activeId === sampleId || details[activeId]) return;
    const controller = new AbortController();
    void api
      .getConversation(activeId, controller.signal)
      .then((detail) => {
        setDetails((current) => {
          const previous = current[detail.conversation.id];
          return {
            ...current,
            [detail.conversation.id]: {
              ...detail,
              messages: mergeMessages(
                previous?.messages ?? [],
                detail.messages,
              ),
            },
          };
        });
        if (detail.pending_run)
          setRuns((current) => ({
            ...current,
            [detail.pending_run!.id]: detail.pending_run!,
          }));
      })
      .catch((error) => {
        if (!(error instanceof DOMException && error.name === "AbortError"))
          setNotice(
            error instanceof Error
              ? error.message
              : "Conversation could not be loaded.",
          );
      });
    return () => controller.abort();
  }, [activeId, details]);

  const activeDetail = activeId === sampleId ? undefined : details[activeId];
  const activeMessages: DisplayMessage[] =
    activeId === sampleId ? sample.messages : (activeDetail?.messages ?? []);
  const activeTitle =
    activeId === sampleId
      ? sample.title
      : (conversations.find((conversation) => conversation.id === activeId)
          ?.title ??
        activeDetail?.conversation.title ??
        "Loading conversation");
  useEffect(() => {
    if (!activeDetail) return;
    const controller = new AbortController();
    for (const message of activeDetail.messages) {
      if (message.origin !== "assistant_answer" || !message.routing_run_id)
        continue;
      const runId = message.routing_run_id;
      if (runsRef.current[runId]) continue;
      void api
        .getRoutingRun(runId, controller.signal)
        .then((detail) => {
          if (!controller.signal.aborted)
            setRuns((current) => ({ ...current, [runId]: detail.routing_run }));
        })
        .catch((error) => {
          if (!controller.signal.aborted)
            setNotice(
              error instanceof Error
                ? error.message
                : "Answer sources could not be loaded.",
            );
        });
    }
    return () => controller.abort();
  }, [activeId, activeDetail?.messages]);
  const activeDraft = drafts[activeId] ?? "";
  const selectedManuals = manuals.filter((manual) =>
    selectedManualIds.includes(manual.id),
  );
  const activePending =
    activeId !== sampleId && pendingConversationIds.has(activeId);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeId, activeMessages.length, activePending]);

  const chooseManual = (id: ManualId) => {
    setSelectedManualIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    );
  };

  const mergeServerResult = (
    conversationId: string,
    result: { messages: ApiMessage[]; routing_run: RoutingRun },
  ) => {
    setDetails((current) => {
      const existing = current[conversationId];
      if (!existing) return current;
      return {
        ...current,
        [conversationId]: {
          ...existing,
          messages: mergeMessages(existing.messages, result.messages),
          pending_run:
            result.routing_run.status === "pending" ? result.routing_run : null,
        },
      };
    });
    setRuns((current) => ({
      ...current,
      [result.routing_run.id]: result.routing_run,
    }));
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              updated_at:
                result.routing_run.completed_at ??
                result.routing_run.created_at,
            }
          : conversation,
      ),
    );
    void api
      .listConversations()
      .then((list) =>
        setConversations((current) =>
          mergeConversations(current, list.conversations),
        ),
      )
      .catch(() => undefined);
  };

  const selectChat = (id: string) => {
    activeIdRef.current = id;
    setActiveId(id);
  };

  const persistChat = (localId: string) => {
    const inFlight = creatingConversations.current.get(localId);
    if (inFlight) return inFlight;
    const request = api
      .createConversation()
      .then((created) => {
        setConversations((current) => [
          created,
          ...current.filter(
            (conversation) =>
              conversation.id !== localId && conversation.id !== created.id,
          ),
        ]);
        setDetails((current) => {
          const { [localId]: temporary, ...others } = current;
          return {
            ...others,
            [created.id]: {
              conversation: created,
              messages: temporary?.messages ?? [],
              pending_run: temporary?.pending_run ?? null,
            },
          };
        });
        setDrafts((current) => {
          const { [localId]: draft, ...others } = current;
          return { ...others, [created.id]: draft ?? others[created.id] ?? "" };
        });
        if (activeIdRef.current === localId) selectChat(created.id);
        return created;
      })
      .catch((error) => {
        setNotice(
          (error instanceof Error
            ? error.message
            : "A new conversation could not be created.") +
            " Your draft is still here; sending will retry creation.",
        );
        return null;
      })
      .finally(() => creatingConversations.current.delete(localId));
    creatingConversations.current.set(localId, request);
    return request;
  };

  const createChat = () => {
    const id = "local-" + crypto.randomUUID();
    const now = new Date().toISOString();
    const temporary: ConversationSummary = {
      id,
      title: "New conversation",
      created_at: now,
      updated_at: now,
    };
    setConversations((current) => [temporary, ...current]);
    setDetails((current) => ({
      ...current,
      [id]: { conversation: temporary, messages: [], pending_run: null },
    }));
    selectChat(id);
    setNotice(null);
    void persistChat(id);
    return id;
  };

  const send = async () => {
    const content = activeDraft;
    if (!content.trim() || activePending) return;
    const originId = activeId;
    const scope = [...selectedManualIds];
    let destinationId = originId;
    if (originId === sampleId) {
      destinationId = createChat();
      const localId = destinationId;
      setDrafts((current) => ({
        ...current,
        [sampleId]:
          current[sampleId] === content ? "" : (current[sampleId] ?? ""),
        [localId]: content,
      }));
    }
    setPendingConversationIds((current) => new Set(current).add(destinationId));
    try {
      if (destinationId.startsWith("local-")) {
        const created = await persistChat(destinationId);
        if (!created) return;
        setPendingConversationIds((current) => {
          const next = new Set(current);
          next.delete(destinationId);
          next.add(created.id);
          return next;
        });
        destinationId = created.id;
      }
      const messageId = crypto.randomUUID();
      const optimisticMessage: ApiMessage = {
        id: messageId,
        conversation_id: destinationId,
        sequence:
          (details[destinationId]?.messages ?? []).reduce(
            (maximum, message) => Math.max(maximum, message.sequence),
            -1,
          ) + 1,
        role: "user",
        content,
        created_at: new Date().toISOString(),
        origin: "user",
        selected_manual_ids: scope,
        routing_run_id: null,
      };
      setDetails((current) => {
        const existing = current[destinationId];
        if (!existing) return current;
        return {
          ...current,
          [destinationId]: {
            ...existing,
            messages: mergeMessages(existing.messages, [optimisticMessage]),
          },
        };
      });
      setDrafts((current) =>
        current[destinationId] === content
          ? { ...current, [destinationId]: "" }
          : current,
      );
      try {
        const result = await api.sendMessage(destinationId, {
          message_id: messageId,
          content,
          selected_manual_ids: scope,
        });
        mergeServerResult(destinationId, result);
        setNotice(null);
      } catch (error) {
        const apiError = error instanceof ApiError ? error : null;
        const persistedMessages = apiError?.payload?.messages;
        const persistedRun = apiError?.payload?.routing_run;
        if (persistedMessages && persistedRun)
          mergeServerResult(destinationId, {
            messages: persistedMessages,
            routing_run: persistedRun,
          });
        else {
          setDetails((current) => {
            const existing = current[destinationId];
            if (!existing) return current;
            return {
              ...current,
              [destinationId]: {
                ...existing,
                messages: existing.messages.filter(
                  (message) => message.id !== messageId,
                ),
              },
            };
          });
          setDrafts((current) =>
            current[destinationId]
              ? current
              : { ...current, [destinationId]: content },
          );
          setNotice(
            "The question may not have been saved. Reload this conversation to reconcile before sending it again.",
          );
        }
      }
    } finally {
      setPendingConversationIds((current) => {
        const next = new Set(current);
        next.delete(destinationId);
        return next;
      });
    }
  };

  const copy = async (message: DisplayMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedId(message.id);
      window.setTimeout(() => setCopiedId(null), 1600);
    } catch {
      setCopiedId(null);
    }
  };

  const manualName = (id: ManualId) =>
    manuals.find((manual) => manual.id === id)?.label ?? id;

  return (
    <section className="chat-workspace" aria-label="Code reference workspace">
      <aside className="chat-sidebar">
        <div className="chat-brand">
          <div className="chat-brand-mark">
            <BookOpen size={19} />
          </div>
          <div>
            <strong>Reference desk</strong>
            <span>Manual review workspace</span>
          </div>
        </div>
        <button
          type="button"
          className="chat-new-button"
          onClick={() => createChat()}
        >
          <Plus size={17} /> New chat
        </button>
        <div className="chat-side-section">
          <div className="chat-side-heading">Conversations</div>
          <button
            type="button"
            className={`chat-history ${activeId === sampleId ? "is-active" : ""}`}
            onClick={() => selectChat(sampleId)}
          >
            <Sparkles size={15} />
            <span>Sample workflow</span>
            <small>DEMO</small>
          </button>
          {conversations.map((conversation) => (
            <button
              type="button"
              className={`chat-history ${activeId === conversation.id ? "is-active" : ""}`}
              key={conversation.id}
              onClick={() => selectChat(conversation.id)}
            >
              <MessageSquare size={15} />
              <span>{conversation.title}</span>
            </button>
          ))}
        </div>
        <div className="chat-side-section chat-library">
          <div className="chat-side-heading">
            <span>Manual library</span>
            <Library size={14} />
          </div>
          <p className="chat-scope-label" id="manual-scope-label">
            Limit search to selected manuals (optional)
          </p>
          {manuals.map((manual) => (
            <div className="chat-manual-row" key={manual.id}>
              <button
                type="button"
                className={`chat-manual-select ${selectedManualIds.includes(manual.id) ? "is-selected" : ""}`}
                onClick={() => chooseManual(manual.id)}
                aria-describedby="manual-scope-label"
                aria-pressed={selectedManualIds.includes(manual.id)}
              >
                <span className="chat-manual-check">
                  {selectedManualIds.includes(manual.id) && <Check size={12} />}
                </span>
                <span>
                  <strong>{manual.label}</strong>
                  <small>Reference available</small>
                </span>
              </button>
              <button
                className="chat-manual-open"
                type="button"
                aria-label={`View ${manual.title} details`}
                onClick={() => setSourceManual(manual)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          ))}
        </div>
        <div className="chat-sidebar-note">
          <FileText size={15} />
          <span>
            Research workspace
            <br />
            Not an engineering authority
          </span>
        </div>
      </aside>
      <div className="chat-main">
        <header className="chat-header">
          <div>
            <div className="chat-eyebrow">
              {activeId === sampleId
                ? "Illustrative conversation"
                : activeId.startsWith("local-")
                  ? "Creating conversation"
                  : "Saved conversation"}
            </div>
            <h1>{activeTitle}</h1>
          </div>
          <div className="chat-header-status">
            <span className="chat-status-dot" /> Local research <span>•</span>{" "}
            {selectedManuals.length
              ? selectedManuals.length +
                " manual" +
                (selectedManuals.length === 1 ? "" : "s") +
                " selected"
              : "Search all manuals"}
          </div>
        </header>
        {migrationError && (
          <div className="chat-notice chat-notice-error" role="alert">
            {migrationError}
          </div>
        )}
        {notice && (
          <div className="chat-notice" role="status">
            {notice}
          </div>
        )}
        <div className="chat-message-scroll">
          <div className="chat-intro">
            <div className="chat-intro-icon">
              <Sparkles size={19} />
            </div>
            <div>
              <strong>Research the Austin criteria manuals</strong>
              <p>
                Questions are answered from the criteria manuals with citations.
                Web search is used only as a clearly labelled fallback. Confirm
                the applicable source version and project context yourself.
              </p>
            </div>
          </div>
          {activeMessages.length === 0 && (
            <div className="chat-empty">
              <BookOpen size={28} />
              <h2>Start a reference question</h2>
              <p>
                Draft a question; optionally select manuals to narrow the
                search.
              </p>
            </div>
          )}
          <div className="chat-thread" aria-live="polite">
            {activeMessages.map((message) => {
              const run =
                "routing_run_id" in message && message.routing_run_id
                  ? runs[message.routing_run_id]
                  : undefined;
              const isNotice =
                "origin" in message && message.origin === "router_status";
              const isAnswer =
                "origin" in message && message.origin === "assistant_answer";
              return (
                <article
                  className={`chat-message chat-message-${message.role} ${isNotice ? "chat-message-status" : ""}`}
                  key={message.id}
                >
                  <div className="chat-avatar">
                    {message.role === "assistant" ? (
                      <BookOpen size={16} />
                    ) : (
                      "You"
                    )}
                  </div>
                  <div className="chat-bubble">
                    <div className="chat-message-meta">
                      <strong>
                        {message.role === "assistant"
                          ? "Reference desk"
                          : "You"}
                      </strong>
                      {"sample" in message && <span>Illustrative</span>}
                      {isNotice && run?.status === "failed" && (
                        <span>Routing failed</span>
                      )}
                      {isNotice && run?.decision && (
                        <span>
                          {run.decision.route} · {run.decision.reason_code}
                        </span>
                      )}
                      {isAnswer && run?.outcome && (
                        <span className="answer-outcome">
                          {outcomeLabels[run.outcome]}
                        </span>
                      )}
                    </div>
                    <p>{message.content}</p>
                    {isAnswer && run && <Citations citations={run.citations} />}
                    {message.selected_manual_ids.length > 0 && (
                      <div className="chat-sources">
                        <div className="chat-source-label">
                          <FileText size={13} /> Selected scope{" "}
                          <span>{message.selected_manual_ids.length}</span>
                        </div>
                        <div>
                          {message.selected_manual_ids.map((id) => (
                            <button
                              type="button"
                              key={id}
                              className="chat-source-chip"
                              onClick={() =>
                                setSourceManual(
                                  manuals.find((manual) => manual.id === id) ??
                                    null,
                                )
                              }
                            >
                              {manualName(id)}
                              <ChevronRight size={13} />
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {message.role === "assistant" && (
                      <button
                        type="button"
                        className="chat-copy"
                        onClick={() => void copy(message)}
                      >
                        {copiedId === message.id ? (
                          <Check size={14} />
                        ) : (
                          <Clipboard size={14} />
                        )}
                        {copiedId === message.id
                          ? "Copied"
                          : isAnswer
                            ? "Copy answer"
                            : "Copy notice"}
                      </button>
                    )}
                  </div>
                </article>
              );
            })}
            {activePending && (
              <article className="chat-message chat-message-assistant chat-message-status">
                <div className="chat-avatar">
                  <BookOpen size={16} />
                </div>
                <div className="chat-bubble">
                  <div className="chat-message-meta">
                    <strong>Reference desk</strong>
                    <span>Searching the manuals...</span>
                  </div>
                  <p>
                    Searching the manuals... You can continue elsewhere while it
                    completes.
                  </p>
                </div>
              </article>
            )}
          </div>
          <div ref={bottomRef} />
        </div>
        <div className="chat-composer-area">
          <div className="chat-prompt-cards">
            {promptCards.map((prompt) => (
              <button
                type="button"
                key={prompt}
                onClick={() =>
                  setDrafts((current) => ({ ...current, [activeId]: prompt }))
                }
              >
                {prompt}
                <ArrowUp size={14} />
              </button>
            ))}
          </div>
          <div className="chat-composer">
            <textarea
              aria-label="Ask a reference question"
              value={activeDraft}
              placeholder="Ask a reference question…"
              onChange={(event) =>
                setDrafts((current) => ({
                  ...current,
                  [activeId]: event.target.value,
                }))
              }
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              rows={2}
            />
            <button
              type="button"
              className="chat-send"
              aria-label="Send question"
              disabled={!activeDraft.trim() || activePending}
              onClick={() => void send()}
            >
              <Send size={17} />
            </button>
            <div className="chat-composer-foot">
              <span>
                <span className="chat-status-dot" /> Local research backend
              </span>
              <span>Enter to send · Shift + Enter for new line</span>
            </div>
          </div>
        </div>
      </div>
      <aside
        className={`chat-reference ${sourceManual ? "is-open" : ""}`}
        aria-label="Reference details"
      >
        {sourceManual && (
          <>
            <div className="chat-reference-top">
              <div>
                <div className="chat-eyebrow">Source details</div>
                <h2>{sourceManual.label}</h2>
              </div>
              <button
                type="button"
                aria-label="Close source details"
                onClick={() => setSourceManual(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="chat-reference-file">
              <div className="chat-file-icon">
                <FileText size={22} />
              </div>
              <div>
                <strong>{sourceManual.title}</strong>
                <span>
                  {sourceManual.jurisdiction} · Original reference file
                </span>
              </div>
            </div>
            <div className="chat-reference-rule" />
            <div className="chat-reference-copy">
              <h3>Review status</h3>
              <p>
                Review cited text in the original source version and confirm its
                edition, applicability, and context for your project. A selected
                scope alone does not establish applicability.
              </p>
            </div>
            <a
              className="chat-open-manual"
              href={sourceManual.url}
              target="_blank"
              rel="noreferrer"
            >
              Open original PDF <ChevronRight size={16} />
            </a>
          </>
        )}
      </aside>
    </section>
  );
}
