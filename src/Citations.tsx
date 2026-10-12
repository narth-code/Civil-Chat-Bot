import type { Citation, Outcome } from "./api";
import { manuals } from "./manuals";
import "./citations.css";

export const outcomeLabels: Record<Outcome, string> = {
  kb_answer: "Answer from manuals",
  web_answer: "Answer from web search",
  clarification: "Needs clarification",
  blocked: "Blocked",
  insufficient: "Not enough information",
  direct: "Direct answer",
  error: "Error",
};

const outcomeKeys: Outcome[] = [
  "kb_answer",
  "web_answer",
  "clarification",
  "blocked",
  "insufficient",
  "direct",
  "error",
];
export { outcomeKeys };

function firstPage(pages?: string | null) {
  const page = pages?.match(/^\s*(\d+)(?:\s*-\s*\d+)?\s*$/)?.[1];
  return page && Number(page) > 0 ? page : null;
}

function webUrl(url?: string) {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:"
      ? parsed.href
      : null;
  } catch {
    return null;
  }
}

export function Citations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="answer-sources">
      <strong>Sources</strong>
      <ol>
        {citations.map((citation, index) => {
          if (citation.type === "web") {
            const url = webUrl(citation.url);
            return (
              <li key={index}>
                <span className="answer-source-kind">
                  Web search - not a registered source version
                </span>
                <span>{citation.title || citation.url || "Web evidence"}</span>
                {url ? (
                  <a href={url} target="_blank" rel="noopener noreferrer">
                    {citation.url}
                  </a>
                ) : (
                  citation.url && <span>{citation.url}</span>
                )}
              </li>
            );
          }
          const manual = manuals.find(
            (item) => item.id === citation.document_id,
          );
          const page = firstPage(citation.pages);
          const label = [
            manual?.title.replace(/^Austin, TX /, "") ??
              citation.title ??
              citation.document_id ??
              "Registered source",
            citation.identifier,
            citation.pages ? `pp. ${citation.pages}` : null,
          ]
            .filter(Boolean)
            .join(" - ");
          const pdfUrl =
            manual && page
              ? `/api/manuals/${encodeURIComponent(manual.id)}/pdf#page=${page}`
              : null;
          return (
            <li key={index}>
              {pdfUrl ? (
                <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
                  {label}
                </a>
              ) : (
                <span>{label}</span>
              )}
              {citation.quote && (
                <details>
                  <summary>Show cited text</summary>
                  <blockquote>{citation.quote}</blockquote>
                </details>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
