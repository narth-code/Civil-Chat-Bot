import { useLayoutEffect, useState } from "react";
import {
  Building2,
  ChartNoAxesCombined,
  ChevronRight,
  FlaskConical,
  MessageSquareText,
  Moon,
  Sun,
} from "lucide-react";
import ChatWorkspace from "./ChatWorkspace";
import AdminWorkspace from "./AdminWorkspace";

type Theme = "light" | "dark";
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("civil-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* Storage may be disabled; the current session still works. */
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export default function App() {
  const [space, setSpace] = useState<"engineer" | "admin">("engineer");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }, [theme]);
  function toggleTheme() {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    try {
      localStorage.setItem("civil-theme", next);
    } catch {
      /* Session-only preference if storage is unavailable. */
    }
  }
  return (
    <div className="app-shell">
      <header className="app-header">
        <div
          className="app-brand"
          aria-label="Civil engineering reference workspace"
        >
          <span className="app-logo">
            <Building2 size={24} strokeWidth={1.8} />
          </span>
          <span className="app-brand-name">
            civil<span className="app-brand-dot">.</span>
          </span>
          <span className="app-brand-divider" />
          <span className="app-breadcrumb">
            Workspace <ChevronRight size={14} />{" "}
            <strong>
              {space === "engineer"
                ? "Engineering assistant"
                : "Admin analytics"}
            </strong>
          </span>
        </div>
        <div className="app-header-actions">
          <span className="app-development">
            <FlaskConical size={13} /> Development mode
          </span>
          <div
            className="app-space-switch"
            role="group"
            aria-label="Development workspace"
          >
            <button
              type="button"
              aria-pressed={space === "engineer"}
              onClick={() => setSpace("engineer")}
            >
              <MessageSquareText size={15} /> Engineer
            </button>
            <button
              type="button"
              aria-pressed={space === "admin"}
              onClick={() => setSpace("admin")}
            >
              <ChartNoAxesCombined size={15} /> Admin
            </button>
          </div>
          <span className="app-action-divider" />
          <button
            type="button"
            className="app-theme-toggle"
            onClick={toggleTheme}
            aria-label={
              theme === "light" ? "Switch to dark mode" : "Switch to light mode"
            }
            title={
              theme === "light" ? "Switch to dark mode" : "Switch to light mode"
            }
          >
            {theme === "light" ? <Moon size={19} /> : <Sun size={19} />}
          </button>
        </div>
      </header>
      <main
        className="app-main"
        aria-label={
          space === "engineer" ? "Engineer workspace" : "Admin workspace"
        }
      >
        <div className="workspace-frame" hidden={space !== "engineer"}>
          <ChatWorkspace />
        </div>
        <div className="workspace-frame" hidden={space !== "admin"}>
          <AdminWorkspace active={space === "admin"} />
        </div>
      </main>
    </div>
  );
}
