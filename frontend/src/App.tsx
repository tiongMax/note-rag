import {
  Database,
  Menu,
  MessageSquare,
  PanelLeftClose,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { ChatView } from "./ChatView";
import { DocumentsView } from "./DocumentsView";
import type { Conversation, Document } from "./types";

type View = "documents" | "chat";

export function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong";
}

export const formatNumber = new Intl.NumberFormat();
export const shortDate = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "numeric",
});
export const fullDate = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

function App() {
  const [view, setView] = useState<View>("documents");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toast, setToast] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.documents());
      setOnline(true);
    } catch (error) {
      setOnline(false);
      setToast(errorText(error));
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.conversations());
    } catch (error) {
      setToast(errorText(error));
    }
  }, []);

  useEffect(() => {
    void Promise.all([
      api.health().then(() => setOnline(true)).catch(() => setOnline(false)),
      refreshDocuments(),
      refreshConversations(),
    ]);
  }, [refreshConversations, refreshDocuments]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  return (
    <div className={`shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Sparkles size={17} />
          </span>
          <span>Note RAG</span>
        </div>
        <nav className="primary-nav" aria-label="Main navigation">
          <button
            className={view === "documents" ? "active" : ""}
            onClick={() => setView("documents")}
          >
            <Database size={18} />
            Knowledge
          </button>
          <button
            className={view === "chat" ? "active" : ""}
            onClick={() => setView("chat")}
          >
            <MessageSquare size={18} />
            Chat
          </button>
        </nav>
        <div className="sidebar-spacer" />
        <div className="service-status">
          <span className={`status-dot ${online ? "online" : ""}`} />
          <div>
            <strong>
              {online === null
                ? "Connecting"
                : online
                  ? "System online"
                  : "Offline"}
            </strong>
            <small>{documents.length} documents</small>
          </div>
        </div>
        <button
          className="collapse-button"
          onClick={() => setSidebarOpen(false)}
        >
          <PanelLeftClose size={17} />
          Collapse
        </button>
      </aside>
      {!sidebarOpen && (
        <button className="open-sidebar" onClick={() => setSidebarOpen(true)}>
          <Menu size={19} />
        </button>
      )}
      <main className="main">
        {view === "documents" ? (
          <DocumentsView
            documents={documents}
            loading={loading}
            refresh={refreshDocuments}
            notify={setToast}
            startChat={() => setView("chat")}
          />
        ) : (
          <ChatView
            documents={documents}
            conversations={conversations}
            refreshConversations={refreshConversations}
            notify={setToast}
          />
        )}
      </main>
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

export default App;
