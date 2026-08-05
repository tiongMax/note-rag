import {
  ArrowLeft,
  Check,
  ChevronDown,
  Database,
  FileText,
  LoaderCircle,
  MessageSquare,
  Plus,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import { errorText } from "./App";
import { api, streamChat } from "./api";
import type {
  Chunk,
  Citation,
  Conversation,
  Document,
  Message,
} from "./types";

interface Props {
  documents: Document[];
  conversations: Conversation[];
  refreshConversations: () => Promise<void>;
  notify: (message: string) => void;
}

export function ChatView({
  documents,
  conversations,
  refreshConversations,
  notify,
}: Props) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [sending, setSending] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [filterOpen, setFilterOpen] = useState(false);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [citationChunk, setCitationChunk] = useState<Chunk | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const openConversation = async (id: string) => {
    try {
      const detail = await api.conversation(id);
      setConversationId(id);
      setMessages(detail.messages);
    } catch (error) {
      notify(errorText(error));
    }
  };

  const newConversation = () => {
    setConversationId(null);
    setMessages([]);
    setQuery("");
  };

  const send = async (event: FormEvent) => {
    event.preventDefault();
    const text = query.trim();
    if (!text || sending) return;
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      citations: [],
      created_at: new Date().toISOString(),
    };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      userMessage,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        created_at: new Date().toISOString(),
      },
    ]);
    setQuery("");
    setSending(true);
    let resolvedId = conversationId;
    try {
      await streamChat({
        query: text,
        conversationId,
        documentIds: selectedDocuments,
        onDelta: (delta) =>
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + delta }
                : message,
            ),
          ),
        onDone: (result) => {
          resolvedId = result.conversation_id;
          setConversationId(result.conversation_id);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    id: result.message_id,
                    citations: result.citations,
                  }
                : message,
            ),
          );
        },
      });
      await refreshConversations();
      if (resolvedId) {
        const detail = await api.conversation(resolvedId);
        setMessages(detail.messages);
      }
    } catch (error) {
      notify(errorText(error));
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId && !message.content
            ? {
                ...message,
                content: "I couldn’t complete that request. Please try again.",
              }
            : message,
        ),
      );
    } finally {
      setSending(false);
    }
  };

  const inspectCitation = async (item: Citation) => {
    setCitation(item);
    setCitationChunk(null);
    try {
      const chunks = await api.chunks(item.document_id);
      setCitationChunk(
        chunks.find((chunk) => chunk.id === item.chunk_id) ?? null,
      );
    } catch (error) {
      notify(errorText(error));
    }
  };

  return (
    <div className="chat-layout">
      <aside className="history-panel">
        <div className="history-title">
          <span>Conversations</span>
          <button className="icon-button" onClick={newConversation}>
            <Plus size={18} />
          </button>
        </div>
        <div className="history-list">
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              className={conversation.id === conversationId ? "active" : ""}
              onClick={() => void openConversation(conversation.id)}
            >
              <MessageSquare size={15} />
              <span>
                <strong>{conversation.title}</strong>
                <small>{conversation.message_count} messages</small>
              </span>
            </button>
          ))}
          {conversations.length === 0 && (
            <p className="history-empty">
              Your conversations will appear here.
            </p>
          )}
        </div>
      </aside>

      <section className="chat-main">
        <header className="chat-header">
          <div>
            <span className="eyebrow">Grounded assistant</span>
            <h1>{conversationId ? "Conversation" : "New conversation"}</h1>
          </div>
          <button className="secondary-button" onClick={newConversation}>
            <Plus size={16} /> New chat
          </button>
        </header>

        <div className={`messages ${messages.length ? "" : "welcome"}`}>
          {messages.length === 0 && (
            <div className="welcome-content">
              <span className="welcome-mark">
                <Sparkles size={25} />
              </span>
              <h2>What would you like to know?</h2>
              <p>
                Ask a question and I’ll answer using evidence from your
                knowledge base.
              </p>
              <div className="suggestions">
                <button
                  onClick={() =>
                    setQuery("Summarize the key themes in my documents.")
                  }
                >
                  Summarize the key themes
                </button>
                <button
                  onClick={() =>
                    setQuery("What are the most important facts to remember?")
                  }
                >
                  Find the important facts
                </button>
              </div>
            </div>
          )}
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar">
                {message.role === "assistant" ? (
                  <Sparkles size={16} />
                ) : (
                  "You"
                )}
              </div>
              <div className="message-body">
                <span>{message.role === "assistant" ? "Note RAG" : "You"}</span>
                <p
                  className={
                    sending &&
                    message.role === "assistant" &&
                    !message.content
                      ? "typing"
                      : ""
                  }
                >
                  {message.content ||
                    (sending ? "Thinking with your sources" : "")}
                </p>
                {message.citations.length > 0 && (
                  <div className="citations">
                    {message.citations.map((item) => (
                      <button
                        key={item.chunk_id}
                        onClick={() => void inspectCitation(item)}
                      >
                        <FileText size={14} />
                        <span>
                          [{item.citation_id}] {item.filename}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={send}>
            <textarea
              rows={1}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask a question about your documents…"
            />
            <div className="composer-tools">
              <div className="filter-control">
                <button
                  type="button"
                  className="source-filter"
                  onClick={() => setFilterOpen(!filterOpen)}
                >
                  <Database size={14} />
                  {selectedDocuments.length
                    ? `${selectedDocuments.length} selected`
                    : "All sources"}
                  <ChevronDown size={13} />
                </button>
                {filterOpen && (
                  <div className="source-menu">
                    <button
                      type="button"
                      className={
                        selectedDocuments.length === 0 ? "selected" : ""
                      }
                      onClick={() => setSelectedDocuments([])}
                    >
                      <span>All indexed sources</span>
                      {selectedDocuments.length === 0 && <Check size={15} />}
                    </button>
                    {documents
                      .filter(
                        (document) =>
                          document.indexing_status === "indexed",
                      )
                      .map((document) => {
                        const selected = selectedDocuments.includes(document.id);
                        return (
                          <button
                            type="button"
                            key={document.id}
                            className={selected ? "selected" : ""}
                            onClick={() =>
                              setSelectedDocuments((current) =>
                                selected
                                  ? current.filter((id) => id !== document.id)
                                  : [...current, document.id],
                              )
                            }
                          >
                            <span>{document.filename}</span>
                            {selected && <Check size={15} />}
                          </button>
                        );
                      })}
                  </div>
                )}
              </div>
              <span className="composer-hint">
                Enter to send · Shift + Enter for new line
              </span>
              <button
                className="send-button"
                type="submit"
                disabled={!query.trim() || sending}
              >
                {sending ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <Send size={17} />
                )}
              </button>
            </div>
          </form>
          <p>
            Answers come from your indexed sources. Verify important details.
          </p>
        </div>
      </section>

      {citation && (
        <aside className="citation-panel">
          <header>
            <button
              className="icon-button"
              onClick={() => setCitation(null)}
            >
              <ArrowLeft size={18} />
            </button>
            <div>
              <span>Source [{citation.citation_id}]</span>
              <strong>{citation.filename}</strong>
            </div>
            <button
              className="icon-button"
              onClick={() => setCitation(null)}
            >
              <X size={18} />
            </button>
          </header>
          <div className="citation-content">
            <div className="citation-meta">
              <span>Chunk {citation.position + 1}</span>
              {Object.entries(citation.source_metadata).map(([key, value]) => (
                <span key={key}>
                  {key}: {String(value)}
                </span>
              ))}
            </div>
            <h3>Referenced passage</h3>
            {citationChunk ? (
              <blockquote>{citationChunk.text}</blockquote>
            ) : (
              <LoaderCircle className="spin centered" />
            )}
          </div>
        </aside>
      )}
    </div>
  );
}
