import {
  BookOpen,
  Check,
  CircleAlert,
  FileText,
  LoaderCircle,
  MessageSquare,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  DragEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { errorText, formatNumber, fullDate, shortDate } from "./App";
import { api } from "./api";
import type { Chunk, Document, IngestionJob } from "./types";

interface Props {
  documents: Document[];
  loading: boolean;
  refresh: () => Promise<void>;
  notify: (message: string) => void;
  startChat: () => void;
}

export function StatusBadge({ status }: { status: string }) {
  const good = ["indexed", "completed", "ready"].includes(status);
  const failed = status === "failed";
  return (
    <span
      className={`badge ${good ? "success" : failed ? "failure" : "working"}`}
    >
      {good ? (
        <Check size={12} />
      ) : failed ? (
        <CircleAlert size={12} />
      ) : (
        <LoaderCircle className="spin" size={12} />
      )}
      {status}
    </span>
  );
}

export function DocumentsView({
  documents,
  loading,
  refresh,
  notify,
  startChat,
}: Props) {
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [selected, setSelected] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [menuId, setMenuId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(
    () =>
      documents.filter((document) =>
        document.filename.toLowerCase().includes(query.toLowerCase()),
      ),
    [documents, query],
  );

  useEffect(() => {
    const active = jobs.some(
      (job) => !["completed", "failed"].includes(job.status),
    );
    if (!active) return;
    const timer = window.setInterval(async () => {
      const next = await Promise.all(
        jobs.map((job) =>
          ["completed", "failed"].includes(job.status)
            ? job
            : api.job(job.id).catch(() => job),
        ),
      );
      setJobs(next);
      if (next.some((job) => job.status === "completed")) void refresh();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobs, refresh]);

  const uploadFiles = async (files: File[]) => {
    const accepted = files.filter((file) =>
      /\.(txt|md|markdown|pdf)$/i.test(file.name),
    );
    if (!accepted.length) {
      notify("Choose a TXT, Markdown, or PDF document.");
      return;
    }
    setUploading(true);
    try {
      for (const file of accepted) {
        const result = await api.upload(file);
        if (result.job_id) {
          const job = await api.job(result.job_id);
          setJobs((current) => [
            job,
            ...current.filter((item) => item.id !== job.id),
          ]);
        }
        notify(
          result.duplicate
            ? `${file.name} was already in the library.`
            : `${file.name} uploaded successfully.`,
        );
      }
      await refresh();
    } catch (error) {
      notify(errorText(error));
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    void uploadFiles(Array.from(event.dataTransfer.files));
  };

  const inspect = async (document: Document) => {
    setSelected(document);
    setChunksLoading(true);
    try {
      setChunks(await api.chunks(document.id));
    } catch (error) {
      notify(errorText(error));
    } finally {
      setChunksLoading(false);
    }
  };

  const reindex = async (document: Document) => {
    setBusyId(document.id);
    setMenuId(null);
    try {
      await api.reindex(document.id);
      notify(`${document.filename} was re-indexed.`);
      await refresh();
    } catch (error) {
      notify(errorText(error));
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (document: Document) => {
    setMenuId(null);
    if (!window.confirm(`Delete “${document.filename}” and all its chunks?`)) {
      return;
    }
    setBusyId(document.id);
    try {
      await api.deleteDocument(document.id);
      notify(`${document.filename} was deleted.`);
      await refresh();
    } catch (error) {
      notify(errorText(error));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Workspace</span>
          <h1>Knowledge base</h1>
          <p>Upload and manage the sources your assistant can draw from.</p>
        </div>
        <button
          className="primary-button"
          onClick={() => inputRef.current?.click()}
        >
          <Plus size={17} />
          Add document
        </button>
      </header>
      <input
        ref={inputRef}
        type="file"
        accept=".txt,.md,.markdown,.pdf"
        multiple
        hidden
        onChange={(event) =>
          void uploadFiles(Array.from(event.target.files ?? []))
        }
      />
      <section
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <span className="upload-icon">
          {uploading ? (
            <LoaderCircle className="spin" size={23} />
          ) : (
            <UploadCloud size={23} />
          )}
        </span>
        <div>
          <strong>
            {uploading ? "Processing upload…" : "Drop documents here"}
          </strong>
          <p>
            or{" "}
            <button onClick={() => inputRef.current?.click()}>
              browse files
            </button>
            {" · "}TXT, Markdown, PDF · 10 MB max
          </p>
        </div>
      </section>

      {jobs.length > 0 && (
        <section className="job-strip">
          {jobs.slice(0, 3).map((job) => (
            <div className="job-item" key={job.id}>
              {job.status === "completed" ? (
                <Check size={16} />
              ) : job.status === "failed" ? (
                <CircleAlert size={16} />
              ) : (
                <LoaderCircle className="spin" size={16} />
              )}
              <span>{job.status}</span>
              <div className="progress-track">
                <i style={{ width: `${job.progress}%` }} />
              </div>
              <small>{job.progress}%</small>
            </div>
          ))}
        </section>
      )}

      <section className="library-card">
        <div className="library-toolbar">
          <div>
            <h2>Documents</h2>
            <span>{documents.length} total</span>
          </div>
          <label className="search-box">
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search documents"
            />
          </label>
        </div>
        {loading ? (
          <Empty loading />
        ) : filtered.length === 0 ? (
          <Empty query={query} />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Status</th>
                  <th>Size</th>
                  <th>Added</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {filtered.map((document) => (
                  <tr key={document.id}>
                    <td>
                      <button
                        className="document-cell"
                        onClick={() => void inspect(document)}
                      >
                        <span
                          className={`file-icon ${
                            document.media_type.includes("pdf") ? "pdf" : ""
                          }`}
                        >
                          <FileText size={19} />
                        </span>
                        <span>
                          <strong>{document.filename}</strong>
                          <small>{document.media_type}</small>
                        </span>
                      </button>
                    </td>
                    <td>
                      <StatusBadge
                        status={
                          document.status === "failed"
                            ? "failed"
                            : document.indexing_status
                        }
                      />
                    </td>
                    <td>
                      <span className="muted">
                        {formatNumber.format(document.chunk_count)} chunks
                        <small>
                          {formatNumber.format(document.token_count)} tokens
                        </small>
                      </span>
                    </td>
                    <td className="muted">
                      {shortDate.format(new Date(document.created_at))}
                    </td>
                    <td className="actions-cell">
                      <button
                        className="icon-button"
                        onClick={() =>
                          setMenuId(menuId === document.id ? null : document.id)
                        }
                      >
                        {busyId === document.id ? (
                          <LoaderCircle className="spin" size={18} />
                        ) : (
                          <MoreHorizontal size={19} />
                        )}
                      </button>
                      {menuId === document.id && (
                        <div className="action-menu">
                          <button onClick={() => void inspect(document)}>
                            <FileText size={15} /> Inspect chunks
                          </button>
                          <button onClick={() => void reindex(document)}>
                            <RefreshCw size={15} /> Re-index
                          </button>
                          <button
                            className="danger"
                            onClick={() => void remove(document)}
                          >
                            <Trash2 size={15} /> Delete
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <footer className="card-footer">
          <span>
            Showing {filtered.length} of {documents.length}
          </span>
          <button onClick={startChat}>
            Ask your knowledge base <MessageSquare size={15} />
          </button>
        </footer>
      </section>

      {selected && (
        <div className="overlay" onMouseDown={() => setSelected(null)}>
          <aside
            className="drawer"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <span className="file-icon large">
                <FileText size={22} />
              </span>
              <div>
                <h2>{selected.filename}</h2>
                <p>
                  {formatNumber.format(selected.token_count)} tokens across{" "}
                  {selected.chunk_count} chunks
                </p>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelected(null)}
              >
                <X size={20} />
              </button>
            </header>
            {(selected.error_message || selected.indexing_error) && (
              <div className="error-panel">
                <CircleAlert size={17} />
                {selected.error_message || selected.indexing_error}
              </div>
            )}
            <div className="drawer-meta">
              <span>
                <small>Status</small>
                <StatusBadge status={selected.indexing_status} />
              </span>
              <span>
                <small>Embedding model</small>
                {selected.embedding_model ?? "Not indexed"}
              </span>
              <span>
                <small>Added</small>
                {fullDate.format(new Date(selected.created_at))}
              </span>
            </div>
            <div className="drawer-section-title">
              <h3>Extracted chunks</h3>
              <span>{chunks.length}</span>
            </div>
            <div className="chunk-list">
              {chunksLoading ? (
                <LoaderCircle className="spin centered" />
              ) : (
                chunks.map((chunk) => (
                  <article className="chunk-card" key={chunk.id}>
                    <div>
                      <span>Chunk {chunk.position + 1}</span>
                      <small>{chunk.token_count} tokens</small>
                    </div>
                    <p>{chunk.text}</p>
                    <code>{JSON.stringify(chunk.source_metadata)}</code>
                  </article>
                ))
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

function Empty({ loading, query }: { loading?: boolean; query?: string }) {
  return (
    <div className="empty-state">
      {loading ? (
        <LoaderCircle className="spin" />
      ) : (
        <BookOpen size={31} strokeWidth={1.5} />
      )}
      <h3>
        {loading
          ? "Loading your knowledge base"
          : query
            ? "No matching documents"
            : "Your library is empty"}
      </h3>
      <p>
        {query
          ? "Try a different filename."
          : "Upload a source to begin asking grounded questions."}
      </p>
    </div>
  );
}
