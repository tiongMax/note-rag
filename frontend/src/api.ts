import type {
  Chunk,
  Conversation,
  ConversationDetail,
  Document,
  IngestionJob,
  StreamDone,
  UploadResult,
} from "./types";

const API = "/api/v1";
const TOKEN_KEY = "note-rag-api-token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function setApiToken(token: string) {
  const normalized = token.trim();
  if (normalized) {
    sessionStorage.setItem(TOKEN_KEY, normalized);
  } else {
    sessionStorage.removeItem(TOKEN_KEY);
  }
}

function authenticatedHeaders(headers?: HeadersInit) {
  const resolved = new Headers(headers);
  const token = sessionStorage.getItem(TOKEN_KEY);
  if (token) resolved.set("Authorization", `Bearer ${token}`);
  return resolved;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: authenticatedHeaders(init?.headers),
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    let code: string | undefined;
    let requestId: string | undefined;
    try {
      const body = (await response.json()) as {
        detail?: string;
        error_message?: string | null;
        indexing_error?: string | null;
        error?: {
          code?: string;
          message?: string;
          request_id?: string;
        };
      };
      detail =
        body.error?.message ??
        body.detail ??
        body.indexing_error ??
        body.error_message ??
        detail;
      code = body.error?.code;
      requestId = body.error?.request_id;
    } catch {
      // The fallback includes the useful HTTP status.
    }
    if (response.status === 401) {
      window.dispatchEvent(new Event("note-rag:unauthorized"));
    }
    throw new ApiError(detail, response.status, code, requestId);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: async () => {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Backend unavailable");
    return response.json() as Promise<{ status: string }>;
  },
  documents: () => request<Document[]>("/documents"),
  chunks: (documentId: string) =>
    request<Chunk[]>(`/documents/${documentId}/chunks`),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResult>("/documents", { method: "POST", body: form });
  },
  reindex: (documentId: string) =>
    request(`/documents/${documentId}/index`, { method: "POST" }),
  job: (jobId: string) => request<IngestionJob>(`/ingestion-jobs/${jobId}`),
  deleteDocument: (documentId: string) =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),
  conversations: () => request<Conversation[]>("/conversations"),
  conversation: (id: string) =>
    request<ConversationDetail>(`/conversations/${id}`),
};

interface ChatOptions {
  query: string;
  conversationId: string | null;
  documentIds: string[];
  onDelta: (text: string) => void;
  onDone: (result: StreamDone) => void;
}

export async function streamChat(options: ChatOptions): Promise<void> {
  const response = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: authenticatedHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      query: options.query,
      conversation_id: options.conversationId,
      filters: { document_ids: options.documentIds },
    }),
  });
  if (!response.ok || !response.body) {
    if (response.status === 401) {
      window.dispatchEvent(new Event("note-rag:unauthorized"));
    }
    throw new ApiError(
      `Chat request failed (${response.status})`,
      response.status,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "";
  let dataLines: string[] = [];

  const dispatch = () => {
    if (!eventName || dataLines.length === 0) return;
    const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    if (eventName === "delta") {
      options.onDelta(String(payload.delta ?? payload.text ?? ""));
    } else if (eventName === "done") {
      options.onDone(payload as unknown as StreamDone);
    } else if (eventName === "error") {
      throw new Error(String(payload.detail ?? "Streaming failed"));
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split(/\r?\n/);
    buffer = done ? "" : (lines.pop() ?? "");
    for (const line of lines) {
      if (line === "") {
        dispatch();
        eventName = "";
        dataLines = [];
      } else if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (done) {
      dispatch();
      break;
    }
  }
}
