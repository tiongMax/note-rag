import type {
  Chunk,
  Conversation,
  ConversationDetail,
  Course,
  CourseProgress,
  Document,
  IngestionJob,
  ItemProgress,
  ProgressSnapshot,
  GenerationJob,
  StudyItem,
  ReviewAttempt,
  ReviewQueue,
  ReviewRating,
  StudySession,
  SourcePassage,
  Topic,
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
  courses: () => request<Course[]>("/courses"),
  createCourse: (title: string, description: string) =>
    request<Course>("/courses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, description }),
    }),
  updateCourse: (id: string, changes: Partial<Pick<Course, "title" | "description">>) =>
    request<Course>(`/courses/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    }),
  deleteCourse: (id: string) => request<void>(`/courses/${id}`, { method: "DELETE" }),
  attachDocuments: (courseId: string, documentIds: string[]) =>
    request<Course>(`/courses/${courseId}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: documentIds }),
    }),
  detachDocument: (courseId: string, documentId: string) =>
    request<void>(`/courses/${courseId}/documents/${documentId}`, { method: "DELETE" }),
  topics: (courseId: string) => request<Topic[]>(`/courses/${courseId}/topics`),
  topicSources: (courseId: string, topicId: string) => request<SourcePassage[]>(`/courses/${courseId}/topics/${topicId}/sources`),
  createTopic: (courseId: string, input: { title: string; parent_id?: string | null; position?: number; state?: Topic["state"] }) =>
    request<Topic>(`/courses/${courseId}/topics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  updateTopic: (courseId: string, topicId: string, changes: Partial<Pick<Topic, "title" | "description" | "parent_id" | "position" | "state">>) =>
    request<Topic>(`/courses/${courseId}/topics/${topicId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    }),
  deleteTopic: (courseId: string, topicId: string) => request<void>(`/courses/${courseId}/topics/${topicId}`, { method: "DELETE" }),
  generateCurriculum: (courseId: string) => request<GenerationJob>(`/courses/${courseId}/generate-curriculum`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  generationJob: (jobId: string) => request<GenerationJob>(`/generation-jobs/${jobId}`),
  generateStudyItems: (courseId: string, topicId: string) => request<GenerationJob>(`/courses/${courseId}/topics/${topicId}/generate-items`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  studyItems: (courseId: string, topicId: string) => request<StudyItem[]>(`/courses/${courseId}/topics/${topicId}/study-items`),
  updateStudyItem: (itemId: string, changes: Partial<Pick<StudyItem, "prompt" | "answer" | "explanation" | "difficulty" | "options" | "approval_status">>) => request<StudyItem>(`/study-items/${itemId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes) }),
  archiveStudyItem: (itemId: string) => request<void>(`/study-items/${itemId}`, { method: "DELETE" }),
  regenerateStudyItem: (itemId: string) => request<GenerationJob>(`/study-items/${itemId}/regenerate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  mergeTopic: (courseId: string, topicId: string, targetTopicId: string) => request<Topic>(`/courses/${courseId}/topics/${topicId}/merge`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target_topic_id: targetTopicId }) }),
  reviewQueue: (courseId: string, limit = 20) => request<ReviewQueue>(`/study/queue?course_id=${encodeURIComponent(courseId)}&limit=${limit}`),
  createStudySession: (input: { course_id: string; topic_id?: string; item_ids?: string[]; mode?: StudySession["mode"]; limit?: number }) => request<StudySession>("/study/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) }),
  studySession: (sessionId: string) => request<StudySession>(`/study/sessions/${sessionId}`),
  submitStudyAnswer: (sessionId: string, input: { item_id: string; submitted_answer: string; rating: ReviewRating; confidence: number; response_time_ms: number; hint_used: boolean; idempotency_key: string }) => request<ReviewAttempt>(`/study/sessions/${sessionId}/answers`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) }),
  completeStudySession: (sessionId: string) => request<StudySession>(`/study/sessions/${sessionId}/complete`, { method: "POST" }),
  overrideReview: (attemptId: string, input: { correct: boolean; score: number; reason: string }) => request<ReviewAttempt>(`/review-attempts/${attemptId}/override`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input) }),
  courseProgress: (courseId: string) => request<CourseProgress>(`/courses/${courseId}/progress`),
  courseProgressHistory: (courseId: string) => request<ProgressSnapshot[]>(`/courses/${courseId}/progress/history`),
  topicProgress: (topicId: string) => request<import("./types").TopicProgress>(`/topics/${topicId}/progress`),
  itemProgress: (itemId: string) => request<ItemProgress>(`/study-items/${itemId}/progress`),
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
