export type DocumentStatus = "pending" | "ready" | "failed";
export type IndexingStatus = "pending" | "indexing" | "indexed" | "failed";

export interface Document {
  id: string;
  filename: string;
  media_type: string;
  status: DocumentStatus;
  token_count: number;
  chunk_count: number;
  error_message: string | null;
  indexing_status: IndexingStatus;
  embedding_model: string | null;
  indexed_at: string | null;
  indexing_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface Chunk {
  id: string;
  document_id: string;
  position: number;
  text: string;
  token_count: number;
  token_start: number;
  token_end: number;
  char_start: number;
  char_end: number;
  source_metadata: Record<string, unknown>;
}

export interface Citation {
  citation_id: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  position: number;
  source_metadata: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface UploadResult {
  document_id: string;
  job_id: string | null;
  status: DocumentStatus;
  duplicate: boolean;
  chunk_count: number;
  token_count: number;
  error_message: string | null;
  indexing_status: IndexingStatus;
  indexing_error: string | null;
}

export interface IngestionJob {
  id: string;
  document_id: string;
  status:
    | "queued"
    | "parsing"
    | "chunking"
    | "embedding"
    | "indexing"
    | "completed"
    | "failed";
  progress: number;
  attempts: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface StreamDone {
  conversation_id: string;
  message_id: string;
  citations: Citation[];
}
