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

export interface Course {
  id: string;
  title: string;
  description: string;
  document_ids: string[];
  topic_count: number;
  created_at: string;
  updated_at: string;
}

export interface Topic {
  id: string;
  course_id: string;
  parent_id: string | null;
  title: string;
  description: string;
  position: number;
  state: "draft" | "approved";
  source_chunk_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface GenerationJob {
  id: string;
  course_id: string;
  topic_id: string | null;
  item_id: string | null;
  kind: "curriculum" | "study_items" | "study_item";
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  attempts: number;
  error_message: string | null;
}

export interface SourcePassage {
  chunk_id: string;
  document_id: string;
  filename: string;
  position: number;
  text: string;
}

export interface StudyItem {
  id: string;
  topic_id: string;
  item_type: "flashcard" | "multiple_choice" | "short_answer";
  prompt: string;
  answer: string;
  explanation: string;
  options: Array<{ text: string; correct: boolean }>;
  difficulty: number;
  approval_status: "draft" | "approved" | "archived";
  generation_version: string | null;
  sources: SourcePassage[];
  created_at: string;
  updated_at: string;
}

export type ReviewRating = "again" | "hard" | "good" | "easy";

export interface StudyQuestion {
  id: string;
  topic_id: string;
  item_type: StudyItem["item_type"];
  prompt: string;
  options: string[];
  difficulty: number;
  reason: string;
}

export interface ReviewQueueItem extends StudyQuestion {
  due_at: string | null;
  predicted_recall: number | null;
}

export interface ReviewQueue {
  course_id: string;
  generated_at: string;
  estimated_minutes: number;
  items: ReviewQueueItem[];
}

export interface StudySession {
  id: string;
  course_id: string;
  topic_id: string | null;
  mode: "daily_review" | "course" | "topic" | "selected";
  status: "active" | "completed";
  items: StudyQuestion[];
  answered_item_ids: string[];
  completed_at: string | null;
  created_at: string;
}

export interface MemoryState {
  study_item_id: string;
  half_life_days: number;
  difficulty: number;
  last_review_at: string;
  next_review_at: string;
  predicted_recall: number;
  successful_reviews: number;
  failed_reviews: number;
  scheduler_version: string;
}

export interface ReviewAttempt {
  id: string;
  session_id: string;
  study_item_id: string;
  submitted_answer: string;
  expected_answer: string;
  correct: boolean;
  score: number;
  rating: ReviewRating;
  confidence: number;
  response_time_ms: number;
  hint_used: boolean;
  grading_details: {
    correct_concepts: string[];
    missing_concepts: string[];
    mistaken_concepts: string[];
    rationale: string;
    supporting_chunk_ids: string[];
  };
  sources: SourcePassage[];
  reviewed_at: string;
  overridden_correct: boolean | null;
  overridden_score: number | null;
  override_reason: string | null;
  memory: MemoryState;
}

export interface ProgressSummary {
  coverage: number;
  mastery: number;
  predicted_retention: number;
  encountered_items: number;
  total_items: number;
  factors: Record<string, number | string>;
}

export interface TopicProgress extends ProgressSummary {
  topic_id: string;
  title: string;
}

export interface CourseProgress extends ProgressSummary {
  course_id: string;
  title: string;
  topics: TopicProgress[];
  weakest_topic_id: string | null;
  recommended_action: string;
}

export interface ProgressSnapshot extends ProgressSummary {
  id: string;
  captured_at: string;
}

export interface ItemProgress {
  study_item_id: string;
  prompt: string;
  mastery: number;
  factors: Record<string, number>;
  memory: MemoryState | null;
  observations: Array<{
    attempt_id: string;
    reviewed_at: string;
    score: number;
    correct: boolean;
    overridden: boolean;
  }>;
  predictions: Array<{ at: string; predicted_recall: number }>;
}

export interface StreamDone {
  conversation_id: string;
  message_id: string;
  citations: Citation[];
}
