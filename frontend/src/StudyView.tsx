import {
  ArrowRight,
  BookOpenCheck,
  BrainCircuit,
  Check,
  Clock3,
  LoaderCircle,
  RotateCcw,
  Sparkles,
  Target,
  TrendingUp,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { errorText, fullDate } from "./App";
import { api } from "./api";
import type {
  Course,
  CourseProgress,
  ProgressSnapshot,
  ReviewAttempt,
  ReviewQueue,
  ReviewRating,
  StudyQuestion,
  StudySession,
} from "./types";

export const ACTIVE_SESSION_KEY = "note-rag-active-study-session";

export function StudyView({ notify }: { notify: (message: string) => void }) {
  const [courses, setCourses] = useState<Course[]>([]);
  const [courseId, setCourseId] = useState("");
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [progress, setProgress] = useState<CourseProgress | null>(null);
  const [history, setHistory] = useState<ProgressSnapshot[]>([]);
  const [session, setSession] = useState<StudySession | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDashboard = useCallback(async (id: string) => {
    const [nextQueue, nextProgress, nextHistory] = await Promise.all([
      api.reviewQueue(id),
      api.courseProgress(id),
      api.courseProgressHistory(id),
    ]);
    setQueue(nextQueue);
    setProgress(nextProgress);
    setHistory(nextHistory);
  }, []);

  useEffect(() => {
    const restore = async () => {
      try {
        const nextCourses = await api.courses();
        setCourses(nextCourses);
        const storedSessionId = localStorage.getItem(ACTIVE_SESSION_KEY);
        if (storedSessionId) {
          const active = await api.studySession(storedSessionId);
          if (active.status === "active") {
            setSession(active);
            setCourseId(active.course_id);
            return;
          }
          localStorage.removeItem(ACTIVE_SESSION_KEY);
        }
        if (nextCourses[0]) {
          setCourseId(nextCourses[0].id);
          await loadDashboard(nextCourses[0].id);
        }
      } catch (error) {
        notify(errorText(error));
      } finally {
        setLoading(false);
      }
    };
    void restore();
  }, [loadDashboard, notify]);

  const start = async () => {
    if (!courseId) return;
    try {
      setLoading(true);
      const active = await api.createStudySession({
        course_id: courseId,
        mode: "daily_review",
      });
      localStorage.setItem(ACTIVE_SESSION_KEY, active.id);
      setSession(active);
    } catch (error) {
      notify(errorText(error));
    } finally {
      setLoading(false);
    }
  };

  if (session) {
    return (
      <StudyPlayer
        session={session}
        notify={notify}
        update={setSession}
        close={async () => {
          localStorage.removeItem(ACTIVE_SESSION_KEY);
          setSession(null);
          await loadDashboard(session.course_id);
        }}
      />
    );
  }

  return (
    <div className="page study-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Adaptive review</span>
          <h1>Today</h1>
          <p>Review the right material before it fades.</p>
        </div>
        <select
          value={courseId}
          onChange={(event) => {
            const id = event.target.value;
            setCourseId(id);
            void loadDashboard(id).catch((error) => notify(errorText(error)));
          }}
        >
          {courses.map((course) => (
            <option key={course.id} value={course.id}>{course.title}</option>
          ))}
        </select>
      </header>
      {loading ? (
        <LoaderCircle className="spin centered" />
      ) : !courses.length ? (
        <section className="today-empty"><BookOpenCheck size={28} /><h2>Create a course first</h2><p>Today will fill with approved learning items.</p></section>
      ) : !queue?.items.length ? (
        <section className="today-empty"><Check size={28} /><h2>Nothing ready to study</h2><p>Approve study items in a course to build the review queue.</p></section>
      ) : (
        <>
          {progress && (
            <section className="today-progress">
              <ProgressMetric label="Coverage" value={progress.coverage} detail={`${progress.encountered_items}/${progress.total_items} items`} />
              <ProgressMetric label="Observed mastery" value={progress.mastery} detail="Recent graded performance" />
              <ProgressMetric label="Predicted retention" value={progress.predicted_retention} detail="Estimated recall now" predicted />
              <div className="today-recommendation"><Target size={18} /><span><small>Recommended next</small><strong>{progress.recommended_action}</strong></span>{progress.weakest_topic_id && <button className="secondary-button" onClick={async () => { try { const active = await api.createStudySession({ course_id: courseId, topic_id: progress.weakest_topic_id ?? undefined }); localStorage.setItem(ACTIVE_SESSION_KEY, active.id); setSession(active); } catch (error) { notify(errorText(error)); } }}>Practise</button>}</div>
              {history.length > 1 && <HistorySparkline history={history} />}
            </section>
          )}
          <section className="review-summary">
            <div><BrainCircuit size={22} /><span><strong>{queue.items.length}</strong> items queued</span></div>
            <div><Clock3 size={22} /><span><strong>{queue.estimated_minutes}</strong> estimated minutes</span></div>
            <button className="primary-button" onClick={() => void start()}><Sparkles size={15} /> Start daily review</button>
          </section>
          <section className="review-queue">
            <div className="section-heading"><div><h3>Review queue</h3><span>Ordered by urgency and predicted recall</span></div></div>
            {queue.items.map((item, index) => (
              <article key={item.id}>
                <span className="queue-position">{index + 1}</span>
                <div><strong>{item.prompt}</strong><small>{item.reason}</small></div>
                <span className="queue-type">{item.item_type.replace("_", " ")}</span>
              </article>
            ))}
          </section>
        </>
      )}
    </div>
  );
}

function ProgressMetric({ label, value, detail, predicted = false }: { label: string; value: number; detail: string; predicted?: boolean }) {
  return <div className="progress-metric"><span>{label}{predicted && <em>Prediction</em>}</span><strong>{Math.round(value * 100)}%</strong><div className="metric-track"><i style={{ width: `${value * 100}%` }} /></div><small>{detail}</small></div>;
}

function HistorySparkline({ history }: { history: ProgressSnapshot[] }) {
  const points = history.map((snapshot, index) => `${(index / Math.max(1, history.length - 1)) * 100},${34 - snapshot.mastery * 30}`).join(" ");
  const change = (history.at(-1)?.mastery ?? 0) - history[0].mastery;
  return <div className="history-sparkline"><span><TrendingUp size={13} /> Observed mastery {change >= 0 ? "+" : ""}{Math.round(change * 100)} pts</span><svg viewBox="0 0 100 38" preserveAspectRatio="none" role="img" aria-label="Observed mastery over completed reviews"><polyline points={points} /></svg></div>;
}

function StudyPlayer({ session, update, close, notify }: { session: StudySession; update: (session: StudySession) => void; close: () => Promise<void>; notify: (message: string) => void }) {
  const answered = new Set(session.answered_item_ids);
  const index = session.items.findIndex((item) => !answered.has(item.id));
  const question = index >= 0 ? session.items[index] : null;
  const [answer, setAnswer] = useState("");
  const [rating, setRating] = useState<ReviewRating>("good");
  const [confidence, setConfidence] = useState(3);
  const [hintUsed, setHintUsed] = useState(false);
  const [result, setResult] = useState<ReviewAttempt | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [startedAt, setStartedAt] = useState(Date.now());

  const progress = session.items.length - (index < 0 ? 0 : session.items.length - index);
  const choices = useMemo(() => question?.options ?? [], [question]);

  const submit = useCallback(async () => {
    if (!question || !answer.trim() || submitting) return;
    setSubmitting(true);
    try {
      const attempt = await api.submitStudyAnswer(session.id, {
        item_id: question.id,
        submitted_answer: answer.trim(),
        rating,
        confidence,
        response_time_ms: Date.now() - startedAt,
        hint_used: hintUsed,
        idempotency_key: crypto.randomUUID(),
      });
      setResult(attempt);
    } catch (error) {
      notify(errorText(error));
    } finally {
      setSubmitting(false);
    }
  }, [answer, confidence, hintUsed, notify, question, rating, session.id, startedAt, submitting]);

  const next = useCallback(async () => {
    if (!question || !result) return;
    const nextSession = await api.studySession(session.id);
    update(nextSession);
    setAnswer("");
    setResult(null);
    setHintUsed(false);
    setRating("good");
    setConfidence(3);
    setStartedAt(Date.now());
  }, [question, result, session.id, update]);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (event.key >= "1" && event.key <= "4") {
        setRating((["again", "hard", "good", "easy"] as ReviewRating[])[Number(event.key) - 1]);
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        if (result) void next(); else void submit();
      }
    };
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  }, [next, result, submit]);

  if (!question) {
    return <div className="page study-page"><section className="session-complete"><Check size={34} /><span className="eyebrow">Session complete</span><h1>{session.items.length} items reviewed</h1><p>Your next review dates have been updated from today’s results.</p><button className="primary-button" onClick={async () => { try { await api.completeStudySession(session.id); await close(); } catch (error) { notify(errorText(error)); } }}>Return to Today <ArrowRight size={15} /></button></section></div>;
  }

  return (
    <div className="study-player">
      <header><button className="secondary-button" onClick={() => void close()}>Exit</button><div><span>{progress + 1} of {session.items.length}</span><div className="progress-track"><i style={{ width: `${((progress + (result ? 1 : 0)) / session.items.length) * 100}%` }} /></div></div><small>{question.reason}</small></header>
      <main>
        <article className="question-card">
          <div className="item-meta"><span>{question.item_type.replace("_", " ")}</span><span>Difficulty {question.difficulty}/5</span></div>
          <h1>{question.prompt}</h1>
          {choices.length ? <div className="answer-options">{choices.map((choice) => <button key={choice} className={answer === choice ? "selected" : ""} disabled={Boolean(result)} onClick={() => setAnswer(choice)}>{choice}</button>)}</div> : <textarea value={answer} disabled={Boolean(result)} onChange={(event) => setAnswer(event.target.value)} placeholder="Type your answer…" rows={5} autoFocus />}
          {!result && <><div className="study-controls"><label>Confidence <select value={confidence} onChange={(event) => setConfidence(Number(event.target.value))}>{[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}/5</option>)}</select></label><label><input type="checkbox" checked={hintUsed} onChange={(event) => setHintUsed(event.target.checked)} /> I used a hint</label></div><div className="rating-row">{(["again", "hard", "good", "easy"] as ReviewRating[]).map((value, ratingIndex) => <button key={value} className={rating === value ? "selected" : ""} onClick={() => setRating(value)}><kbd>{ratingIndex + 1}</kbd>{value}</button>)}</div><button className="primary-button submit-answer" disabled={!answer.trim() || submitting} onClick={() => void submit()}>{submitting ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />} Check answer</button></>}
          {result && <Feedback result={result} notify={notify} update={setResult} />}
        </article>
        {result && <button className="primary-button next-question" onClick={() => void next()}>{index === session.items.length - 1 ? "Finish" : "Next item"} <ArrowRight size={15} /></button>}
      </main>
    </div>
  );
}

function Feedback({ result, update, notify }: { result: ReviewAttempt; update: (result: ReviewAttempt) => void; notify: (message: string) => void }) {
  const effectiveCorrect = result.overridden_correct ?? result.correct;
  return <section className={`answer-feedback ${effectiveCorrect ? "correct" : "incorrect"}`}><header><strong>{effectiveCorrect ? "Correct" : result.score > 0 ? "Partly correct" : "Needs review"}</strong><span>{Math.round((result.overridden_score ?? result.score) * 100)}% · next {fullDate.format(new Date(result.memory.next_review_at))}</span></header><div><h3>Expected answer</h3><p>{result.expected_answer}</p><p>{result.grading_details.rationale}</p>{result.grading_details.missing_concepts.length > 0 && <p><strong>Missing:</strong> {result.grading_details.missing_concepts.join(", ")}</p>}</div><RetentionCurve result={result} /><details><summary>{result.sources.length} cited source passage{result.sources.length === 1 ? "" : "s"}</summary>{result.sources.map((source) => <blockquote key={source.chunk_id}><strong>{source.filename} · passage {source.position + 1}</strong>{source.text}</blockquote>)}</details><button className="secondary-button" onClick={async () => { const reason = window.prompt("Why should this grade be overridden?"); if (!reason?.trim()) return; try { const changed = await api.overrideReview(result.id, { correct: !effectiveCorrect, score: effectiveCorrect ? 0 : 1, reason: reason.trim() }); update(changed); notify("Grade override saved; schedule recalculated."); } catch (error) { notify(errorText(error)); } }}><RotateCcw size={13} /> Override grade</button></section>;
}

function RetentionCurve({ result }: { result: ReviewAttempt }) {
  const points = Array.from({ length: 31 }, (_, day) => {
    const recall = 0.5 ** (day / result.memory.half_life_days);
    return `${(day / 30) * 100},${36 - recall * 32}`;
  }).join(" ");
  const reviewedScore = result.overridden_score ?? result.score;
  const intervalDays = Math.max(0, (new Date(result.memory.next_review_at).getTime() - new Date(result.reviewed_at).getTime()) / 86_400_000);
  const nextX = Math.min(100, (intervalDays / 30) * 100);
  return <figure className="retention-curve"><figcaption><span><i className="observed-dot" /> Observed result</span><span><i className="prediction-line" /> Predicted recall</span></figcaption><svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="Predicted forgetting curve with observed review result"><line className="review-marker" x1={nextX} x2={nextX} y1="2" y2="38" /><polyline className="prediction" points={points} /><circle className="observation" cx="0" cy={36 - reviewedScore * 32} r="2.2" /></svg><small>Prediction only · next review marker at {fullDate.format(new Date(result.memory.next_review_at))}</small></figure>;
}
