import { Archive, BookOpen, Check, ChevronDown, ChevronRight, FilePlus2, FolderOpen, LoaderCircle, Pencil, Plus, RotateCcw, Save, Sparkles, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { errorText } from "./App";
import { api } from "./api";
import type { Course, Document, GenerationJob, SourcePassage, StudyItem, Topic } from "./types";

interface Props {
  documents: Document[];
  notify: (message: string) => void;
}

export function CoursesView({ documents, notify }: Props) {
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [reviewTopic, setReviewTopic] = useState<Topic | null>(null);
  const selected = courses.find((course) => course.id === selectedId) ?? null;

  const refreshCourses = async (preferId?: string) => {
    const next = await api.courses();
    setCourses(next);
    setSelectedId((current) => preferId ?? current ?? next[0]?.id ?? null);
  };

  const refreshTopics = async (courseId: string) => setTopics(await api.topics(courseId));

  useEffect(() => {
    refreshCourses().catch((error) => notify(errorText(error))).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedId) { setTopics([]); return; }
    refreshTopics(selectedId).catch((error) => notify(errorText(error)));
  }, [selectedId]);

  const run = async (work: () => Promise<void>) => {
    try { await work(); } catch (error) { notify(errorText(error)); }
  };

  if (loading) return <LoaderCircle className="spin centered" />;

  return (
    <div className="page courses-page">
      <header className="page-header">
        <div><span className="eyebrow">Learning workspace</span><h1>Courses</h1><p>Organize source material into an editable curriculum.</p></div>
        <button className="primary-button" onClick={() => setCreating(true)}><Plus size={17} /> New course</button>
      </header>
      {courses.length === 0 ? (
        <section className="course-empty"><BookOpen size={30} /><h2>Build your first course</h2><p>Group existing documents, then map the topics you want to master.</p><button className="primary-button" onClick={() => setCreating(true)}><Plus size={16} /> Create course</button></section>
      ) : (
        <div className="courses-layout">
          <aside className="course-list">
            {courses.map((course) => <button key={course.id} className={course.id === selectedId ? "active" : ""} onClick={() => setSelectedId(course.id)}><FolderOpen size={17} /><span><strong>{course.title}</strong><small>{course.document_ids.length} sources · {course.topic_count} topics</small></span></button>)}
          </aside>
          {selected && <section className="course-detail">
            <header><div><span className="eyebrow">Course</span><h2>{selected.title}</h2><p>{selected.description || "No description yet."}</p></div><div className="course-actions"><button className="secondary-button" onClick={() => setAttaching(true)}><FilePlus2 size={15} /> Sources</button><button className="icon-button danger" title="Delete course" onClick={() => run(async () => { if (!window.confirm(`Delete “${selected.title}”? Its source documents will be kept.`)) return; await api.deleteCourse(selected.id); await refreshCourses(); notify("Course deleted; source documents were kept."); })}><Trash2 size={16} /></button></div></header>
            <CourseSources course={selected} documents={documents} onDetach={(documentId) => run(async () => { await api.detachDocument(selected.id, documentId); await refreshCourses(selected.id); notify("Source detached."); })} />
            <LearningMaterials course={selected} topics={topics} reload={async () => { await refreshTopics(selected.id); await refreshCourses(selected.id); }} notify={notify} openItems={setReviewTopic} />
            <Curriculum course={selected} topics={topics} reload={async () => { await refreshTopics(selected.id); await refreshCourses(selected.id); }} notify={notify} />
          </section>}
        </div>
      )}
      {creating && <CreateCourse close={() => setCreating(false)} submit={(title, description) => run(async () => { const course = await api.createCourse(title, description); await refreshCourses(course.id); setCreating(false); notify("Course created."); })} />}
      {attaching && selected && <AttachDocuments course={selected} documents={documents} close={() => setAttaching(false)} submit={(ids) => run(async () => { await api.attachDocuments(selected.id, ids); await refreshCourses(selected.id); setAttaching(false); notify("Sources attached."); })} />}
      {reviewTopic && selected && <StudyItemEditor course={selected} topic={reviewTopic} close={() => setReviewTopic(null)} notify={notify} />}
    </div>
  );
}

function CourseSources({ course, documents, onDetach }: { course: Course; documents: Document[]; onDetach: (id: string) => void }) {
  const attached = documents.filter((document) => course.document_ids.includes(document.id));
  return <section className="course-section"><div className="section-heading"><div><h3>Source documents</h3><span>{attached.length}</span></div></div>{attached.length ? <div className="source-chips">{attached.map((document) => <span key={document.id}>{document.filename}<button title="Detach source" onClick={() => onDetach(document.id)}><X size={13} /></button></span>)}</div> : <p className="inline-empty">No source documents attached yet.</p>}</section>;
}

function Curriculum({ course, topics, reload, notify }: { course: Course; topics: Topic[]; reload: () => Promise<void>; notify: (message: string) => void }) {
  const [addingParent, setAddingParent] = useState<string | null | undefined>(undefined);
  const roots = useMemo(() => topics.filter((topic) => topic.parent_id === null).sort((a, b) => a.position - b.position), [topics]);
  const mutate = async (work: () => Promise<unknown>) => { try { await work(); await reload(); } catch (error) { notify(errorText(error)); } };
  return <section className="course-section curriculum"><div className="section-heading"><div><h3>Curriculum</h3><span>{topics.length} topics</span></div><button className="secondary-button" onClick={() => setAddingParent(null)}><Plus size={14} /> Topic</button></div>{roots.length ? <div className="topic-tree">{roots.map((topic, index) => <TopicRow key={topic.id} topic={topic} siblings={roots} index={index} all={topics} addChild={() => setAddingParent(topic.id)} update={(changes) => mutate(() => api.updateTopic(course.id, topic.id, changes))} remove={() => mutate(async () => { if (window.confirm(`Delete “${topic.title}” and its subtopics?`)) await api.deleteTopic(course.id, topic.id); })} courseId={course.id} reload={reload} notify={notify} />)}</div> : <p className="inline-empty">No topics yet. Add the first learning objective.</p>}{addingParent !== undefined && <TopicForm parentId={addingParent} close={() => setAddingParent(undefined)} submit={(title) => mutate(async () => { await api.createTopic(course.id, { title, parent_id: addingParent }); setAddingParent(undefined); })} />}</section>;
}

function TopicRow({ topic, siblings, index, all, addChild, update, remove, courseId, reload, notify }: { topic: Topic; siblings: Topic[]; index: number; all: Topic[]; addChild: () => void; update: (changes: Partial<Topic>) => void; remove: () => void; courseId: string; reload: () => Promise<void>; notify: (message: string) => void }) {
  const [open, setOpen] = useState(true); const [editing, setEditing] = useState(false);
  const children = all.filter((item) => item.parent_id === topic.id).sort((a, b) => a.position - b.position);
  const mutateChild = async (child: Topic, changes: Partial<Topic>) => { try { await api.updateTopic(courseId, child.id, changes); await reload(); } catch (error) { notify(errorText(error)); } };
  return <div className="topic-branch"><div className="topic-row"><button className="tree-toggle" onClick={() => setOpen(!open)} disabled={!children.length}>{children.length ? (open ? <ChevronDown size={14} /> : <ChevronRight size={14} />) : <span />}</button>{editing ? <input autoFocus defaultValue={topic.title} onBlur={(event) => { const title = event.target.value.trim(); if (title && title !== topic.title) update({ title }); setEditing(false); }} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} /> : <strong>{topic.title}</strong>}<button className={`topic-state ${topic.state}`} onClick={() => update({ state: topic.state === "draft" ? "approved" : "draft" })}>{topic.state === "approved" && <Check size={11} />}{topic.state}</button><div className="topic-tools"><button title="Move up" disabled={index === 0} onClick={() => update({ position: index - 1 })}>↑</button><button title="Move down" disabled={index === siblings.length - 1} onClick={() => update({ position: index + 1 })}>↓</button><button title="Rename" onClick={() => setEditing(true)}><Pencil size={13} /></button><button title="Add subtopic" onClick={addChild}><Plus size={14} /></button><button title="Delete" className="danger" onClick={remove}><Trash2 size={13} /></button></div></div>{open && children.length > 0 && <div className="topic-children">{children.map((child, childIndex) => <TopicRow key={child.id} topic={child} siblings={children} index={childIndex} all={all} addChild={() => { const title = window.prompt("Subtopic title"); if (title?.trim()) void api.createTopic(courseId, { title: title.trim(), parent_id: child.id }).then(reload).catch((error) => notify(errorText(error))); }} update={(changes) => void mutateChild(child, changes)} remove={() => { if (window.confirm(`Delete “${child.title}” and its subtopics?`)) void api.deleteTopic(courseId, child.id).then(reload).catch((error) => notify(errorText(error))); }} courseId={courseId} reload={reload} notify={notify} />)}</div>}</div>;
}

async function waitForJob(initial: GenerationJob): Promise<GenerationJob> {
  let job = initial;
  for (let attempt = 0; attempt < 120 && !["completed", "failed"].includes(job.status); attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    job = await api.generationJob(job.id);
  }
  return job;
}

function LearningMaterials({ course, topics, reload, notify, openItems }: { course: Course; topics: Topic[]; reload: () => Promise<void>; notify: (message: string) => void; openItems: (topic: Topic) => void }) {
  const [job, setJob] = useState<GenerationJob | null>(null);
  const generate = async () => {
    try {
      const initial = await api.generateCurriculum(course.id);
      setJob(initial);
      const result = await waitForJob(initial);
      setJob(result);
      if (result.status === "failed") throw new Error(result.error_message ?? "Generation failed");
      await reload();
      notify("Draft curriculum generated. Review it before approval.");
    } catch (error) { notify(errorText(error)); }
  };
  return <section className="course-section generation-panel"><div className="section-heading"><div><h3>AI learning materials</h3><span>{topics.filter((topic) => topic.state === "draft").length} drafts · {topics.filter((topic) => topic.state === "approved").length} approved</span></div><button className="primary-button compact" disabled={!course.document_ids.length || (job !== null && !["completed", "failed"].includes(job.status))} onClick={() => void generate()}>{job && !["completed", "failed"].includes(job.status) ? <LoaderCircle className="spin" size={14} /> : <Sparkles size={14} />} Generate curriculum</button></div>{job && <div className={`generation-status ${job.status}`}><span>{job.status === "failed" ? job.error_message : `Generation ${job.status}`}</span><div className="progress-track"><i style={{ width: `${job.progress}%` }} /></div></div>}<div className="review-topic-list">{topics.map((topic) => <button key={topic.id} onClick={() => openItems(topic)}><span><strong>{topic.title}</strong><small>{topic.state} · inspect and author study items</small></span><ChevronRight size={15} /></button>)}</div></section>;
}

function StudyItemEditor({ course, topic, close, notify }: { course: Course; topic: Topic; close: () => void; notify: (message: string) => void }) {
  const [items, setItems] = useState<StudyItem[]>([]);
  const [sources, setSources] = useState<SourcePassage[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const load = async () => {
    const [nextItems, nextSources] = await Promise.all([
      api.studyItems(course.id, topic.id),
      api.topicSources(course.id, topic.id),
    ]);
    setItems(nextItems);
    setSources(nextSources);
  };
  useEffect(() => { load().catch((error) => notify(errorText(error))).finally(() => setLoading(false)); }, [topic.id]);
  const generate = async () => { setGenerating(true); try { const result = await waitForJob(await api.generateStudyItems(course.id, topic.id)); if (result.status === "failed") throw new Error(result.error_message ?? "Generation failed"); await load(); notify("Draft study items generated."); } catch (error) { notify(errorText(error)); } finally { setGenerating(false); } };
  return <div className="overlay review-overlay"><section className="review-drawer"><header><div><span className="eyebrow">Study-item review</span><h2>{topic.title}</h2><p>Generated content remains draft until you approve it.</p></div><button className="icon-button" onClick={close}><X size={18} /></button></header><div className="review-toolbar"><span>{items.length} active items · {sources.length} source passages</span><button className="primary-button compact" disabled={generating || !topic.source_chunk_ids.length} onClick={() => void generate()}>{generating ? <LoaderCircle className="spin" size={14} /> : <Sparkles size={14} />} Generate items</button></div>{sources.length > 0 && <details className="topic-source-review"><summary>Inspect curriculum evidence</summary>{sources.map((source) => <blockquote key={source.chunk_id}><strong>{source.filename} · passage {source.position + 1}</strong>{source.text}</blockquote>)}</details>}{loading ? <LoaderCircle className="spin centered" /> : items.length ? <div className="study-item-list">{items.map((item) => <StudyItemCard key={item.id} item={item} reload={load} notify={notify} />)}</div> : <p className="review-empty">Generate cited flashcards and questions after reviewing this topic's supporting passages.</p>}</section></div>;
}

function StudyItemCard({ item, reload, notify }: { item: StudyItem; reload: () => Promise<void>; notify: (message: string) => void }) {
  const [prompt, setPrompt] = useState(item.prompt);
  const [answer, setAnswer] = useState(item.answer);
  const [explanation, setExplanation] = useState(item.explanation);
  const unsupported = answer.length > 3 && !item.sources.some((source) => source.text.toLocaleLowerCase().includes(answer.toLocaleLowerCase()));
  const act = async (work: () => Promise<unknown>, message: string) => { try { await work(); await reload(); notify(message); } catch (error) { notify(errorText(error)); } };
  return <article className="study-item-card"><div className="item-meta"><span>{item.item_type.replace("_", " ")}</span><span className={`topic-state ${item.approval_status}`}>{item.approval_status}</span><span>Difficulty {item.difficulty}/5</span></div><label>Prompt<textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={2} /></label>{item.options.length > 0 && <div className="option-preview">{item.options.map((option) => <span key={option.text} className={option.correct ? "correct" : ""}>{option.correct && <Check size={11} />}{option.text}</span>)}</div>}<label>Answer<textarea value={answer} onChange={(event) => setAnswer(event.target.value)} rows={2} /></label>{unsupported && <p className="support-warning">Edited answer text does not appear verbatim in the cited passages. Recheck its support before approval.</p>}<label>Explanation<textarea value={explanation} onChange={(event) => setExplanation(event.target.value)} rows={3} /></label><details><summary>{item.sources.length} supporting passage{item.sources.length === 1 ? "" : "s"}</summary>{item.sources.map((source) => <blockquote key={source.chunk_id}><strong>{source.filename} · passage {source.position + 1}</strong>{source.text}</blockquote>)}</details><div className="item-actions"><button className="secondary-button" onClick={() => void act(() => api.updateStudyItem(item.id, { prompt, answer, explanation }), "Changes saved.")}><Save size={13} /> Save</button>{item.approval_status !== "approved" && <button className="primary-button compact" onClick={() => void act(() => api.updateStudyItem(item.id, { prompt, answer, explanation, approval_status: "approved" }), "Study item approved.")}><Check size={13} /> Approve</button>}<button className="secondary-button" onClick={() => void act(async () => { const result = await waitForJob(await api.regenerateStudyItem(item.id)); if (result.status === "failed") throw new Error(result.error_message ?? "Regeneration failed"); }, "Replacement draft generated.")}><RotateCcw size={13} /> Regenerate</button><button className="icon-button danger" title="Archive" onClick={() => void act(() => api.archiveStudyItem(item.id), "Study item archived.")}><Archive size={14} /></button></div></article>;
}

function CreateCourse({ close, submit }: { close: () => void; submit: (title: string, description: string) => void }) { const [title, setTitle] = useState(""); const [description, setDescription] = useState(""); return <Modal title="Create course" close={close}><form onSubmit={(event) => { event.preventDefault(); if (title.trim()) submit(title.trim(), description.trim()); }}><label>Course title<input value={title} onChange={(event) => setTitle(event.target.value)} autoFocus maxLength={255} /></label><label>Description<textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={3} /></label><button className="primary-button" disabled={!title.trim()}>Create course</button></form></Modal>; }
function TopicForm({ parentId, close, submit }: { parentId: string | null; close: () => void; submit: (title: string) => void }) { const [title, setTitle] = useState(""); return <Modal title={parentId ? "Add subtopic" : "Add topic"} close={close}><form onSubmit={(event) => { event.preventDefault(); if (title.trim()) submit(title.trim()); }}><label>Topic title<input value={title} onChange={(event) => setTitle(event.target.value)} autoFocus /></label><button className="primary-button" disabled={!title.trim()}>Add topic</button></form></Modal>; }
function AttachDocuments({ course, documents, close, submit }: { course: Course; documents: Document[]; close: () => void; submit: (ids: string[]) => void }) { const available = documents.filter((document) => !course.document_ids.includes(document.id)); const [ids, setIds] = useState<string[]>([]); return <Modal title="Attach source documents" close={close}>{available.length ? <form onSubmit={(event) => { event.preventDefault(); submit(ids); }}><div className="document-picker">{available.map((document) => <label key={document.id}><input type="checkbox" checked={ids.includes(document.id)} onChange={() => setIds((current) => current.includes(document.id) ? current.filter((id) => id !== document.id) : [...current, document.id])} /><span>{document.filename}</span></label>)}</div><button className="primary-button" disabled={!ids.length}>Attach selected</button></form> : <p className="inline-empty">Every document is already attached.</p>}</Modal>; }
function Modal({ title, close, children }: { title: string; close: () => void; children: React.ReactNode }) { return <div className="overlay modal-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}><section className="modal-card"><header><h2>{title}</h2><button className="icon-button" onClick={close}><X size={17} /></button></header>{children}</section></div>; }
