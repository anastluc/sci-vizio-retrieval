import { type DragEvent, type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError, api } from "../api/client";
import { type Collection, type DocumentItem, UNFILED } from "../api/types";
import { useMe } from "../auth/MeContext";
import { Banner, EmptyState, StatusBadge, UsageMeter } from "../components/ui";
import { formatBytes, formatDate, pluralize } from "../lib/format";
import { uploadPdf } from "../lib/upload";

interface UploadItem {
  key: string;
  name: string;
  progress: number;
  error?: string;
}

const POLL_MS = 3000;
const BUSY = new Set(["uploading", "queued", "processing"]);

export function Library() {
  const { me, refresh: refreshMe } = useMe();
  const [params, setParams] = useSearchParams();
  const selected = params.get("collection") ?? "";
  const [collections, setCollections] = useState<Collection[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [filterInput, setFilterInput] = useState("");
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [newCollection, setNewCollection] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadCollections = useCallback(() => api.collections().then(setCollections), []);
  const loadDocuments = useCallback(async () => {
    const page = await api.documents({ collection: selected || undefined, q: filter.trim() || undefined, limit: 200 });
    setDocuments(page.items);
    setTotal(page.total);
  }, [selected, filter]);

  useEffect(() => {
    setLoading(true);
    Promise.all([loadCollections(), loadDocuments()])
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [loadCollections, loadDocuments]);

  // Debounce the filename filter so typing doesn't send a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setFilter(filterInput), 250);
    return () => clearTimeout(timer);
  }, [filterInput]);

  // Poll while any document is still being processed.
  const anyBusy = documents.some((d) => BUSY.has(d.status));
  useEffect(() => {
    if (!anyBusy) return;
    const timer = setInterval(() => {
      loadDocuments().catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [anyBusy, loadDocuments]);

  const atLimit = !!me && me.usage.documents >= me.plan.max_documents;
  const currentCollection = collections.find((c) => c.id === selected);
  const uploadTarget = selected && selected !== UNFILED ? selected : null;

  async function handleFiles(files: FileList | File[]) {
    setError(null);
    const pdfs = Array.from(files).filter((f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length < Array.from(files).length) setError("Only PDF files can be uploaded.");

    for (const file of pdfs) {
      const key = `${file.name}-${crypto.randomUUID()}`;
      setUploads((u) => [...u, { key, name: file.name, progress: 0 }]);
      try {
        await uploadPdf(file, uploadTarget, (progress) =>
          setUploads((u) => u.map((x) => (x.key === key ? { ...x, progress } : x))));
        setUploads((u) => u.filter((x) => x.key !== key));
      } catch (e) {
        const message = e instanceof Error ? e.message : "Upload failed";
        setUploads((u) => u.map((x) => (x.key === key ? { ...x, error: message } : x)));
        if (e instanceof ApiError && e.code === "document_limit") break;
      } finally {
        await Promise.all([loadDocuments(), refreshMe(), loadCollections()]).catch(() => undefined);
      }
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (!atLimit) void handleFiles(e.dataTransfer.files);
  }

  async function act(action: () => Promise<unknown>) {
    setError(null);
    try {
      await action();
      await Promise.all([loadDocuments(), loadCollections(), refreshMe()]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    }
  }

  function onCreateCollection(e: FormEvent) {
    e.preventDefault();
    const name = newCollection.trim();
    if (!name) return;
    void act(async () => {
      const created = await api.createCollection(name);
      setNewCollection("");
      setParams({ collection: created.id });
    });
  }

  function select(collection: string) {
    setParams(collection ? { collection } : {});
  }

  async function download(doc: DocumentItem) {
    try {
      const { url } = await api.downloadUrl(doc.id);
      window.location.assign(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed");
    }
  }

  const heading = selected === UNFILED ? "Unfiled" : currentCollection?.name ?? "All documents";

  return (
    <div className="layout-sidebar">
      <aside className="sidebar" aria-label="Collections">
        <h2 className="sidebar-title">Library</h2>
        <nav className="sidebar-nav">
          <button className={!selected ? "active" : ""} onClick={() => select("")}>All documents</button>
          <button className={selected === UNFILED ? "active" : ""} onClick={() => select(UNFILED)}>Unfiled</button>
        </nav>
        <h3 className="sidebar-subtitle">Collections</h3>
        <nav className="sidebar-nav">
          {collections.map((c) => (
            <button key={c.id} className={selected === c.id ? "active" : ""} onClick={() => select(c.id)}>
              <span className="truncate">{c.name}</span>
              <span className="count">{c.document_count}</span>
            </button>
          ))}
        </nav>
        <form onSubmit={onCreateCollection} className="inline-form">
          <input placeholder="New collection" value={newCollection} maxLength={200}
            onChange={(e) => setNewCollection(e.target.value)} aria-label="New collection name" />
          <button className="btn btn-secondary btn-sm" disabled={!newCollection.trim()}>Add</button>
        </form>
        {me && (
          <div className="sidebar-usage">
            <UsageMeter label="PDFs" used={me.usage.documents} limit={me.plan.max_documents} />
            {me.plan.id === "free" && <Link to="/app/account" className="small">Upgrade for more PDFs</Link>}
          </div>
        )}
      </aside>

      <section className="content">
        <div className="content-header">
          <div>
            <h1>{heading}</h1>
            <p className="muted">{pluralize(total, "document")}</p>
          </div>
          {currentCollection && (
            <div className="actions">
              <button className="btn btn-ghost btn-sm" onClick={() => {
                const name = window.prompt("Rename collection", currentCollection.name)?.trim();
                if (name && name !== currentCollection.name) void act(() => api.renameCollection(currentCollection.id, name));
              }}>Rename</button>
              <button className="btn btn-ghost btn-sm danger" onClick={() => {
                if (window.confirm(`Delete “${currentCollection.name}”? Its documents are kept and become unfiled.`))
                  void act(async () => { await api.deleteCollection(currentCollection.id); select(""); });
              }}>Delete</button>
            </div>
          )}
        </div>

        {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}

        {atLimit ? (
          <Banner tone="info">
            You've reached the {me!.plan.name} plan limit of {me!.plan.max_documents} PDFs.{" "}
            {me!.plan.id === "free" ? <Link to="/app/account">Upgrade to Premium</Link> : "Delete some documents to add more."}
          </Banner>
        ) : (
          <div
            className={`dropzone ${dragging ? "dropzone-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInput.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInput.current?.click()}
            role="button"
            tabIndex={0}
          >
            <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple hidden
              onChange={(e) => { if (e.target.files) void handleFiles(e.target.files); e.target.value = ""; }} />
            <strong>Drop PDFs here</strong> or click to choose files
            {uploadTarget && currentCollection && <span className="muted"> — they'll go into “{currentCollection.name}”</span>}
            {me && <div className="muted small">Up to {me.plan.max_file_mb} MB each</div>}
          </div>
        )}

        {uploads.length > 0 && (
          <ul className="upload-list">
            {uploads.map((u) => (
              <li key={u.key}>
                <span className="truncate">{u.name}</span>
                {u.error ? (
                  <>
                    <span className="text-error small">{u.error}</span>
                    <button className="link small" onClick={() => setUploads((x) => x.filter((y) => y.key !== u.key))}>Dismiss</button>
                  </>
                ) : (
                  <progress value={u.progress} max={1} aria-label={`Uploading ${u.name}`} />
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="toolbar">
          <input type="search" placeholder="Filter by file name or title" value={filterInput}
            onChange={(e) => setFilterInput(e.target.value)} aria-label="Filter documents" />
        </div>

        {loading ? (
          <div className="page-center"><span className="spinner" aria-label="Loading" /></div>
        ) : documents.length === 0 ? (
          <EmptyState title={filter ? "No matching documents" : "No documents yet"}>
            {!filter && <p className="muted">Upload a PDF to get started. Figures become searchable once processing finishes.</p>}
          </EmptyState>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Document</th><th>Status</th><th>Figures</th><th>Size</th><th>Added</th><th>Collection</th><th /></tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id}>
                    <td className="doc-cell">
                      <div className="doc-title truncate" title={doc.title ?? doc.filename}>{doc.title ?? doc.filename}</div>
                      {doc.title && <div className="muted small truncate">{doc.filename}</div>}
                      {doc.error && <div className={`small ${doc.status === "failed" ? "text-error" : "muted"}`}>{doc.error}</div>}
                    </td>
                    <td><StatusBadge status={doc.status} /></td>
                    <td>
                      {doc.status === "ready" ? (
                        doc.figure_count > 0
                          ? <Link to={`/app/search?document=${doc.id}`}>{doc.figure_count}</Link>
                          : <span className="muted">0</span>
                      ) : "–"}
                    </td>
                    <td className="nowrap">{formatBytes(doc.size_bytes)}</td>
                    <td className="nowrap">{formatDate(doc.created_at)}</td>
                    <td>
                      <select value={doc.collection_id ?? ""} aria-label={`Collection for ${doc.filename}`}
                        onChange={(e) => void act(() => api.updateDocument(doc.id, { collection_id: e.target.value || null }))}>
                        <option value="">Unfiled</option>
                        {collections.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                      </select>
                    </td>
                    <td className="row-actions">
                      {doc.status === "failed" && (
                        <button className="btn btn-ghost btn-sm" onClick={() => void act(() => api.retryDocument(doc.id))}>Retry</button>
                      )}
                      {doc.status !== "uploading" && (
                        <button className="btn btn-ghost btn-sm" onClick={() => void download(doc)}>Download</button>
                      )}
                      <button className="btn btn-ghost btn-sm danger" onClick={() => {
                        if (window.confirm(`Delete “${doc.title ?? doc.filename}” and its figures?`))
                          void act(() => api.deleteDocument(doc.id));
                      }}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
