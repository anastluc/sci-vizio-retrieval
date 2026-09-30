import { type FormEvent, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api } from "../api/client";
import type { SearchHit, SearchResponse } from "../api/types";
import { Banner, EmptyState, FacetGroup } from "../components/ui";
import { pluralize } from "../lib/format";

const PAGE_SIZE = 24;

/** Read the search state from the URL so searches can be shared and navigated with back/forward. */
function useSearchState() {
  const [params, setParams] = useSearchParams();
  const state = {
    query: params.get("q") ?? "",
    figureTypes: params.getAll("type"),
    collections: params.getAll("collection"),
    documents: params.getAll("document"),
    includeDecorative: params.get("decorative") === "1",
  };

  function update(changes: Partial<typeof state>) {
    const next = { ...state, ...changes };
    const p = new URLSearchParams();
    if (next.query) p.set("q", next.query);
    next.figureTypes.forEach((v) => p.append("type", v));
    next.collections.forEach((v) => p.append("collection", v));
    next.documents.forEach((v) => p.append("document", v));
    if (next.includeDecorative) p.set("decorative", "1");
    setParams(p);
  }
  return [state, update] as const;
}

function toggle(list: string[], value: string) {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

export function Search() {
  const [state, update] = useSearchState();
  const [input, setInput] = useState(state.query);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<SearchHit | null>(null);
  const requestId = useRef(0);

  const body = {
    query: state.query,
    figure_types: state.figureTypes,
    collections: state.collections,
    documents: state.documents,
    include_decorative: state.includeDecorative,
  };
  const key = JSON.stringify(body);

  useEffect(() => setInput(state.query), [state.query]);

  useEffect(() => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    api.search({ ...JSON.parse(key), limit: PAGE_SIZE, offset: 0 })
      .then((res) => {
        if (id !== requestId.current) return; // a newer search superseded this one
        setData(res);
        setResults(res.results);
      })
      .catch((e) => id === requestId.current && setError(e.message))
      .finally(() => id === requestId.current && setLoading(false));
  }, [key]);

  async function loadMore() {
    setLoading(true);
    try {
      const res = await api.search({ ...body, limit: PAGE_SIZE, offset: results.length });
      setResults((r) => [...r, ...res.results]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    update({ query: input.trim() });
  }

  const hasFilters = state.figureTypes.length + state.collections.length + state.documents.length > 0;

  return (
    <div className="layout-sidebar">
      <aside className="sidebar" aria-label="Filters">
        <h2 className="sidebar-title">Filters</h2>
        {data && (
          <>
            <FacetGroup title="Figure type" values={data.facets.figure_types} selected={state.figureTypes}
              onToggle={(v) => update({ figureTypes: toggle(state.figureTypes, v) })} />
            <label className="facet-option toggle">
              <input type="checkbox" checked={state.includeDecorative}
                onChange={(e) => update({ includeDecorative: e.target.checked })} />
              <span>Include decorative images (logos, icons…)</span>
            </label>
            <FacetGroup title="Collection" values={data.facets.collections} selected={state.collections}
              onToggle={(v) => update({ collections: toggle(state.collections, v) })} />
            <FacetGroup title="Document" values={data.facets.documents} selected={state.documents}
              onToggle={(v) => update({ documents: toggle(state.documents, v) })} />
          </>
        )}
        {hasFilters && (
          <button className="btn btn-ghost btn-sm" onClick={() => update({ figureTypes: [], collections: [], documents: [] })}>
            Clear filters
          </button>
        )}
      </aside>

      <section className="content">
        <form className="search-bar" onSubmit={onSubmit} role="search">
          <input type="search" value={input} onChange={(e) => setInput(e.target.value)} maxLength={500}
            placeholder="Describe the figure, e.g. “line chart of training loss”" aria-label="Search figures" />
          <button className="btn btn-primary" type="submit">Search</button>
        </form>

        {error && <Banner tone="error" onClose={() => setError(null)}>{error}</Banner>}

        {data && (
          <p className="muted result-summary">
            {state.query ? `${pluralize(data.total, "figure")} ranked by relevance to “${state.query}”`
              : `${pluralize(data.total, "figure")}, newest papers first`}
          </p>
        )}

        {data && data.total === 0 && !loading ? (
          <EmptyState title="No figures found">
            <p className="muted">
              {hasFilters ? "Try removing some filters." : <>Upload PDFs in your <Link to="/app/library">library</Link>; their figures appear here once processed.</>}
            </p>
          </EmptyState>
        ) : (
          <div className="figure-grid" aria-busy={loading} style={loading ? { opacity: 0.6 } : undefined}>
            {results.map((hit) => (
              <button key={hit.id} className="figure-card" onClick={() => setOpen(hit)}>
                <div className="figure-thumb"><img src={hit.image_url} alt={hit.title ?? hit.figure_type} loading="lazy" /></div>
                <div className="figure-meta">
                  <span className="type-chip">{hit.figure_type}</span>
                  <div className="figure-title">{hit.title || "Untitled figure"}</div>
                  <div className="figure-caption">Page {hit.page} of {hit.document.title ?? hit.document.filename}</div>
                </div>
              </button>
            ))}
          </div>
        )}

        {loading && results.length === 0 && <div className="page-center"><span className="spinner" aria-label="Loading" /></div>}
        {data && !loading && results.length < data.total && (
          <div className="center"><button className="btn btn-secondary" onClick={() => void loadMore()}>Load more</button></div>
        )}
      </section>

      {open && <FigureDialog hit={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function FigureDialog({ hit, onClose }: { hit: SearchHit; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    // No cleanup: removing the element closes it. Calling close() in a cleanup would
    // fire onClose and dismiss the dialog under React StrictMode's double effects.
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal?.();
  }, []);

  async function download() {
    const { url } = await api.downloadUrl(hit.document.id);
    window.location.assign(url);
  }

  return (
    <dialog ref={ref} className="dialog" onClose={onClose} onClick={(e) => e.target === ref.current && onClose()}
      aria-labelledby="figure-title">
      <div className="dialog-body">
        <button className="dialog-close" onClick={onClose} aria-label="Close">×</button>
        <img className="dialog-image" src={hit.image_url} alt={hit.title ?? hit.figure_type} />
        <div className="dialog-details">
          <span className="type-chip">{hit.figure_type}</span>
          <h2 id="figure-title">{hit.title || "Untitled figure"}</h2>
          <p className="figure-caption">
            Page {hit.page} of {hit.document.title ?? hit.document.filename}
            {hit.collection && <>, in {hit.collection.name}</>}
          </p>
          {hit.description && <p>{hit.description}</p>}
          <details>
            <summary>Full analysis</summary>
            <pre className="json">{JSON.stringify(hit.analysis, null, 2)}</pre>
          </details>
          <div className="actions">
            <button className="btn btn-primary btn-sm" onClick={() => void download()}>Download PDF</button>
            <Link className="btn btn-ghost btn-sm" to={`/app/search?document=${hit.document.id}`} onClick={onClose}>
              All figures in this paper
            </Link>
          </div>
        </div>
      </div>
    </dialog>
  );
}
