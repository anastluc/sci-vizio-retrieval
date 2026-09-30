"""
Faceted semantic search over a user's figures.

Each user's corpus is small (at most a few thousand figures), so ranking is an
exact cosine-distance scan restricted by an index on ``user_id``. That is fast
at this size and, unlike an approximate (HNSW) index combined with a per-user
filter, never silently drops matches.
"""

import uuid
from typing import List, Optional, Tuple

from sqlalchemy import Select, and_, func, null, or_, select
from sqlalchemy.orm import Session

from sci_vizio_retrieval.figure_types import DECORATIVE
from sci_vizio_saas.models import Collection, Document, Figure, User
from sci_vizio_saas.schemas import UNFILED, Facets, FacetValue, SearchRequest

MAX_DOCUMENT_FACETS = 20


def _conditions(user: User, req: SearchRequest, skip: str = None) -> list:
    """WHERE clauses for the request; ``skip`` omits one facet's own filter (for its counts)."""
    conds = [Figure.user_id == user.id]

    if skip != "figure_types":
        if req.figure_types:
            conds.append(Figure.figure_type.in_(req.figure_types))
        elif not req.include_decorative:
            conds.append(Figure.figure_type != DECORATIVE)

    if skip != "collections" and req.collections:
        ids = [uuid.UUID(c) for c in req.collections if c != UNFILED]
        options = []
        if ids:
            options.append(Document.collection_id.in_(ids))
        if UNFILED in req.collections:
            options.append(Document.collection_id.is_(None))
        conds.append(or_(*options))

    if skip != "documents" and req.documents:
        conds.append(Figure.document_id.in_(req.documents))

    return conds


def _base(conds) -> Select:
    return select(Figure).join(Document, Figure.document_id == Document.id).where(and_(*conds))


def compute_facets(db: Session, user: User, req: SearchRequest) -> Facets:
    type_rows = db.execute(
        _base(_conditions(user, req, skip="figure_types"))
        .with_only_columns(Figure.figure_type, func.count())
        .group_by(Figure.figure_type)
        .order_by(func.count().desc())
    ).all()

    collection_rows = db.execute(
        _base(_conditions(user, req, skip="collections"))
        .outerjoin(Collection, Document.collection_id == Collection.id)
        .with_only_columns(Document.collection_id, Collection.name, func.count())
        .group_by(Document.collection_id, Collection.name)
        .order_by(func.count().desc())
    ).all()

    document_rows = db.execute(
        _base(_conditions(user, req, skip="documents"))
        .with_only_columns(Document.id, Document.title, Document.filename, func.count())
        .group_by(Document.id, Document.title, Document.filename)
        .order_by(func.count().desc())
        .limit(MAX_DOCUMENT_FACETS)
    ).all()

    return Facets(
        figure_types=[FacetValue(value=t, label=t, count=n) for t, n in type_rows],
        collections=[
            FacetValue(value=str(cid) if cid else UNFILED, label=name or "Unfiled", count=n)
            for cid, name, n in collection_rows
        ],
        documents=[
            FacetValue(value=str(did), label=title or filename, count=n)
            for did, title, filename, n in document_rows
        ],
    )


def search_figures(
    db: Session, user: User, req: SearchRequest, query_vector=None
) -> Tuple[int, List[Tuple[Figure, Document, Optional[Collection], Optional[float]]]]:
    """Return (total matches, page of (figure, document, collection, distance))."""
    conds = _conditions(user, req)
    total = db.scalar(_base(conds).with_only_columns(func.count()))

    stmt = (
        _base(conds)
        .outerjoin(Collection, Document.collection_id == Collection.id)
        .add_columns(Document, Collection)
    )
    if query_vector is not None:
        distance = Figure.embedding.cosine_distance(query_vector)
        stmt = stmt.add_columns(distance.label("distance")).order_by(distance, Figure.id)
    else:
        stmt = stmt.add_columns(null().label("distance")).order_by(Document.created_at.desc(), Figure.page, Figure.image_index)

    rows = db.execute(stmt.limit(req.limit).offset(req.offset)).all()
    return total, [(fig, doc, col, dist) for fig, doc, col, dist in rows]
