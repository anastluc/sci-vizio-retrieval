from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sci_vizio_saas.api.documents import figure_out
from sci_vizio_saas.deps import get_current_user, get_db, get_services
from sci_vizio_saas.models import User
from sci_vizio_saas.schemas import CollectionRef, DocumentRef, SearchHit, SearchRequest, SearchResponse
from sci_vizio_saas.search import compute_facets, search_figures
from sci_vizio_saas.services import Services

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db),
           services: Services = Depends(get_services)):
    """
    Semantic search over the user's figures with facet filters.

    With an empty query, returns figures newest-document first (browse mode).
    """
    query = body.query.strip()
    query_vector = services.embedder.embed([query])[0] if query else None
    total, rows = search_figures(db, user, body, query_vector)

    results = [
        SearchHit(
            **figure_out(figure, services),
            score=None if distance is None else round(1.0 - float(distance), 4),
            document=DocumentRef(id=document.id, filename=document.filename, title=document.title),
            collection=CollectionRef(id=collection.id, name=collection.name) if collection else None,
        )
        for figure, document, collection, distance in rows
    ]
    return SearchResponse(total=total, results=results, facets=compute_facets(db, user, body))
