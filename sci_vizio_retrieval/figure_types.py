"""Map the vision model's free-text ``image_type`` onto a small set of facet categories."""

import re
from typing import Optional

DECORATIVE = "Decorative"
OTHER = "Other"

# First matching rule wins, so specific patterns come before generic ones.
_RULES = [
    ("Bar chart", r"\bbar\b"),
    ("Line chart", r"\bline (graph|chart|plot)\b"),
    ("Scatter plot", r"\bscatter"),
    ("Pie chart", r"\bpie\b|\bdonut\b"),
    ("Histogram", r"\bhistogram"),
    ("Heatmap", r"heat ?map|confusion matrix"),
    ("Table", r"\btable\b"),
    ("Flowchart", r"flow ?chart|\bflow diagram|process diagram"),
    (DECORATIVE, r"\b(icon|logo|emblem|gradient|abstract|solid|blank|plain|button|barcode|qr|"
                 r"background|pattern|texture|placeholder)"),
    ("Chart", r"\b(graph|chart|plot|radar|dashboard|curve)"),
    ("Diagram", r"diagram|schematic|architecture|network|tree|illustration of (a|the) (model|system)"),
    ("Equation", r"equation|formula"),
    ("Map", r"\bmap\b"),
    ("Screenshot", r"screen ?shot|user interface|\bui\b|webpage"),
    ("Photo", r"photo|microscop|x-ray|scan\b"),
    ("Illustration", r"illustration|drawing|cartoon|sketch|rendering"),
    ("Text", r"\btext|document|form\b|bibliography|page"),
]
_COMPILED = [(category, re.compile(pattern)) for category, pattern in _RULES]

CATEGORIES = [category for category, _ in _RULES] + [OTHER]


def normalize_figure_type(image_type: Optional[str]) -> str:
    """Return the facet category for a model-reported image type, e.g. 'horizontal bar graph' -> 'Bar chart'."""
    text = (image_type or "").strip().lower().replace("_", " ").replace("-", " ")
    if not text:
        return OTHER
    for category, pattern in _COMPILED:
        if pattern.search(text):
            return category
    return OTHER
