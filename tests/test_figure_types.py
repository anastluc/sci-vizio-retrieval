import pytest

from sci_vizio_retrieval.figure_types import CATEGORIES, DECORATIVE, OTHER, normalize_figure_type


@pytest.mark.parametrize("raw, category", [
    ("bar chart", "Bar chart"),
    ("Horizontal stacked bar graph", "Bar chart"),
    ("multi-panel line graph", "Line chart"),
    ("scatter_plot", "Scatter plot"),
    ("donut chart", "Pie chart"),
    ("confusion matrix", "Heatmap"),
    ("comparison table", "Table"),
    ("flowchart", "Flowchart"),
    ("network architecture diagram", "Diagram"),
    ("icon", DECORATIVE),
    ("solid color background", DECORATIVE),
    ("abstract illustration", DECORATIVE),
    ("company logo", DECORATIVE),
    ("radar chart", "Chart"),
    ("webpage screenshot", "Screenshot"),
    ("photograph", "Photo"),
    ("text excerpt", "Text"),
    ("mathematical equations", "Equation"),
    ("choropleth map", "Map"),
    ("sequence of images", OTHER),
    ("", OTHER),
    (None, OTHER),
])
def test_normalize(raw, category):
    assert normalize_figure_type(raw) == category
    assert category in CATEGORIES
