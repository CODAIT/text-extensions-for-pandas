#
# util.py
#
# Part of pandas_text
#
# Internal utility functions, not exposed in the public API.
#

# For convenience, give our utility functions full access to the public API
from pandas_text import *

from typing import *


def pretty_print_html(column: Union["CharSpanArray", "TokenSpanArray"]) -> str:
    """
    HTML pretty-printing of a series of spans for Jupyter notebooks.

    Args:
        column: Span column (either character or token spans)
    """
    # span_lines = [
    #    "<div>{}</div>".format(e) for e in self
    # ]
    # spans_html = "\n".join(span_lines)

    # Generate a dataframe of atomic types to pretty-print the spans
    spans_html = column.as_frame().to_html()

    # Build up a mask of which characters in the target text are within
    # at least one span.
    text = column.target_text
    mask = np.zeros(shape=(len(text)), dtype=np.bool)
    # TODO: Vectorize
    for e in column:
        mask[e.begin:e.end] = True

    # Walk through the text, building up an HTML representation
    text_pieces = []
    for i in range(len(text)):
        if mask[i] and (i == 0 or not mask[i - 1]):
            # Starting a bold region
            text_pieces.append("<b>")
        elif not (mask[i]) and i > 0 and mask[i - 1]:
            # End of a bold region
            text_pieces.append("</b>")
        if text[i] == "\n":
            text_pieces.append("<br>")
        text_pieces.append(text[i])

    # TODO: Use CSS here instead of embedding formatting into the
    #  generated HTML
    return """
    <div id="spanArray">
        <div id="spans" 
         style="background-color:lightblue; float:left; padding:10px;">
            {}
        </div>
        <div id="text"
         style="float:right; background-color:lightgreen; width: 60%;">
            <div style="float:center; padding:10px">
                <p style="font-family:monospace">
                    {}
                </p>
            </div>
        </div>
    </div>
    """.format(spans_html, "".join(text_pieces))
