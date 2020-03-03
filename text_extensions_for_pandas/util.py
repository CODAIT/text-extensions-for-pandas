#
# util.py
#
# Part of text_extensions_for_pandas
#
# Internal utility functions, not exposed in the public API.
#

import numpy as np
from typing import *

# Internal imports


def pretty_print_html(column: Union["CharSpanArray", "TokenSpanArray"]) -> str:
    """
    HTML pretty-printing of a series of spans for Jupyter notebooks.

    Args:
        column: Span column (either character or token spans)
    """

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
            # Starting a highlighted region
            text_pieces.append(
                """<span style="background-color:yellow">""")
        elif not (mask[i]) and i > 0 and mask[i - 1]:
            # End of a bold region
            text_pieces.append("</span>")
        if text[i] == "\n":
            text_pieces.append("<br>")
        text_pieces.append(text[i])

    # TODO: Use CSS here instead of embedding formatting into the
    #  generated HTML
    return """
    <div id="spanArray">
        <div id="spans" 
         style="background-color:#F0F0F0; border: 1px solid #E0E0E0; float:left; padding:10px;">
            {}
        </div>
        <div id="text"
         style="float:right; background-color:#F5F5F5; border: 1px solid #E0E0E0; width: 60%;">
            <div style="float:center; padding:10px">
                <p style="font-family:monospace">
                    {}
                </p>
            </div>
        </div>
    </div>
    """.format(spans_html, "".join(text_pieces))
