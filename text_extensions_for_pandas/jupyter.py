#
#  Copyright (c) 2020 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
The ``jupyter`` module contains functions to support the use of Text Extensions for Pandas
 in Jupyter notebooks.
"""
#
# jupyter.py
#
# Part of text_extensions_for_pandas
#
#
#
import textwrap

import pandas as pd
import numpy as np
import time
from typing import *


def run_with_progress_bar(num_items: int, fn: Callable, item_type: str = "doc") \
        -> List[pd.DataFrame]:
    """
    Display a progress bar while iterating over a list of dataframes.

    :param num_items: Number of items to iterate over
    :param fn: A function that accepts a single integer argument -- let's
     call it `i` -- and performs processing for document `i` and returns
     a `pd.DataFrame` of results
    :param item_type: Human-readable name for the items that the calling
     code is iterating over

    """
    # Imports inline to avoid creating a hard dependency on ipywidgets/IPython
    # for programs that don't call this funciton.
    # noinspection PyPackageRequirements
    import ipywidgets
    # noinspection PyPackageRequirements
    from IPython.display import display

    _UPDATE_SEC = 0.1
    result = []  # Type: List[pd.DataFrame]
    last_update = time.time()
    progress_bar = ipywidgets.IntProgress(0, 0, num_items,
                                          description="Starting...",
                                          layout=ipywidgets.Layout(width="100%"),
                                          style={"description_width": "12%"})
    display(progress_bar)
    for i in range(num_items):
        result.append(fn(i))
        now = time.time()
        if i == num_items - 1 or now - last_update >= _UPDATE_SEC:
            progress_bar.value = i + 1
            progress_bar.description = f"{i + 1}/{num_items} {item_type}s"
            last_update = now
    progress_bar.bar_style = "success"
    return result


#########################################################################################
# HTML style constants
# Eventually these should be replaced by a proper stylesheet.

# Background for boxes containing span info and annotated doc
_BG_COLOR = "color: var(--jp-layout-color2)"

# Border of boxes containing span info and annotated doc
_BORDER_STYLE = "border: 1px solid var(--jp-border-color0)"

# Background for highlighted span locations
#
# We ought to use the following combination:
# _HL_COLOR = ("background-color:var(--jp-info-color2); "
#              "color:var(--jp-content-font-color2)")
# ...but currently the stylesheets for JupyterLab dark mode don't display text on
# accent colors very well. So instead we mix orange-yellow into the current theme's
# background color.
_HL_COLOR = (
    "background-color:rgba(255, 215, 0, 0.5)"
)

# Font of rendered document text
_DOC_FONT = "font-family:var(--jp-code-font-family); font-size:var(--jp-code-font-size)"

# END HTML style constants
#########################################################################################


def _pretty_print_text(column: Union["SpanArray", "TokenSpanArray"]) -> List[str]:
    # Subroutine of pretty_print_html() below.
    # Should only be called for single-document span arrays.
    if not column.is_single_document:
        raise ValueError("Array contains spans from multiple documents. Can only "
                         "render one document at a time.")

    # Build up a mask of which characters in the target text are within
    # at least one span.
    text = column.document_text
    mask = np.zeros(shape=(len(text)), dtype=bool)
    # TODO: Vectorize
    for e in column:
        mask[e.begin:e.end] = True

    # Walk through the text, building up an HTML representation
    text_pieces = []
    for i in range(len(text)):
        if mask[i] and (i == 0 or not mask[i - 1]):
            # Starting a highlighted region
            text_pieces.append(
                f"""<span style="{_HL_COLOR}">""")
        elif not (mask[i]) and i > 0 and mask[i - 1]:
            # End of a bold region
            text_pieces.append("</span>")
        if text[i] == "\n":
            text_pieces.append("<br>")
        elif text[i] == "&":
            text_pieces.append("&amp;")
        elif text[i] == "<":
            text_pieces.append("&lt;")
        elif text[i] == ">":
            text_pieces.append("&gt;")
        elif text[i] == "\"":
            # Not strictly necessary, but just in case.
            text_pieces.append("&quot;")
        elif text[i] == "'":
            # Not strictly necessary, but just in case.
            text_pieces.append("&#39;")
        elif text[i] == "$":
            # Dollar sign messes up Jupyter's JavaScript UI.
            # Place dollar sign in its own sub-span to avoid being misinterpeted as a LaTeX delimiter
            text_pieces.append("<span>&#36;</span>")
        else:
            text_pieces.append(text[i])
    return text_pieces


def pretty_print_html(column: Union["SpanArray", "TokenSpanArray"],
                      show_offsets: bool) -> str:
    """
    HTML pretty-printing of a series of spans for Jupyter notebooks.

    Args:
        column: Span column (either character or token spans).
        show_offsets: True to generate a table of span offsets in addition
         to the marked-up text
    """
    # Local import to prevent circular dependencies
    from text_extensions_for_pandas.array.span import SpanArray
    from text_extensions_for_pandas.array.token_span import TokenSpanArray
    if not isinstance(column, (SpanArray, TokenSpanArray)):
        raise TypeError(f"Expected SpanArray or TokenSpanArray, but received "
                        f"{column} of type {type(column)}")



    doc_divs = [
        f"""
                <div style="float:center; padding:10px">
                    <p style="{_DOC_FONT}">
                        {"".join(_pretty_print_text(column_slice))}
                    </p>
                </div>
        """
        for column_slice in column.split_by_document()
    ]

    # TODO: Add a proper CSS stylesheet instead of embedding formatting into the
    #  generated HTML
    _NEWLINE = "\n"
    if show_offsets:
        # Generate a dataframe of atomic types to pretty-print the spans
        spans_html = column.as_frame().to_html()
        return textwrap.dedent(f"""
<div id="spanArray">
    <div id="spans" 
     style="{_BG_COLOR}; {_BORDER_STYLE}; float:left; padding:10px;">
        {spans_html}
    </div>
    <div id="text"
     style="float:right; {_BORDER_STYLE}; width: 60%;">
{_NEWLINE.join(doc_divs)}
    </div>
</div>
        """)
    else:  # if not show_offsets
        return textwrap.dedent(f"""
<div id="text"
 style="float:right; {_BG_COLOR}; {_BORDER_STYLE}; width: 100%;">
{_NEWLINE.join(doc_divs)}
</div>
        """)
