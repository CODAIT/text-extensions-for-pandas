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
import text_extensions_for_pandas.resources

# TODO: This try/except block is for Python 3.6 support, and should be reduced to just importing importlib.resources when 3.6 support is dropped.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

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


def _get_sanitized_doctext(column: Union["SpanArray", "TokenSpanArray"]) -> List[str]:
    # Subroutine of pretty_print_html() below.
    # Should only be called for single-document span arrays.
    if not column.is_single_document:
        raise ValueError("Array contains spans from multiple documents. Can only "
                         "render one document at a time.")

    text = column.document_text

    text_pieces = []
    for i in range(len(text)):
        if text[i] == "'":
            text_pieces.append("\\'")
        else:
            text_pieces.append(text[i])
    return "".join(text_pieces)

# Limits the max number of displayed documents. Matches Pandas' default display.max_seq_items.
_DOCUMENT_DISPLAY_LIMIT = 100

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


    # Gets the main script and stylesheet from the 'resources' sub-package
    style_text: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "span_array.css")
    script_text: str = pkg_resources.read_text(text_extensions_for_pandas.resources, "span_array.js")

    # Declare initial variables common to all render calls
    instance_init_script_list: List[str] = []

    # For each document, pass the array of spans and document text into the script's render function
    document_columns = column.split_by_document()
    for column_index in range(min(_DOCUMENT_DISPLAY_LIMIT, len(document_columns))):
        # Get a javascript representation of the column
        span_array = []
        for e in document_columns[column_index]:
            span_array.append(f"""[{e.begin},{e.end}]""")
        
        instance_init_script_list.append(f"""
            {{
                const doc_spans = Span.arrayFromSpanArray([{','.join(span_array)}])
                const doc_text = '{_get_sanitized_doctext(document_columns[column_index])}'
                documents.push({{doc_text: doc_text, doc_spans: doc_spans}})
            }}
        """)

    # Defines a list of DOM strings to be appended to the end of the returned HTML.
    postfix_tags: List[str] = []
    
    if len(document_columns) > _DOCUMENT_DISPLAY_LIMIT:
        postfix_tags.append(f"""
            <footer>Documents truncated. Showing {_DOCUMENT_DISPLAY_LIMIT} of {len(document_columns)}</footer>
        """)

    # Get the show_offsets parameter as a JavaScript boolean
    show_offset_string = 'true' if show_offsets else 'false'
    
    return textwrap.dedent(f"""
        <script>
        {{
            {textwrap.indent(script_text, '        ')}
        }}
        </script>
        <style>
            {textwrap.indent(style_text, '        ')}
        </style>
        <div class="span-array">
            If you're reading this message, your notebook viewer does not support Javascript execution. Try pasting the URL into a service like nbviewer.
        </div>
        <script>
            {{
                const Span = window.SpanArray.Span
                const script_context = document.currentScript
                const documents = []
                {''.join(instance_init_script_list)}
                const instance = new window.SpanArray.SpanArray(documents, {show_offset_string}, script_context)
                instance.render()
            }}
        </script>
        {''.join(postfix_tags)}
    """)
