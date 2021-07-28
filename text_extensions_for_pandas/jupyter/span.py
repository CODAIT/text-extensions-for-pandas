#
#  Copyright (c) 2021 IBM Corp.
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

#
# span.py
#
# Part of text_extensions_for_pandas
#
# Support for span-centric Jupyter rendering and utilities
#

import textwrap
from typing import *
from enum import Enum
import text_extensions_for_pandas.resources

# TODO: This try/except block is for Python 3.6 support, and should be
# reduced to just importing importlib.resources when 3.6 support is dropped.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


# Limits the max number of displayed documents. Matches Pandas' default display.max_seq_items.
_DOCUMENT_DISPLAY_LIMIT = 100


class SetType(Enum):
    NESTED=1
    OVERLAP=2

class RegionType(Enum):
    NESTED=1
    COMPLEX=2
    SOLO=3


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
        token_span_array = []
        for e in document_columns[column_index]:
            span_array.append(f"""[{e.begin},{e.end}]""")
            if hasattr(e, "tokens"):
                token_span_array.append(f"""[{e.begin_token},{e.end_token}]""")

        document_object_script = f"""
            const doc_spans = [{','.join(span_array)}]
            const doc_text = '{_get_escaped_doctext(document_columns[column_index])}'
        """

        # If the documents are a TokenSpanArray, include the start and end token indices in the document object.
        if len(token_span_array) > 0:
            document_object_script += f"""
                const doc_token_spans = [{','.join(token_span_array)}]
                documents.push({{doc_text: doc_text, doc_spans: doc_spans, doc_token_spans: doc_token_spans}})
            """
        else:
            document_object_script += """
                documents.push({doc_text: doc_text, doc_spans: doc_spans})
            """

        
        instance_init_script_list.append(f"""
            {{
                {document_object_script}
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
        <style class="span-array-css">
            {textwrap.indent(style_text, '        ')}
        </style>
        <script>
        {{
            {textwrap.indent(script_text, '        ')}
        }}
        </script>
        <div class="span-array">
            {_get_initial_static_html(column, show_offsets)}
            <span style="font-size: 0.8em;color: #b3b3b3;">Your notebook viewer does not support Javascript execution. The above rendering will not be interactive.</span>
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

def _get_escaped_doctext(column: Union["SpanArray", "TokenSpanArray"]) -> List[str]:
    # Subroutine of pretty_print_html() above.
    # Should only be called for single-document span arrays.
    if not column.is_single_document:
        raise ValueError("Array contains spans from multiple documents. Can only "
                         "render one document at a time.")

    text = column.document_text

    text_pieces = []
    for i in range(len(text)):
        if text[i] == "'":
            text_pieces.append("\\'")
        elif text[i] == "\n":
            text_pieces.append("\\n")
        else:
            text_pieces.append(text[i])
    return "".join(text_pieces)

def _get_initial_static_html(column: Union["SpanArray", "TokenSpanArray"],
                      show_offsets: bool) -> str:
    # Subroutine of pretty_print_html above.
    # Gets the initial static html representation of the column for notebook viewers without JavaScript support.
    # Iterates over each document and constructs the DOM string with template literals.

    # ! Text inserted into the DOM as raw HTML should always be sanitized to prevent unintended DOM manipulation
    # and XSS attacks.

    documents = column.split_by_document()
    documents_html = []

    for column_index in range(min(_DOCUMENT_DISPLAY_LIMIT, len(documents))):
        document = documents[column_index]


        # Generate a dictionary to store span information, including relationships with spans occupying the same region.
        spans = {}
        is_token_document = False
        sorted_span_ids = []
        for i in range(len(document)):

            span_data = {}
            span_data["id"] = i
            span_data["begin"] = document[i].begin
            span_data["end"] = document[i].end
            if hasattr(document[i], "tokens"):
                is_token_document = True
                span_data["begin_token"] = document[i].begin_token
                span_data["end_token"] = document[i].end_token
            span_data["sets"] = []
            spans[i] = span_data

            sorted_span_ids.append(i)

        # Sort IDs
        sorted_span_ids.sort(key=lambda id: (spans[id]["begin"], -spans[id]["end"]))

        for i in range(len(sorted_span_ids)):
            span_data = spans[sorted_span_ids[i]]

            for j in range(i+1, len(sorted_span_ids)):
                sub_span_data = spans[sorted_span_ids[j]]
                # If the spans do not overlap, exit the sub-loop
                if(sub_span_data["begin"] >= span_data["end"]):
                    break
                else:
                    if(sub_span_data["end"] <= span_data["end"]):
                        span_data["sets"].append({"type": SetType.NESTED, "id": sub_span_data["id"]})
                    else:
                        span_data["sets"].append({"type": SetType.OVERLAP, "id": sub_span_data["id"]})

            spans[sorted_span_ids[i]] = span_data


        # Generate the table rows DOM string from span data.
        table_rows_html = []
        for i in range(len(spans)):
            span = spans[i]
            table_rows_html.append(f"""
                <tr>
                    <td><b>{span["id"]}</b></td>
                    <td>{span["begin"]}</td>
                    <td>{span["end"]}</td>
            """)

            if is_token_document:
                table_rows_html.append(f"""
                    <td>{span["begin_token"]}</td>
                    <td>{span["end_token"]}</td>
                """)

            table_rows_html.append(f"""
                    <td>{_get_sanitized_text(document.document_text[span["begin"]:span["end"]])}</td>
                </tr>
            """)


        # Generate the regions of the document_text to highlight from span data.
        mark_regions = []
        
        i = 0
        while i < len(document):

            region = {}
            region["root_id"] = i
            region["begin"] = spans[i]["begin"]

            set_span = _get_set_span(spans, i)
            region["end"] = set_span["end"]

            if len(spans[i]["sets"]) > 0:
                # get set span and type
                if(_is_complex(spans, i)):
                    region["type"] = RegionType.COMPLEX
                else:
                    region["type"] = RegionType.NESTED
            else:
                region["type"] = RegionType.SOLO
            mark_regions.append(region)

            i = set_span["highest_id"] + 1
        
        # Generate the document_text DOM string from the regions created above.
        context_html = []
        
        if len(mark_regions) == 0:
            # There are no marked regions. Just append the sanitized text as a raw string.
            context_html.append(_get_sanitized_text(document.document_text))
        else:
            # Iterate over each marked region and contruct the HTML for preceding text and marked text.
            # Then, append that HTML to the list of DOM strings for the document_text.
            snippet_begin = 0
            for region in mark_regions:
                context_html.append(f"""
                    {_get_sanitized_text(document.document_text[snippet_begin:region["begin"]])}
                """)
                
                if region["type"] == RegionType.COMPLEX:
                    context_html.append(f"""
                        <span class='mark btn-info complex-set' style='
                            padding:0.4em;
                            border-radius:0.35em;
                            background:linear-gradient(to right, #a0c4ff, #ffadad);
                            color: black;
                            '>{_get_sanitized_text(document.document_text[region["begin"]:region["end"]])}
                            <span class='mark-tag' style='
                                font-weight: bolder;
                                font-size: 0.8em;
                                font-variant: small-caps;
                                font-variant-caps: small-caps;
                                font-variant-caps: all-small-caps;
                                margin-left: 8px;
                                text-transform: uppercase;
                                color: black;
                                '>Set</span>
                        </span>
                    """)

                elif region["type"] == RegionType.NESTED:
                    mark_html = []
                    nested_snippet_begin = region["begin"]
                    # Iterate over each span nested within the root span of the marked region
                    for nested_span in map( \
                            lambda set: spans[set["id"]], 
                            spans[region["root_id"]]["sets"]):

                        mark_html.append(f"""
                            {_get_sanitized_text(document.document_text[nested_snippet_begin:nested_span["begin"]])}
                            <span class='mark btn-warning' style='
                                padding:0.2em 0.4em;
                                border-radius:0.35em;
                                background-color: #ffadad;
                                color: black;
                                '>{_get_sanitized_text(document.document_text[nested_span["begin"]:nested_span["end"]])}</span>
                        """)
                        nested_snippet_begin = nested_span["end"]
                    mark_html.append(_get_sanitized_text(document.document_text[nested_snippet_begin:region["end"]]))
                    context_html.append(f"""
                        <span class='mark btn-primary' style='padding:0.4em;border-radius:0.35em;background-color: #a0c4ff;color:black;'>{"".join(mark_html)}</span>
                    """)

                elif region["type"] == RegionType.SOLO:
                    context_html.append(f"""
                        <span class='mark btn-primary' style='padding:0.4em;border-radius:0.35em;background-color: #a0c4ff;color:black;'>{_get_sanitized_text(document.document_text[region["begin"]:region["end"]])}</span>
                    """)

                snippet_begin = region["end"]
            context_html.append(_get_sanitized_text(document.document_text[snippet_begin:]))
        
        # Generate the document's DOM string
        documents_html.append(f"""
            <div class='document'>
                <table style='
                    table-layout: auto;
                    overflow: hidden;
                    width: 100%;
                    border-collapse: collapse;
                    '>
                    <thead style='font-variant-caps: all-petite-caps;'>
                        <th></th>
                        <th>begin</th>
                        <th>end</th>
                        {"<th>begin token</th><th>end token</th>" if is_token_document else ""}
                        <th style='text-align:right;width:100%'>context</th>
                    </tr></thead>
                    <tbody>
                        {"".join(table_rows_html)}
                    </tbody>
                </table>
                <p style='
                    padding: 1em;
                    line-height: calc(var(--jp-content-line-height, 1.6) * 1.6);
                    '>
                    {"".join(context_html)}
                </p>
            </div>
        """)

    # Concat all documents and return the final DOM string
    return "".join(documents_html)

def _get_set_span(spans: Dict, id: int) -> Dict:
    # Subroutine of _get_initial_static_html() above.
    # Recursive algorithm to get the last end and ID values of the set of spans connected to span with the given ID
    # Will raise a KeyError exception if an invalid key is given
    
    end = spans[id]["end"]
    highest_id = id

    # For each span in the set of spans, get the return values and track the greatest endpoint index and ID values.
    for set in spans[id]["sets"]:
        other = _get_set_span(spans, set["id"])
        if other["end"] > end:
            end = other["end"]
        if other["highest_id"] > highest_id:
            highest_id = other["highest_id"]

    return {"end": end, "highest_id": highest_id}

def _is_complex(spans: Dict, id: int) -> bool:
    # Subroutine of _get_initial_static_html() above.
    # Returns True if the provided span should be considered a "Complex" span. Implementation details below.
    # Will raise a KeyError exception if an invalid key is given

    # If any connection sets are of type:overlap or nested beyond a depth of 1, return True
    for set in spans[id]["sets"]:
        if set["type"] == SetType.OVERLAP:
            return True
        elif set["type"] == SetType.NESTED:
            if len(spans[set["id"]]["sets"]) > 0:
                return True
    return False

def _get_sanitized_text(text: str) -> str:
    # Subroutine of _get_initial_static_html() above.
    # Returns a string with HTML reserved character replacements to avoid issues while rendering text as HTML

    text_pieces = []
    for i in range(len(text)):
        if text[i] == "&":
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
        elif text[i] == "\n" or text[i] == "\r":
            # Support for in-document newlines by replacing with line break elements 
            text_pieces.append("<br>")
        else:
            text_pieces.append(text[i])
    return "".join(text_pieces)
