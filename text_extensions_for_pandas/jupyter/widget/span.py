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
# Contains the span elements of the dataframe/spanarray widget
#

import bisect
import collections

from text_extensions_for_pandas.jupyter.widget.stubs import (
    ipw, display, clear_output, HTML)

_COLOR_PALETTE = [
    "F6DF5F",
    "8A7CCC",
    "9dcfaa",
    "EA878B",
    "718AF7",
    "6099E2",
    "6DD1F6",
    "5DDEE0",
    "87F4D3",
]


def DataFrameDocumentContainerComponent(widget): # -> ipw.Widget
    """A Component that separates a dataframe by document and generates their components."""

    dataframe = widget._df

    target_texts = []
    spans_by_text = {}

    span_columns = []
    # Check if any of the columns are of dtype SpanArray or TokenSpanArray
    for column_name in dataframe.columns:
        column = dataframe[column_name]

        if str(column.dtype) == "SpanDtype" or str(column.dtype) == "TokenSpanDtype":
            span_columns.append(column.name)

    if len(span_columns) > 0:
        # Iterate over all rows in the dataframe and add that span's text
        for span_column in span_columns:
            for i_loc in range(len(dataframe[span_column])):
                item = dataframe[span_column][i_loc]
                target_text = item.target_text
                if target_text not in target_texts:
                    target_texts.append(target_text)
                    spans_by_text[target_text] = []

                spans_by_text[target_text].append(
                    {"span": item, "i_loc": i_loc, "column": span_column,}
                )

        # Add the document display controls
        controls = DataFrameDocumentControlsComponent(
            widget=widget, span_columns=span_columns
        )

        # Generate a widget component for each document
        documents = []
        for text in target_texts:
            documents.append(
                DataFrameDocumentComponent(
                    widget=widget, text=text, spans=spans_by_text[text]
                )
            )

        documents_widget = ipw.VBox(children=[controls, *documents])
        documents_widget.add_class("tep--spanvis")
        return documents_widget


def DataFrameDocumentControlsComponent(widget, span_columns):  # -> ipw.Widget:
    """A widget that exposes controls for rendering spans in various ways."""

    control_widgets = []

    # Branch based on number of span columns in the dataframe

    # If there's only one span column, we just want to choose the tag column from all
    # columns
    if len(span_columns) == 1:
        # Tag dropdown
        tag_values = map(lambda v: (v, ("column_data", v)), widget._df.columns.values)
        tag_dropdown = ipw.Dropdown(
            options=[("None", None), *tag_values],
            description="tag",
            value=widget._tag_display,
        )
        tag_dropdown.observe(widget._update_tag, names="value")
        control_widgets.append(tag_dropdown)

        # Color Mode Dropdown
        color_mode_values = [("By Row", "ROW"), ("By Tag", "TAG")]
        color_dropdown = ipw.Dropdown(
            options=color_mode_values,
            description="color_mode",
            value=widget._color_mode,
        )
        color_dropdown.observe(widget._update_color_mode, names="value")
        control_widgets.append(color_dropdown)

        return ipw.HBox(control_widgets)

    # If there are multiple span columns, add some additional options
    elif len(span_columns) > 1:
        # Tag Dropdown
        tag_values = map(lambda v: (v, ("column_data", v)), widget._df.columns.values)
        tag_dropdown = ipw.Dropdown(
            options=[("None", None), ("Col Headers", ("column_header")), *tag_values],
            description="tag",
            value=widget._tag_display,
        )
        tag_dropdown.observe(widget._update_tag, names="value")
        control_widgets.append(tag_dropdown)

        # Color Mode Dropdown
        color_mode_values = [("By Row", "ROW"), ("By Tag", "TAG")]
        color_dropdown = ipw.Dropdown(
            options=color_mode_values,
            description="color_mode",
            value=widget._color_mode,
        )
        color_dropdown.observe(widget._update_color_mode, names="value")
        control_widgets.append(color_dropdown)

        return ipw.HBox(control_widgets)

    return ipw.HTML("")


def DataFrameDocumentComponent(widget, text, spans): # -> ipw.Widget
    """A component that renders the context of a document by generating visisble
    highlights for a column of spans."""

    # Create some color tracking variables
    _color_map = {}
    _color_index = 0

    # Create stacks
    begin_stack = []
    end_stack = []

    for entry in spans:
        span = entry["span"]
        i_loc = entry["i_loc"]  # Index in the dataframe
        column = entry["column"]  # Column header
        if span.begin < span.end:
            bisect.insort(begin_stack, (span.begin, i_loc, column))
            bisect.insort(end_stack, (span.end, i_loc, column))

    begin_stack = collections.deque(begin_stack)
    end_stack = collections.deque(end_stack)

    start_index = 0
    open_spans = []

    document_elements: list(str) = []

    while len(begin_stack) > 0 and len(end_stack) > 0:
        if begin_stack[0][0] < end_stack[0][0]:
            if len(open_spans) == 0:
                document_elements = document_elements + _get_linebreak_text_array(
                    text[start_index : begin_stack[0][0]]
                )

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
            else:
                span_tag = _get_span_tag(
                    widget._df, widget._tag_display, begin_stack[0]
                )
                span_text = text[start_index : begin_stack[0][0]]
                span_color, _color_map, _color_index = _get_span_color(
                    widget._color_mode,
                    begin_stack[0],
                    span_tag,
                    _color_map,
                    _color_index,
                )
                document_elements.append(
                    DocumentSpan(
                        text=text[start_index : begin_stack[0][0]],
                        tag=span_tag,
                        show_tag=False,
                        bgcolor=span_color,
                        span_indices=open_spans,
                    )
                )

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
        else:
            span_tag = _get_span_tag(widget._df, widget._tag_display, end_stack[0])
            span_text = text[start_index : end_stack[0][0]]
            span_color, _color_map, _color_index = _get_span_color(
                widget._color_mode, end_stack[0], span_tag, _color_map, _color_index
            )
            document_elements.append(
                DocumentSpan(
                    text=span_text,
                    tag=span_tag,
                    show_tag=True,
                    bgcolor=span_color,
                    span_indices=open_spans,
                )
            )

            start_index = end_stack[0][0]
            open_spans.remove(end_stack[0][1])
            end_stack.popleft()

    while len(end_stack) > 0:
        span_tag = _get_span_tag(widget._df, widget._tag_display, end_stack[0])
        span_text = text[start_index : end_stack[0][0]]
        span_color, _color_map, _color_index = _get_span_color(
            widget._color_mode, end_stack[0], span_tag, _color_map, _color_index
        )
        document_elements.append(
            DocumentSpan(
                text=span_text,
                tag=span_tag,
                show_tag=True,
                bgcolor=span_color,
                span_indices=open_spans,
            )
        )

        start_index = end_stack[0][0]
        open_spans.remove(end_stack[0][1])
        end_stack.popleft()

    document_elements.append(text[start_index:])

    return ipw.HTML(
        f"""
            <div class='document'>
                {"".join(document_elements)}
            </div>
        """
    )


def DocumentSpan(
    text: str,
    tag: str,
    show_tag=True,
    bgcolor="rgba(200, 180, 255, 0.5)",
    span_indices=[],
) -> str:
    return f"""
        <span
            class="tep--spanvis--span"
            data-ids="{" ".join(map(lambda x: str(x), span_indices))}"
            style="line-height: 2; display: inline-block; padding: 0 0.2em; background-color: {bgcolor};">
            {text}
            {f"<span class='tep--spanvis--tag' style='margin:0 0.2em; font-size: 0.8em; font-weight: bold'>{tag}</span>" if show_tag else ""}
        </span>
    """


def _get_linebreak_text_array(in_text: str) -> str:
    splitarr = in_text.split("\n")
    i = 1
    while i < len(splitarr):
        splitarr.insert(i, "<br>")
        i += 2
    return splitarr


def _get_span_tag(df, tag_data, stack_data) -> str:
    """Gets the appropriate tag for a span from either the column header or cell."""
    if tag_data:
        if tag_data[0] == "column_data":
            return df[tag_data[1]][stack_data[1]]
        elif tag_data == "column_header":
            return str(stack_data[2])
    return ""


def _get_span_color(
    color_mode: str, stack_data, tag: str, color_map: dict, color_index: int
) -> tuple:
    """Gets the appropriate color for a span and returns (the color, the updated color map, the updated color index)"""
    if color_mode == "ROW":
        return (
            f"#{_COLOR_PALETTE[stack_data[1] % len(_COLOR_PALETTE)]}",
            color_map,
            color_index,
        )
    elif color_mode == "TAG":
        if tag in color_map:
            return color_map[tag], color_map, color_index
        else:
            color_map.update(
                {tag: f"#{_COLOR_PALETTE[color_index % len(_COLOR_PALETTE)]}"}
            )
            color_index += 1
            return color_map[tag], color_map, color_index
