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

import ipywidgets as ipw
import bisect
import collections

def DataFrameDocumentContainerComponent(dataframe, span_column=None, tag_column=None):
    """A Component that separates a dataframe by document and generates their components."""

    target_texts = []
    spans_by_text = {}

    if span_column != None:
        # Iterate over all rows in the frame and separate spans by target_text
        for index in range(len(dataframe["data"])):
            target_text = dataframe["data"][index][span_column].target_text
            if target_text not in target_texts:
                target_texts.append(target_text)
                spans_by_text[target_text] = []
            
            spans_by_text[target_text].append({
                "span": dataframe["data"][index][span_column],
                "index": index
            })

        # Generate a widget component for each document
        documents = []
        for text in target_texts:
            documents.append(DataFrameDocumentComponent(text=text, spans=spans_by_text[text]))
        
        return ipw.VBox(
            children=documents
        )

def DataFrameDocumentComponent(text, spans):
    """A component that renders the context of a document by generating visisble highlights for a column of spans."""

    # Create stacks
    begin_stack = []
    end_stack = []

    for entry in spans:
        span = entry["span"]
        index = entry["index"] # Index in the dataframe
        bisect.insort(begin_stack, (span.begin, index))
        bisect.insort(end_stack, (span.end, index))
    
    begin_stack = collections.deque(begin_stack)
    end_stack = collections.deque(end_stack)

    start_index = 0
    open_spans = []

    document_elements : list(str) = []

    
    while len(begin_stack) > 0 and len(end_stack) > 0:
        if(begin_stack[0][0] < end_stack[0][0]):
            if(len(open_spans) == 0):
                document_elements = document_elements + _get_linebreak_text_array(text[start_index:begin_stack[0][0]])

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
            else:
                span_tag = "PL"
                span_text = text[start_index:begin_stack[0][0]]
                document_elements.append(DocumentSpan(text=text[start_index:begin_stack[0][0]], tag=span_tag, show_tag=False))

                start_index = begin_stack[0][0]
                # Add the span's ID to the open ID list
                open_spans.append(begin_stack[0][1])
                begin_stack.popleft()
        else:
            span_tag = "PL"
            span_text = text[start_index:end_stack[0][0]]
            document_elements.append(DocumentSpan(text=span_text, tag=span_tag, show_tag=True))

            start_index = end_stack[0][0]
            open_spans.remove(end_stack[0][1])
            end_stack.popleft()

    while len(end_stack) > 0:
        span_tag = "PL"
        span_text = text[start_index:end_stack[0][0]]
        document_elements.append(DocumentSpan(text=span_text, tag=span_tag, show_tag=True))

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

def DocumentSpan(text, tag, show_tag=True, bgcolor="rgba(200, 180, 255, 0.5)") -> str:
    return f"""
        <span style="line-height: 2; display: inline-block; padding: 0 0.2em; background-color: {bgcolor};">{text}</span>
    """

def _get_linebreak_text_array(in_text: str) -> str:
    splitarr = in_text.split('\n')
    i = 1
    while(i < len(splitarr)):
        splitarr.insert(i, "<br>")
        i += 2
    return splitarr