#
# util.py
#
# Part of text_extensions_for_pandas
#
# Internal utility functions, not exposed in the public API.
#

import math
import numpy as np
from typing import *
import unittest

# Internal imports

_ELLIPSIS = " [...] "
_ELLIPSIS_LEN = len(_ELLIPSIS)


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
            text_pieces.append("&#36;")

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


class TestBase(unittest.TestCase):
    """
    Base class to hold common utility code used by test cases in multiple files.
    """

    def _assertArrayEquals(self, a1: Union[np.ndarray, List[Any]],
                           a2: Union[np.ndarray, List[Any]]) -> None:
        """
        Assert that two arrays are completely identical, with useful error
        messages if they are not.

        :param a1: first array to compare. Lists automatically converted to
         arrays.
        :param a2: second array (or list)
        """
        a1 = np.array(a1) if isinstance(a1, np.ndarray) else a1
        a2 = np.array(a2) if isinstance(a2, np.ndarray) else a2
        if len(a1) != len(a2):
            raise self.failureException(
                f"Arrays:\n"
                f"   {a1}\n"
                f"and\n"
                f"   {a2}\n"
                f"have different lengths {len(a1)} and {len(a2)}"
            )
        mask = (a1 == a2)
        if not np.all(mask):
            raise self.failureException(
                f"Arrays:\n"
                f"   {a1}\n"
                f"and\n"
                f"   {a2}\n"
                f"differ at positions: {np.argwhere(~mask)}"
            )
