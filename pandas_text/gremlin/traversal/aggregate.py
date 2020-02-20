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

# aggregate.py
#
# Gremlin traversal steps that perform aggregation

import pandas as pd

from pandas_text.gremlin.traversal.base import GraphTraversal, UnaryTraversal


class SumTraversal(UnaryTraversal):
    """
    Gremlin `sum()` step.

    If applied to a span-valued input, combines the spans into a minimal span
    that covers all inputs.
    """
    def __init__(self, parent):
        UnaryTraversal.__init__(self, parent)

    def compute_impl(self) -> None:
        tail_type = self.parent.step_types[-1]
        if tail_type != "p":
            raise NotImplementedError(f"Sum of step type '{tail_type}' not "
                                      f"implemented")
        input_series = self.parent.last_step()
        self._set_attrs(
            paths=pd.DataFrame({0: [input_series.sum()]}),
            step_types=["p"],
            aliases={}
        )

