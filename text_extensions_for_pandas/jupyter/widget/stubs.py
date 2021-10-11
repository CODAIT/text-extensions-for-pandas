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
# stubs.py
#
# Part of text_extensions_for_pandas
#
# Stub code for generating useful error messages when the ipywidgets
# and IPython libraries are not present.
#

from typing import List

class MissingLibraryStub:
    """
    Generic stub for a missing library. Every operation on it throws an exception
    if the library isn't present.
    """
    def __init__(self, library_name: str, types_list: List[str] = []):
        """
        :param library_name: Name of the library that this stub replaces.
        :param types_list: List of the types that this stub should emulate so as not to
         break Python type hints that reference the library stub.
        """
        self._library_name = library_name
        self._types_list = types_list

    def __getattr__(self, item):
        if item in self._types_list:
            # Return a type to keep Python type hints happy.
            return object
        raise ModuleNotFoundError(
            f"The library '{self._library_name}', which is required for this operation, "
            f"is not installed on this system."
        )


try:
    # noinspection PyPackageRequirements
    import ipywidgets as ipw
except ModuleNotFoundError:
    ipw = MissingLibraryStub("ipywidgets", ["Widget"])

try:
    # noinspection PyPackageRequirements
    from IPython.display import display, clear_output, HTML
except ModuleNotFoundError:
    display = MissingLibraryStub("IPython")
    clear_output = MissingLibraryStub("IPython")
    HTML = MissingLibraryStub("IPython")

__all__ = ("ipw", "display", "clear_output", "HTML")





