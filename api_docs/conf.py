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




import os
import sys

import sphinx_rtd_theme

# This file is located one level above the root of the project. Add .. to the
# path so we can find the Python code to be documented.
sys.path.insert(0, os.path.abspath(".."))

print(f"sys.path is {sys.path}")

# -- Project information -----------------------------------------------------

project = "Text Extensions for Pandas"
copyright = "2021, IBM"
author = "IBM"

# Front page of API docs is located at index.rst
master_doc = "index"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# What Sphinx extensions to activate. If something is not on this list, it
# won't run.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",

    # Uncomment the following line to enable full automatic generation of
    # API documentation files from code (currently we hard-code an 
    # entry point for each module and rely on autodoc)
    # "sphinxcontrib.apidoc"

    # Third-party theme that provides a floating table of contents
    "sphinx_rtd_theme",

    # Third-party plugin that creates tables of contents for classes.
    # Used to generate TOC entries for our extension array types.
    "autoclasstoc"
]

# Configure the sphinx.ext.autodoc extension
autodoc_default_options = {
    # TODO: Re-enable this once readthedocs.org upgrades to a version of
    #  Sphinx where True is an acceptable value for the "members" option.
    #  Then remove the redundant :members: and :undoc-members: annotations
    #  from index.rst.
    #"members": True,
    #"undoc-members": True,
}

# Configure the sphinxcontrib.apidoc extension (currently not used)
apidoc_module_dir = "../text_extensions_for_pandas"
apidoc_output_dir = "."
apidoc_excluded_paths = ["test_*.py"]
apidoc_separate_modules = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# Sections to include in class tables of contents from the ``autoclasstoc``
# plugin.
autoclasstoc_sections = [
    "public-attrs",
    "public-methods",
]



# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','text_extensions_for_pandas.util.rst',
                    'text_extensions_for_pandas.io.watson_util.rst']

#look through files in this directory that include `test_` in their name. 
# add them to the exclude_patterns list 
for filename in os.listdir():
    if ".test_" in filename:
        exclude_patterns.append(filename)


# -- Options for HTML output -------------------------------------------------


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": -1,
}
print(f"{html_theme_options}")

html_static_path = ['_static']



