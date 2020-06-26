#! /bin/bash


# this script generates auto-generated documentation for the Text_extensions for pandas package
# using Sphinx apidoc and Sphinx autodoc 
# 
# To make adjustments to the landing page, edit the index.rst file in the docs directory 
# To make adjustments to the style or change settings, modify the conf.py in the docs directory 
# 
# This script is made to run using the same python environment as is used for the package
# and should be run accordingly 


#install sphinx if it is not already installed
conda install -y sphinx

# part one of generating sphinx autodocumentation 
# builds the .rst files that sphinx then translates into documents
sphinx-apidoc -f -e -o docs text_extensions_for_pandas

# part two of generating sphinx autodocumentation 
# you may see several warnings at this point. This is normal. 
sphinx-build -b html docs docs/_build/html

# clean up rst files created by the documentation process. 
find docs -not -name 'index.rst' -name '*.rst' -delete
