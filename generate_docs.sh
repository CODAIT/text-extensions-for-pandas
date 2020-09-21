#! /bin/bash


# Generate API documentation for the Text_extensions for pandas package
# using Sphinx.
# 
# To make adjustments to the landing page, edit the index.rst file in the 
# api_docs directory 
# To make adjustments to the style or change settings, modify the conf.py in the 
# api_docs directory 
# 
# This script is made to run using the same python environment as is used for the package
# and should be run accordingly 

# Clean up the results of previous runs
rm -rf api_docs/_build/html

# Old code to remove the outputs of Sphinx autoapi; no longer needed
#find api_docs -not -name 'index.rst' -name '*.rst' -delete

# Invoke Sphinx
sphinx-build -b html api_docs api_docs/_build/html

