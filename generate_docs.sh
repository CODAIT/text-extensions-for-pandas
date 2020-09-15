#! /bin/bash


# Generate API documentation for the Text_extensions for pandas package
# using Sphinx.
# 
# To make adjustments to the landing page, edit the index.rst file in the docs directory 
# To make adjustments to the style or change settings, modify the conf.py in the docs directory 
# 
# This script is made to run using the same python environment as is used for the package
# and should be run accordingly 

# Clean up the results of previous runs
rm -rf docs/_build/html

# Old code to remove the outputs of Sphinx autoapi; no longer needed
#find docs -not -name 'index.rst' -name '*.rst' -delete

# Invoke Sphinx
sphinx-build -b html docs docs/_build/html

