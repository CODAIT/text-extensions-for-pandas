
# Release process

Steps to release a new version:

1. Ensure all regression tests are passing on all supported versions of Python
   and Pandas. The current versions are:
   * Python: 3.6, 3.7, 3.8
   * Pandas: 1.0.x, 1.1.x

1. Activate your Text Extensions for Pandas build environment (usually called 
   `pd`)

1. Ensure that all the notebooks under the `notebooks` directory run and
   produce substantially the same output as before. This step only needs to be
   done with our primary versions of Python and Pandas at the time of the
   release.

1. Ensure that all the notebooks under the `tutorials` directory run and
   produce substantially the same output as before. Note that some of these
   notebooks need to run overnight.

1. Ensure that the API docs generate without errors or warnings.

1. Increment the version number in `setup.py`.

1. Create and merge a pull request against master that increments the version 
   number.

1. Remove the `dist` directory if present and run 
   ```
   python setup.py sdist bdist_wheel
   ```

1. Inspect the contents of the `dist` directory. It should look something like
   this:
   ```
   (pd) freiss@fuzzy:~/pd/tep-alpha2$ ls dist
   text_extensions_for_pandas-0.1a2-py3-none-any.whl
   text_extensions_for_pandas-0.1a2.tar.gz
   (pd) freiss@fuzzy:~/pd/tep-alpha2$ 
   ```   

1. Upload to PyPI by running:
   ```
   python -m twine upload --repository testpypi dist/*
   ```   
   
1. Tag and create a new release using the Github web UI.

