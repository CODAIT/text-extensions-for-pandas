
# Release process

Steps to release a new version:

1. Ensure all regression tests are passing on all supported versions of Python
   and Pandas. The current versions are:
   * Python: 3.6, 3.7, 3.8
   * Pandas: 1.0.x, 1.1.x, 1.2.x, 1.3.x
   
   It is not necessary to test all combinations; just make sure that you test
   each Python version and each Pandas version at least once.
   
   To install the latest Pandas on the 1.0.x branch, use the command:
   ```
   pip install --upgrade --force "pandas>=1.0,<1.1"
   ```
   
1. Ensure Watson NLU service API tests are enabled and passing by setting 
   IBM_API_KEY and IBM_SERVICE_URL, then running unit tests with pytest.

1. Make sure that the package installs and imports on a Python environment with 
   only the packages in `requirements.txt` by running the following commands 
   from the root of your local copy of the repository:
   ```
   conda create -y --prefix ./testenv python=3.8 pip
   conda activate ./testenv
   pip install -r requirements.txt
   pip install --editable .
   python -c "import text_extensions_for_pandas as tp"
   ```
   The last command should succeed with no exceptions.
   
   You can safely deactivate and remove the Anaconda environment`./testenv` 
   once this step is done.

1. Activate your Text Extensions for Pandas build environment (usually called 
   `pd`).

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

1. Make sure that your build environment (usually `pd`) is still active.
   Remove the `dist` directory if present and run 
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

1. Create a new Anaconda environment and install JupyterLab and the `.whl` file
   you just created into the new environment:

   ```
   conda deactivate
   conda create --prefix testenv python=3.8 pip jupyterlab
   conda activate ./testenv
   pip install dist/text_extensions_for_pandas*.whl
   ```
   
1. Navigate to the `notebooks` directory, ctivate your new environment, start up 
   JupyterLab, and verify that the notebooks under `notebooks` still run.

1. (optional): Do a test upload to TestPyPI by running:
   ```
   python -m twine upload --repository testpypi dist/*
   ```
   
1. Upload to PyPI by running:
   ```
   python -m twine upload dist/*
   ```   
   
1. Tag and create a new release using the Github web UI.

## Conda Release Process

These steps are from the general process described in https://conda-forge.org/docs/maintainer/updating_pkgs.html

**A. Setup local repo and branch for the update**

   1. Fork repo https://github.com/conda-forge/text_extensions_for_pandas-feedstock
   2. Clone fork as origin and add upstream remote
   3. Fetch and rebase local master with upstream/master
   4. Make branch e.g. update_0_1_b3

**B. Edit and test the recipe file**
   1. Edit recipe/meta.yaml
   1. Update version string
   1. Download source code tar.gz file (link with "archive" ~ 23MB)from github release and run:
      openssl sha256 path/to/text_extensions_for_pandas-0.1b3.tar.gz
      update source/sha256 hash string in recipe
   1. Update dependency info to match requirements.txt
   1. Bump the build number if version is unchanged, reset build number to 0 if version is changed
   1. Test changes locally with  "python build-locally.py" (requires Docker)

**C. Push changes to the forked repo and make a PR**
   1. Follow instructions and checklist in PR
   1. Wait for checks to pass
   1. Merge PR to master branch, using Github interface is fine
   1. Once merged, the conda package will be created automatically
