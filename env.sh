#! /bin/bash


# Create conda environment to run the notebooks in this directory.
# 
# By default, the environment will be called "pd". To use a different name,
# pass the name as the first argument to this script, i.e.
#
# $ ./env.sh my_environment_name

# Use Python 3.7 for now because TensorFlow and JupyterLab don't support 3.8
# yet.
PYTHON_VERSION=3.7

############################
# HACK ALERT *** HACK ALERT 
# The friendly folks at Anaconda thought it would be a good idea to make the
# "conda" command a shell function. 
# See https://github.com/conda/conda/issues/7126
# The following workaround will probably be fragile.
if [ -z "$CONDA_HOME" ]
then 
    echo "Error: CONDA_HOME not set."
    exit
fi
if [ -e "${CONDA_HOME}/etc/profile.d/conda.sh" ]
then
    # shellcheck disable=SC1090
    . "${CONDA_HOME}/etc/profile.d/conda.sh"
else
    echo "Error: CONDA_HOME (${CONDA_HOME}) does not appear to be set up."
    exit
fi
# END HACK
############################

# Check whether the user specified an environment name.
if [ "$1" != "" ]; then
    ENV_NAME=$1
else
    ENV_NAME="pd"
fi
echo "Creating an Anaconda environment called '${ENV_NAME}'"


# Remove the detrius of any previous runs of this script
conda env remove -n ${ENV_NAME}


conda create -y --name ${ENV_NAME} python=${PYTHON_VERSION}
conda activate ${ENV_NAME}

################################################################################
# Preferred way to install packages: Anaconda main
conda install -y \
    tensorflow \
    jupyterlab \
    pandas \
    regex \
    matplotlib \
    cython \
    grpcio-tools \
    pytorch \
    black \
    scikit-learn

################################################################################
# Second-best way to install packages: conda-forge
conda install -y -c conda-forge \
    spacy \
    pyarrow \
    fastparquet \
    plotly \
    ipywidgets \

# Post-install steps for ipywidgets on JupyterLab require Node
conda install -y -c conda-forge \
    nodejs


################################################################################
# Third-best way to install packages: pip
pip install memoized-property

# Watson tooling requires pyyaml to be installed this way.
pip install pyyaml

# Huggingface transformers library is currently only on PyPI
pip install transformers

################################################################################
# Least-preferred install method: Custom

# spaCy language models for English
python -m spacy download en_core_web_sm

# Plotly for JupyterLab
# Currently disabled because the install takes 5-10 minutes. 
#echo "Not installing jupyterlab-plotly because the install takes 5-10 minutes."
#echo "To install manually, activate the '${ENV_NAME}' environment and run the "
#echo "following command:"
#echo "   jupyter labextension install jupyterlab-plotly"
#jupyter labextension install jupyterlab-plotly

# Finish installation of ipywidgets from the "conda-forge" section above
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Also install the table of contents extension
jupyter labextension install @jupyterlab/toc

#jupyter lab build

conda deactivate

echo "Anaconda environment '${ENV_NAME}' successfully created."
echo "To use, type 'conda activate ${ENV_NAME}'."

