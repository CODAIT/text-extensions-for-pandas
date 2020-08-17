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

################################################################################
# Preferred way to install packages: Anaconda main
#
# We use YAML files to ensure that the CI environment will use the same
# configuration.
conda env update -n ${ENV_NAME} -f config/dev_env.yml

################################################################################

# All the installation steps that follow must be done from within the new
# environment.
conda activate ${ENV_NAME}

# Ensure a specific version of Pandas is installed
if [ -n "${PANDAS_VERSION}" ]; then
    echo "Ensuring Pandas ${PANDAS_VERSION} is installed"
    conda install pandas=${PANDAS_VERSION}
fi

################################################################################
# Second-best way to install packages: pip

# pip install with the project's requirements.txt so that any hard constraints
# on package versions are respected in the created environment.
pip install -r requirements.txt

# Additional layer of pip-installed stuff for running regression tests
pip install -r config/dev_reqs.txt

# Additional layer of pip-installed stuff for running notebooks
pip install -r config/jupyter_reqs.txt

################################################################################
# Least-preferred install method: Custom

# spaCy language models for English
python -m spacy download en_core_web_sm


# Finish installation of ipywidgets from the "conda-forge" section above
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager

# Also install the table of contents extension
jupyter labextension install --no-build @jupyterlab/toc

# Jupyter debugger extension (requires xeus-python, installed above)
jupyter labextension install --no-build @jupyterlab/debugger

# Build once after installing all the extensions, and skip minimization of the
# JuptyerLab resources, since we'll be running from the local machine.
jupyter lab build --minimize=False

conda deactivate

echo "Anaconda environment '${ENV_NAME}' successfully created."
echo "To use, type 'conda activate ${ENV_NAME}'."

