#! /bin/bash

################################################################################
# Create conda environment to run the notebooks in this directory.
#
# See usage() below for the current set of arguments this script accepts.

################################################################################
# Argument processing

# Default values for parameters passed on command line
# Use environment variables if present.
# (-z predicate means "unset or empty string")
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.8"
fi
ENV_NAME="pd"


usage() {
    echo "Usage: ./env.sh [-h] [--env_name <name>] "
    echo "                     [--use_active_env]"
    echo "                     [--python_version <version>]"
    echo "                     [--pandas_version <version>]"
    echo ""
    echo "You can also use the following environment variables:"
    echo "      PYTHON_VERSION: Version of Python to install"
    echo "      PANDAS_VERSION: version of Pandas to install"
    echo "(command-line arguments override environment variables values)"
}

die() {
    echo $1
    usage
    exit 1
}

# Read command line arguments
# See Bash docs at http://mywiki.wooledge.org/BashFAQ/035
while :; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --env_name)
            if [ "$2" ]; then ENV_NAME=$2; shift
            else die "ERROR: --env_name requires an environment name"
            fi
            ;;
        --use_active_env)
            unset ENV_NAME; shift
            ;;
        --python_version)
            if [ "$2" ]; then PYTHON_VERSION=$2; shift
            else die "ERROR: --python_version requires a python version"
            fi
            ;;
        --pandas_version)
            if [ "$2" ]; then PANDAS_VERSION=$2; shift
            else die "ERROR: --pandas_version requires a pandas version"
            fi
            ;;
        ?*)
            die "Unknown option '$1'"
            ;;
        *) # No more options
            break
    esac
    shift # Move on to next argument
done

if [ -n "${ENV_NAME}" ]; then
    echo "Creating environment '${ENV_NAME}' with Python '${PYTHON_VERSION}'."
else
    echo "Using active environment with Python '${PYTHON_VERSION}'."
fi

if [ -n "${PANDAS_VERSION}" ]; then
    echo "Will use non-default Pandas version '${PANDAS_VERSION}'."
fi


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


################################################################################
# Create the environment if not using active

if [ -n "${ENV_NAME}" ]; then

    # Remove the detrius of any previous runs of this script
    conda env remove -n ${ENV_NAME}

    # Note that we also install pip and wheel, which may or may not be present
    # on the environment.
    conda create -y --name ${ENV_NAME} python=${PYTHON_VERSION} pip wheel

    ################################################################################
    # All the installation steps that follow must be done from within the new
    # environment.
    conda activate ${ENV_NAME}
fi

################################################################################
# Install packages with conda

# We currently install JupyterLab from conda because the pip packages are 
# broken for Anaconda environments with Python 3.6 and 3.8 on Mac, as of
# April 2021.
conda install -y -c conda-forge jupyterlab
conda install -y -c conda-forge/label/main nodejs
conda install -y -c conda-forge jupyterlab-git

################################################################################
# Install packages with pip

# pip install with the project's requirements.txt so that any hard constraints
# on package versions are respected in the created environment.
echo "Installing from requirements.txt..."
pip install -r requirements.txt

# Additional layer of pip-installed stuff for running regression tests
echo "Installing from config/dev_reqs.txt..."
pip install -r config/dev_reqs.txt

# Additional layer of pip-installed stuff for running notebooks
echo "Installing from config/jupyter_reqs.txt..."
pip install -r config/jupyter_reqs.txt

# Additional layer of large packages
echo "Installing from config/big_reqs.txt..."
pip install -r config/big_reqs.txt

# Additional layer of packages that no longer work with Python 3.6
if [ "${PYTHON_VERSION}" == "3.6" ]; then
    echo "Skipping packages in config/non_36_reqs.txt because Python version"
    echo "is set to 3.6."
else
    echo "Installing from config/non_36_reqs.txt..."
    pip install -r config/non_36_reqs.txt
fi

# The previous steps will have installed some version of Pandas.
# Override that version if the user requested it.
if [ -n "${PANDAS_VERSION}" ]; then
    echo "Ensuring Pandas ${PANDAS_VERSION} is installed"
    pip install pandas==${PANDAS_VERSION}
fi

################################################################################
# Non-pip package installation

# spaCy language models for English
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf


# Finish installation of ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager

# Elyra extensions to JupyterLab (enables git integration, debugger, workflow
# editor, table of contents, and other features)
if [ "${PYTHON_VERSION}" == "3.6" ]; then
    echo "Skipping Elyra extensions because Python version is set to 3.6."
else
    pip install --upgrade --use-deprecated=legacy-resolver elyra
fi


# Build once after installing all the extensions, and skip minimization of the
# JuptyerLab resources, since we'll be running from the local machine.
jupyter lab build --minimize=False

if [ -n "${ENV_NAME}" ]; then
    conda deactivate

    echo "Anaconda environment '${ENV_NAME}' successfully created."
    echo "To use, type 'conda activate ${ENV_NAME}'."
else
    echo "Current environment updated successfully."
fi

