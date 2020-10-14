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
    PYTHON_VERSION="3.7"
fi
ENV_NAME="pd"


usage() {
    echo "Usage: ./env.sh [-h] [--env_name <name>] "
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
        --python_version)
            if [ "$2" ]; then PYTHON_VERSION=$2; shift
            else die "ERROR: --python_version requires an environment name"
            fi
            ;;
        --pandas_version)
            if [ "$2" ]; then PANDAS_VERSION=$2; shift
            else die "ERROR: --pandas_version requires an environment name"
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

echo "Creating environment '${ENV_NAME}' with Python '${PYTHON_VERSION}'."
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
# Create the environment

# Remove the detrius of any previous runs of this script
conda env remove -n ${ENV_NAME}

conda create -y --name ${ENV_NAME} python=${PYTHON_VERSION}

################################################################################
# All the installation steps that follow must be done from within the new
# environment.
conda activate ${ENV_NAME}

# Ensure a specific version of Pandas is installed


################################################################################
# Install packages with pip

# pip install with the project's requirements.txt so that any hard constraints
# on package versions are respected in the created environment.
pip install -r requirements.txt

# Additional layer of pip-installed stuff for running regression tests
pip install -r config/dev_reqs.txt

# Additional layer of pip-installed stuff for running notebooks
pip install -r config/jupyter_reqs.txt

# Additional layer of large packages
pip install -r config/big_reqs.txt

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

