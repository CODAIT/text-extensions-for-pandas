#! /bin/bash



# Create conda environment "pd" to run the notebooks in this directory.

# Use Python 3.7 for now because TensorFlow and JupyterLab don't support 3.8
# yet.
PYTHON_VERSION=3.7

ENV_NAME="pd"

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
    . ${CONDA_HOME}/etc/profile.d/conda.sh
else
    echo "Error: CONDA_HOME (${CONDA_HOME}) does not appear to be set up."
    exit
fi
# END HACK
############################


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
    regex


################################################################################
# Second-best way to install packages: conda-forge
conda install -y -c conda-forge spacy

################################################################################
# Third-tier way to install packages: pip
pip install memoized-property

################################################################################
# Least-preferred install method: Custom

# Spacy language models for English
python -m spacy download en_core_web_sm


conda deactivate

