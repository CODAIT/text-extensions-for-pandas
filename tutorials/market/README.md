# Market intelligence use case for Text Extensions for Pandas

The notebooks in this directory go through an example market intelligence use case involving Text Extensions for Pandas, IBM Watson Natural Language Understanding, SpaCy, and Ray.

The use case is broken down as follows:
* [Market_Intelligence_Part1.ipynb](Market_Intelligence_Part1.ipynb): Use Text Extensions for Pandas and Watson Natural Language Understanding to identify executives quoted by name in IBM press releases.
* [Market_Intelligence_Part2.ipynb](Market_Intelligence_Part2.ipynb): Use Text Extensions for Pandas and the SpaCy dependency parser to extract the titles of the executives we found in part 1.
* [Market_Intelligence_Part3.ipynb](Market_Intelligence_Part3.ipynb): Use Text Extensions for Pandas and the semijoin trick to improve dependency parsing performance by 9x.
* [Market_Intelligence_Part4.ipynb](Market_Intelligence_Part4.ipynb): Use Ray actors to apply flow control to your application's interactions with web services like Watson Natural Language Understanding
* [Market_Intelligence_Part5.ipynb](Market_Intelligence_Part5.ipynb): Use Ray and parallel processing to accelerate the throughput of the end-to-end document processing pipeline for this use case
* [Market_Intelligence_Part6.ipynb](Market_Intelligence_Part6.ipynb): (bonus content) Use Ray actors to add detailed monitoring code to the parallel version of the processing pipeline and visualize the processing of individual documents over time.

## Running the notebooks in this directory

Steps to run the notebooks in this directory
1. Follow the instructions in the [top level README file](../../README.md) for setting up a Python environment for running Jupyter notebooks in this project.

1. Activate your Python environment and install the SpaCy dependency parser model (required for Part 2) by running the command:
   ```
   python -m spacy download en_core_web_trf
   ```

1. Create a free instance of Watson Natural Language Understanding by visiting [this page](https://www.ibm.com/cloud/watson-natural-language-understanding) and clicking on the "Get started free" button.

1. Set the following environment variables:
   * `IBM_API_KEY`: The API key for your Natural Language Understanding instance
   * `IBM_SERVICE_URL`: The service URL for your Natural Language Understanding instance
   
1. Type `jupyter lab` to start JupyterLab, navigate to this directory, and run the notebook of your choice.
