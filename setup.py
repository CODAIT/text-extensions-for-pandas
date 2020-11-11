#
#  Copyright (c) 2020 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import setuptools

with open("package.md", "r") as fh:
    long_description = fh.read()

# read requirements from file
with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="text_extensions_for_pandas",
    version="0.1b1",
    author="IBM",
    author_email="frreiss@example.com",
    description="Natural language processing support for Pandas dataframes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Note that this URL is where the project *will* be, not where it currently is.
    url="https://github.com/CODAIT/text-extensions-for-pandas",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
