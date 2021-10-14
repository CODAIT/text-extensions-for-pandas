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

resources_dir = "text_extensions_for_pandas/resources"


setuptools.setup(
    name="text_extensions_for_pandas",
    version="0.2.0",
    author="IBM",
    author_email="frreiss@us.ibm.com",
    description="Natural language processing support for Pandas dataframes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CODAIT/text-extensions-for-pandas",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
    package_data={"": ["LICENSE.txt", 
                       f"{resources_dir}/*.css",
                       f"{resources_dir}/*.js",
                       f"{resources_dir}/*.png",
                       f"{resources_dir}/*.svg",]},
    include_package_data=True
)
