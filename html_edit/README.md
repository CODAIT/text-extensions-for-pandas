
# html_edit

[![Build Status](https://travis-ci.org//html_edit.svg?branch=master)](https://travis-ci.org//html_edit)
[![codecov](https://codecov.io/gh//html_edit/branch/master/graph/badge.svg)](https://codecov.io/gh//html_edit)


A custom widget

## Installation

You can install using `pip`:

```bash
pip install html_edit
```

If you are using Jupyter Notebook 5.2 or earlier, you may also need to enable
the nbextension:
```bash
jupyter nbextension enable --py [--sys-prefix|--user|--system] html_edit
```

## Development Installation

Create a dev environment:
```bash
conda create -n html_edit-dev -c conda-forge nodejs yarn python jupyterlab
conda activate html_edit-dev
```

Install the python. This will also build the TS package.
```bash
pip install -e ".[test, examples]"
```

When developing your extensions, you need to manually enable your extensions with the
notebook / lab frontend. For lab, this is done by the command:

```
jupyter labextension develop --overwrite .
yarn run build
```

For classic notebook, you need to run:

```
jupyter nbextension install --sys-prefix --symlink --overwrite --py html_edit
jupyter nbextension enable --sys-prefix --py html_edit
```

Note that the `--symlink` flag doesn't work on Windows, so you will here have to run
the `install` command every time that you rebuild your extension. For certain installations
you might also need another flag instead of `--sys-prefix`, but we won't cover the meaning
of those flags here.

### How to see your changes
#### Typescript:
If you use JupyterLab to develop then you can watch the source directory and run JupyterLab at the same time in different
terminals to watch for changes in the extension's source and automatically rebuild the widget.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
yarn run watch
# Run JupyterLab in another terminal
jupyter lab
```

After a change wait for the build to finish and then refresh your browser and the changes should take effect.

#### Python:
If you make a change to the python code then you will need to restart the notebook kernel to have it take effect.
