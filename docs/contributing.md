# Contributing

`main` is the maintained branch of this research fork. Follow the
[student quickstart](student_quickstart.md) first, then create one branch per change:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/describe-your-change
# Make and check the change, then commit the intended files.
git push -u origin codex/describe-your-change
```

Open a pull request against `jcbyts/ACI:main`. Include the problem, the change,
validation, and any effect on experiment identity. Keep task, sensor, controller,
and reward changes explicit. Record source revisions and seeds with results;
checkpoints and raw training outputs belong in the experiment artifact store.

Earlier feature branches are historical references. The old `main` is preserved
at `legacy-main-v0.0.0`.

## Setup

There are two forms of contributions: source code or documentation. Editing the documentation is as simple as cloning the repo and adding/editing content within the `docs` folder. All documentation is written in `markdown` and converted to `html` through `myst_parser` and `sphinx`. To edit the source code, as well as the documentation, you will want to install the package [through a symlink](#installing-a-symbolic-linked-version-for-testing).

```{note}
A `conda` or `virtualenv` will add isolation to your python environments and reduce conflicts amongst packages. It is _highly_ recommended to use one!!
```

### Cloning the Repo

Clone the repo as normal:

```bash
git clone https://github.com/jcbyts/ACI.git && cd ACI
```

### Installing a Symbolic Linked Version for Testing

A symbolic link or symlink is a file that references another. The advantages of symlinks is that a folder or file can _essentially_ be placed in two separate locations. In reference to this repository, we want to create a symlinked install because when we edit the code within the cloned repo, we want that change also to be reflected in the installed files.

```bash
# You need the dev requirements to build the documentation
pip install -e '.[docs,dev]'
```

### Documentation publishing

The existing Pages workflow builds documentation when changes reach `main`.
Use a pull request for documentation changes as well as source changes. This fork
has no configured PyPI release workflow; do not treat archival tags as releases.

## Guidelines

A lot of work has gone into making this package functional and scalable. Please consider all of the following guidelines and follow them to ensure the repository will persist for a long time.

### File Structure

The repository is structured as follows:

```
ACI/
├── cambrian/           # Source code
├── docs/               # Contains documentation
├── scripts/            # Scripts for running experiments
├── tools/              # Additional tooling for processing data
├── docker/             # Dockerfiles for running the package in a container
├── docker-compose.yml  # Docker-compose file for running the package in a container
├── LICENSE
├── README.md
└── pyproject.toml      # Package description and installation instructions for pip
```

### Editing the Source Code

If you plan on editing the source code, please visit the `cambrian/` folder. The `cambrian/` folder is structured as follows:

```
cambrian/
├── configs/            # YAML configuration files for experiments
├── models/             # XML model files for MuJoCo
├── utils/              # Utility files for use by the rest of the package
├── agents/             # Agent related utilities, methods, or classes
├── envs/               # Environment related utilities, methods, or classes
├── eyes/               # Eye related utilities, methods, or classes
├── ml/                 # Machine learning related utilities, methods, or classes
├── renderer/           # Renderer related utilities, methods, or classes
├── config.py           # Configuration file for the Hydra-based system
└── main.py             # Package initialization file
```

#### Commenting

Please follow [Google's guidelines for Python Styling](https://google.github.io/styleguide/pyguide.html). These comments are also used to automatically generate the documentation. For Visual Studio Code users, the [Python Docstring Generator](https://github.com/NilsJPWerner/autoDocstring) package may be helpful.

```{note}
Any docstrings parsed by `autosimple`, such as the functions in [](./usage/index.md), are parsed as markdown. Docstrings parsed by autoapi, such as in [](#cambrian.envs.MjCambrianEnv), are parsed as reStructuredText.
```

### Editing the Documentation

If you plan on editing the documentation pages (i.e. adding a tutorial or fixing an existing page), please visit the `docs/` folder. The `docs/` folder is structured as follows:

```
docs/
├── _static/            # Static files that persist through the build process
│   ├── css/custom.css  # Custom css changes that are different from the default furo theme
│   └── ...             # Images, favicons, etc.
├── examples/           # Example tutorials for the ACI package
├── usage/              # Usage reference guide for the ACI package
├── setup.md            # Installation build instructions
├── contributing.md     # Contributing tab with instructions on how to contribute to the repo
├── conf.py             # Settings related to extensions, themes, etc.
└── index.md            # The "home" page
```

Please try to maintain the file structure as described above. All tabs with only a single page (i.e. background or contributing), should have their `markdown` file with in the `docs/` folder. If the tab has or will have multiple pages (i.e. a tutorials tab), create a folder titled the same as that tab. To add pages, insert the name of the file without the `.md` extension within the table of contents inside `index.md`. Each folder should also contain an `index.md` used as the home page of that tab.

Markdown files are converted to reStructuredText by `myst_parser` which is used by the documentation package [Sphinx](https://www.sphinx-doc.org/en/master/). Both Markdown and reStructuredText have their advantages and disadvantages, `myst_parser` allows us to use the easy to understand `markdown` language but also compile the content down to something Sphinx understands. To see additional features of the `myst_parser` library, please visit their [website](https://myst-parser.readthedocs.io/en/latest/).

## Building the Documentation

There are multiple ways to build sphinx documentation. The easiest is using the `Makefile` or `make.bat` file provided directly in this repository. You will need to install all the necessary dependencies and build the html pages. To do that, run the following commands:

```bash
cd docs
make clean html
```

To view the build, go to your browser, and open the `index.html` file located inside `docs/build/html/`.

`sphinx-autobuild` is also extremely easy to use and will automatically build the html pages when a change is made. See their [PyPI page](https://pypi.org/project/sphinx-autobuild/). TL;DR: `pip install sphinx-autobuild` and run `sphinx-autobuild . build`.
