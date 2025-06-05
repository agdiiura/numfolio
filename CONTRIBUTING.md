## Development 👩‍💻

### GitFlow

The development cycle follows a *[GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)*
paradigm. The releases are contained in the `master` branch, which
must contain stable code. The main development branch
is `develop`. Pull requests are used to align feature branches to `develop` and
`develop` to `master`. Automatic tests are run to assess the
code quality, see [`tests`](tests) folder.

### Code style

The style and the quality of the code is managed
by [`ruff`](https://docs.astral.sh/ruff/). See  [`pyproject.toml`](pyproject.toml) for configuration.
A plugin for a documentation is also used, the `md` paradigm for [`mkdocs`](https://www.mkdocs.org/)
has been chosen. See [`docs`](docs) folder for documentation-related issues.
The import rules are managed through [`isort`](https://pycqa.github.io/isort/), see [`pyproject.toml`](pyproject.toml) file, section
`[tool.isort]`.

The style syntax rules are imposed with the help of [`pre-commit`](https://pre-commit.com/) tool.
Those are specified in the [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file.
To initialize the tool run
```bash
pre-commit install
```
A common error on `macOS` is related to `UTC-8` format. See [this](https://stackoverflow.com/questions/60557160/python3-8-fails-with-fatal-python-error-config-get-locale-encoding) issue
on StackOverflow. To update the tool use `pre-commit autoupdate`.


### Contributors

Authors and contributors have been collected in [`AUTHORS.md`](docs/docs/authors.md) file.
