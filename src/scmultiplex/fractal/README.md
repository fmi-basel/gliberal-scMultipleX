# Using scMultipleX Fractal task

Fractal comes with extra dependencies. We're currently working on separating the helper functions into its own library, so that one doesn't need to depend on the full fractal-tasks-core package. In the meantime, the extra dependencies are optional. Thus, to install them, one needs to specify them:
```
cd scmultiplex
pip install ".[fracta]"
```

Or, once this package is on pypi:
```
pip install "scmultiplex[fracta]"
```

To collect the task in Fractal, one can load the Python wheel (see instructions below):
```
fractal task collect /path/to/scmultiplex-version-details.whl --package-extras fractal
```

After that, it's available in Fractal as "scMultipleX Measurements".

### Developer info
To create a new Fractal task, one needs to create a linux executable (e.g. a Python file with a `if __name__ == "__main__":` section) and this executable needs to follow the Fractal standards on how to read in inputs & store outputs ([see details here](https://fractal-analytics-platform.github.io/fractal-tasks-core/task_howto.html)). A typical Fractal task uses pydantic for input validation.

To make the task installable by a Fractal server, there needs to be a `__FRACTAL_MANIFEST__.json` file in the src/scmultiplex folder. This file contains a list of all available tasks, their default parameters as well as the relative path to the executable.

The manifest needs to be included when a package is built. 

For local creation of a Python whl, it means that the setup.cfg contains the following:
```
[options.package_data]
scmultiplex = __FRACTAL_MANIFEST__.json
```
Afterwards, the wheel can be built using `python -m build` and collected by Fractal using the command line client:
```
fractal task collect /path/to/scmultiplex-version-details.whl --package-extras fractal
```

(you may need to `pip install build` to run the wheel creation)

**Working with a Fractal task in development**
The above instructions work well to install the Fractal task as it is available in the package. If you want to run a task through Fractal server that you keep changing, it's not advisable to use the fractal task collection, but instead manually register your task.

For that purpose, create a Python environment that the task runs in (with all dependencies installed) and then use manual task registration pointing to the task Python file that you're working with. [See here for an example](https://github.com/fractal-analytics-platform/fractal-demos/tree/d241c7e29e5016bca6e0fd7647f44947e1501509/examples/08_scMultipleX_task).
