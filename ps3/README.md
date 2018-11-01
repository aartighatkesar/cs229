# CS229 Fall 2018 Problem Set #3


## Setup for Written Parts

1. We have provided a LaTeX template in the `tex` directory to make it easy to typeset your homework solutions.
2. Every problem has its own directory (*e.g.,* `01-simple_nn` for Problem 1).
3. Every subproblem has two files within the parent problem’s directory:
  - The problem statement, *e.g.* `01-simple_nn/01-sigmoid.tex` for Problem 1(a)). You do not need to modify this.
  - Your solution, *e.g.* `01-simple_nn/01-sigmoid_sol.tex` for your solution to Problem 1(a). You will need to modify these files (and the source files in `src` for coding parts).
4. You can use the given `Makefile` to typeset your solution, or use an editor with built-in typesetting such as TeXShop (comes free with the standard [LaTeX distribution](https://www.latex-project.org/get/)) or [Texpad](https://www.texpad.com/) (separate download, not free).


## Setup for Coding Parts

1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. cd into src, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229`
3. Run `source activate cs229`
  - This activates the `cs229` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229/bin/python`
  - Select `OK` then `Apply`
5. Submit to Gradescope
  - Run the `make_zip.py` program to create a zip for Gradescope submission
  - When you submit to Gradescope, the autograder will immediately check that your code runs and produces output files of the correct name, output format, and shape.
