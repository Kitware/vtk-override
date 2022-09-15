# Simple makefile to simplify repetitive build env management tasks under posix

CODESPELL_DIRS ?= ./
CODESPELL_SKIP ?= "*.pyc,*.txt,*.gif,*.svg,*.css,*.png,*.jpg,*.ply,*.vtk,*.vti,*.vts,*.vtr,*.js,*.html,*.doctree,*.ttf,*.woff,*.woff2,*.eot,*.mp4,*.inv,*.pickle,*.ipynb,flycheck*,./.git/*,./.hypothesis/*,*.yml,./doc/_build/*,./doc/images/*,./dist/*,*~,.hypothesis*,./doc/examples/*,*.mypy_cache/*,*cover,./tests/tinypages/_build/*,*/_autosummary/*"


stylecheck: codespell lint

codespell:
	@echo "Running codespell"
	@codespell $(CODESPELL_DIRS) -S $(CODESPELL_SKIP)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle vtk_override --match='(?!coverage).*.py'

doctest:
	@echo "Runnnig module doctesting"
	pytest -v --doctest-modules vtk_override

lint:
	@echo "Linting with flake8"
	flake8 --ignore=E501 vtk_override tests

format:
	@echo "Formatting"
	black .
	isort .
