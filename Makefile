PYTHON = python
RUN = uv run

all: install

install:
	uv sync

run:
	$(RUN) $(PYTHON) -m src $(ARGS)

debug:
	$(RUN) $(PYTHON) -m pdb -m src $(ARGS)

lint:
	$(RUN) flake8 . --exclude=.venv,__pycache__,data
	$(RUN) mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(RUN) flake8 . --exclude=.venv,__pycache__,data
	$(RUN) mypy . --strict --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d \( -name .mypy_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv
	
.PHONY: all install run debug clean fclean lint lint-strict
