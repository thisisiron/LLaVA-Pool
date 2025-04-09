.PHONY: style quality fixup modified_only_fixup

# Set PYTHONPATH to source directory
export PYTHONPATH = .

# Directories to check for style
check_dirs := llava src scripts tests utils

# Exclude certain directories from style checking
exclude_folders := ""

# Check only modified files and fix style issues
modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		python -m ruff check $(modified_py_files) --fix --exclude $(exclude_folders); \
		python -m ruff format $(modified_py_files) --exclude $(exclude_folders); \
	else \
		echo "No library .py files were modified"; \
	fi

# Run style checks on all code without modifying
quality:
	python -m ruff check $(check_dirs) --exclude $(exclude_folders)
	python -m ruff format --check $(check_dirs) --exclude $(exclude_folders)

# Run style checks and fix issues on all code
style:
	python -m ruff check $(check_dirs) --fix --exclude $(exclude_folders)
	python -m ruff format $(check_dirs) --exclude $(exclude_folders)

# Super fast fix and check target that only works on relevant modified files
fixup: modified_only_fixup

# Run tests
test:
	pytest -v tests/

# Install development dependencies
dev-install:
	pip install -r requirements-dev.txt 