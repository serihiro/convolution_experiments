.PHONY: lint docs

lint:
	flake8
	pydocstyle

all:
	@make lint

docs:
	sphinx-build -b singlehtml ./my_convolution/docs ./my_convolution/docs/_build
