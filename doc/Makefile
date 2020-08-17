# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Place the current build in the gh-pages branch.
ghpages: | build
	$(eval stash_hash := $(shell git stash create))
	$(eval commit_hash := $(shell git rev-parse HEAD))
	git reset --hard HEAD
	git checkout gh-pages
	git pull
	cp -r ./build/html/* ../
	git add --all
	git commit -m "Update documentation.\n\nFrom reversion: $(commit_hash)"
	git push
	git checkout -
	if [ -n "$(stash_hash)" ]; then git stash apply "$(stash_hash)"; fi
	cd doc

build:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
