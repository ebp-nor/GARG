PDF_FILES := lectures/simulation.pdf # \
	# lectures/estimation/estimation.pdf \
	# lectures/hypothesis-testing/hypothesis-testing.pdf \
	# lectures/linear-regression/linear-regression.pdf \
	# lectures/logistic-regression/logistic-regression.pdf \
	# lectures/regularization/regularization.pdf \
	# lectures/decision-trees/decision-trees.pdf \
	# lectures/ensemble-methods/ensemble-methods.pdf \
	# lectures/unsupervised-learning/unsupervised-learning.pdf \
	# lectures/feature-engineering/feature-engineering.pdf \
	# lectures/feature-selection/feature-selection.pdf \
	# lectures/evaluation/evaluation.pdf \
	# lectures/interpretability/interpretability.pdf \
	# lectures/production/production.pdf \
	# lectures/ethics/ethics.pdf \
	# lectures/advanced-topics/advanced-topics.pdf

all: $(PDF_FILES)
	rsync -av $(PDF_FILES) ../../dist/lectures/

preview: ## Preview Quarto material
	quarto preview --port 9999

render: ## Render Quarto material
	quarto render --profile embed

install-R:  ## Install additional R packages that require manual installation
	R -e "library(devtools); devtools::install_github('percyfal/pgip-r', dependencies=FALSE)"

.phony: install-R

%.pdf: %.qmd
	@echo "Rendering $< to $@"
	@./node_modules/.bin/decktape http://localhost:9999/$*.html $@
