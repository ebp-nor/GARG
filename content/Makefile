WEBSITE_DIR=../dist

COURSE_PAGES=info/index.html \
	info/schedule.html \
	info/style.css

QUARTO_PDF_FILES=users/percyfal/lectures/wrightfisher.pdf \
	users/percyfal/lectures/coalescent.pdf \
	users/percyfal/lectures/simulation.pdf \
	users/percyfal/lectures/forward-simulation.pdf

all: $(QUARTO_PDF_FILES) $(COURSE_PAGES)
	rsync -av $(COURSE_PAGES) $(WEBSITE_DIR)


# cd users/percyfal, run quarto preview --port 9999
# NB: assumes local node installation of decktape
DECKTAPE=./node_modules/.bin/decktape
users/percyfal/%.pdf: users/percyfal/%.qmd
	@echo "Rendering $< to $@"
	@${DECKTAPE} http://localhost:9999/$*.html $@

info/%.html: info/%.qmd info/style.css
	@echo "Rendering $@ from $<"
	@quarto render $< --to=html -M embed-resources:true
