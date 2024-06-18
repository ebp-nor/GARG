# Genealogies and Ancestral Recombination Graph • Course materials

Course materials by Per Unneberg <per.unneberg@scilifelab.se>.

## About

Course materials consist of [Quarto](https://quarto.org) slides and
[Jupyter](https://jupyter.org/) notebooks.

## Rendering Quarto output

In order to render Quarto output, follow the instructions below.

### Install Quarto

[Install Quarto](https://quarto.org/docs/get-started) version [Quarto\>=1.4.554](https://quarto.org/docs/download/).

### Install Quarto filters

To render properly you need to install a couple of Quarto filters:

```bash
quarto install extension pandoc-ext/diagram
quarto add percyfal/nbis-course-quarto
quarto add royfrancis/quarto-reveal-logo
```

### Install Conda dependencies

Install the Conda dependencies in `environment-dev.yaml` with the
command

```bash
conda env create -n garg --file environment-dev.yaml
conda activate garg
```

### Install custom R packages

Install custom R package with the make command

```bash
make install-R
```

### Preview

To preview material, run

```bash
quarto preview --port 8888
```

### Render

To render material, run

```bash
make render
```

This will embed figures and resources. Unfortunately, this
functionality is buggy and not all resources are embedded properly,
occasionally resulting in misplaced and mal-formatted figures.

### Render pdf

FIXME: render with decktape.
