# HOWTO

## Quarto preview

To preview the site locally, perform the following steps:

1. Install R dependencies listed in `environment.yml`, possibly in a
Conda environment `garg`:

    ```bash
    conda create -n garg environment.yml
    ```

2. Install JupyterLite dependencies

    ```bash
    pip install -r jlite/requirements.txt
    ```

3. Download and install Quarto (<https://quarto.org/docs/download/>) and
   run

    ```bash
    quarto preview --port 8888
    ```

## Lecture notes on Google Drive

Add the **full uri** to lecture located on Google Drive to the
`link_slide` column of the Google Drive spreadsheet
`schedule-vertical`.

## JupyterLite

Add Jupyter Notebooks to the directory `jlite/content`. Add the name
of the notebook to the `link_lab` column of the Google Drive
spreadsheet `schedule-vertical`; doing so will automatically add the
correct link to JupyterLite and Google Colab to the contents page.

The module `jlite/content/workshop.py` contains code to setup
workbooks. Add a Workbook class and setup function to initialize. Quiz
questions can be defined in `jlite/content/quiz.yaml` and are loaded
automatically.
