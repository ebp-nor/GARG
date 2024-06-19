# HOWTO

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
