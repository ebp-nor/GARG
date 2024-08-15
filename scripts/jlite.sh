#!/bin/bash
SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
SCRIPT_PATH="$(cd -- "${SCRIPT_PATH}" && pwd)"
if [[ -z "${SCRIPT_PATH}" ]] ; then
    exit 1
fi
ROOT_PATH=$(realpath "${SCRIPT_PATH}/..")

# Build Jupyter Lite
cd "${ROOT_PATH}/jlite" || exit 1
EXTRA_IGNORE_CONTENTS=".+\.vcz .+\.bcf.* __pycache__ /\.virtual_documents phased_vcfs.*"
jupyter lite build  --extra-ignore-contents "${EXTRA_IGNORE_CONTENTS}" --output-dir ../_site/jlite

# jupyter lite build --contents content --ignore-contents ${IGNORE_CONTENTS} --output-dir ../_site/jlite

#rsync -av "${ROOT_PATH}/jlite/content/ARG_workshop.py" "${ROOT_PATH}/_site/jlite/notebooks/ARG_workshop.py"
#rsync -av "${ROOT_PATH}/jlite/content/img" "${ROOT_PATH}/_site/jlite/notebooks/"
