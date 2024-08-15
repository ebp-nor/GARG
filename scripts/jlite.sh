#!/bin/bash
SCRIPT_PATH="$(dirname -- "${BASH_SOURCE[0]}")"
SCRIPT_PATH="$(cd -- "${SCRIPT_PATH}" && pwd)"
if [[ -z "${SCRIPT_PATH}" ]] ; then
    exit 1
fi
ROOT_PATH=$(realpath "${SCRIPT_PATH}/..")

# Build Jupyter Lite
cd "${ROOT_PATH}/jlite" || exit 1
jupyter lite build --contents content --output-dir ../_site/jlite
