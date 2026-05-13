#!/bin/bash
set -e

scversion="v0.11.0"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    tarball="shellcheck-${scversion}.linux.x86_64.tar.xz"
    wget -qO "$tarball" "https://github.com/koalaman/shellcheck/releases/download/${scversion}/$tarball"
    echo "8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198  $tarball" | sha256sum -c -
    tar -xJvf "$tarball"
    rm "$tarball"

    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi


find . -name "*.sh" -not -path "./trc/*" -print0 | xargs -0 -I {} sh -c 'git check-ignore -q "{}" || shellcheck -s bash "{}"'
