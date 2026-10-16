#!/bin/sh

cd "${0%/*}"

exec find . \
    -type f \
    ! -path '*/venv/*' \
    \( -name \*.py -o -name \*.ipynb \) \
    -exec sed -nE 's/^[^a-z]*(from|import) ([a-zA-Z0-9_]+).*$/\2/p' '{}' + \
    | sort -u \
    | awk -F== '
        BEGIN {
            while ("pip freeze" | getline)
                v[$1] = $2
            close("pip freeze")
        }
        {
            if ($0 in v)
                printf "%s==%s\n", $0, v[$0]
        }
    ' \
    >requirements.txt