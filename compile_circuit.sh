#!/usr/bin/env bash
# Adapted from rqs-svg 10a4da5b29a65f3eec0f556344ce06c3ff24072a.
# Changes: honor conventional host compilation and linking flag overrides.
set -euo pipefail
: "${CC:=gcc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_QCS_H="${SCRIPT_DIR}/include"

# Flag variables are whitespace-separated; pass arguments containing spaces in "$@".
read -r -a cpp_flags <<< "${CPPFLAGS:-}"
read -r -a c_flags <<< "${CFLAGS:-}"
read -r -a ld_flags <<< "${LDFLAGS:-}"
"$CC" -fPIC -shared -I"${DIR_QCS_H}" -std=c11 "${cpp_flags[@]}" "${c_flags[@]}" "$@" "${ld_flags[@]}"
