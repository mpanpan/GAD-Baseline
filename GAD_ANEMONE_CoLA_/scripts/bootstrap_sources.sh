#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
for spec in "ANEMONE https://github.com/mpanpan/ANEMONE.git" "CoLA https://github.com/mpanpan/CoLA.git"; do
  read -r name url <<<"$spec"
  if [[ -d "external/$name/.git" ]]; then
    echo "Already exists: external/$name; unchanged"
  else
    git clone "$url" "external/$name"
  fi
  git -C "external/$name" rev-parse HEAD
done
printf '%s\n' 'Source clones completed; verify licenses, commits and actual code before porting.'
