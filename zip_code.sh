#!/bin/bash

set -e
cd $(dirname $0)
git checkout neurips_code
cd ..
rm -f neurips_code_submission.zip
rm -rf neurips_code_submission/
rsync -avzh --exclude .git/ --exclude .ipynb_checkpoints/ --exclude sandbox/ --exclude "*.sh" --exclude .DS_Store --exclude .gitignore --exclude __pycache__/ \
  slice-and-dice-smol/ neurips_code_submission/
zip -r neurips_code_submission.zip neurips_code_submission/
