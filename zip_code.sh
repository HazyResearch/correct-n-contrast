#!/bin/bash

set -e
cd $(dirname $0)
git checkout neurips21
cd ..
rm -f neurips_code_submission.zip
rm -rf neurips_code_submission/
rsync -avzh --exclude .git/ --exclude .ipynb_checkpoints/ --exclude sandbox/ --exclude "*.sh" --exclude .DS_Store --exclude .gitignore --exclude __pycache__/ \
  correct-and-contrast/ neurips_code_submission/
zip -r neurips_code_submission.zip neurips_code_submission/
