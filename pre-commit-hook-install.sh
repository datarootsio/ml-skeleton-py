#!/bin/sh
cd "$(dirname "$0")"
touch .git/hooks/pre-commit
rm .git/hooks/pre-commit
pushd .git/hooks
ln -s ../../pre-commit-hook.sh ./pre-commit
popd
