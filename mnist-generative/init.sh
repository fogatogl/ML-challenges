#!/bin/bash
# init.sh — Onyxia "init script" for the mnist-generative project,
# nested inside the fogatogl/ML-challenges repo.
#
# HOW TO USE:
#   1. Push this file (and the rest of the mnist-generative/ folder) into
#      your ML-challenges repo.
#   2. When launching the "service-pytorch" (VSCode) on Onyxia, paste the
#      RAW url of this file into the service's "Init script" field, e.g.
#      https://raw.githubusercontent.com/fogatogl/ML-challenges/main/mnist-generative/init.sh
#   3. Every fresh service will clone/pull the repo and install deps automatically.
#
# AUTH NOTE: this script does NOT embed a token. The first time you clone
# in a fresh service, git will prompt for a username + password — use your
# GitHub username and a Personal Access Token as the password (see SETUP.md,
# step "Authentication"). `credential.helper store` then caches it for the
# rest of that service's life. If you want zero manual steps, see the
# "Vault-injected token" section of SETUP.md instead.
#
# EDIT THE TWO VARIABLES BELOW before pushing.

set -euo pipefail

GIT_USER_NAME="fogatogl"
GIT_USER_EMAIL="fogatogl05@gmail.com"
REPO_URL="https://github.com/fogatogl/ML-challenges.git"

WORKDIR="${HOME}/work"
REPO_DIR="${WORKDIR}/ML-challenges"
PROJECT_DIR="${REPO_DIR}/mnist-generative"

echo "== [1/4] git identity =="
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
# credential.helper store avoids retyping a PAT every push in the same service.
git config --global credential.helper store

echo "== [2/4] clone or pull repo =="
mkdir -p "${WORKDIR}"
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "== [3/4] python dependencies =="
cd "${PROJECT_DIR}"
pip install --no-cache-dir --user -r requirements.txt

echo "== [4/4] S3 / MinIO check =="
# Onyxia usually pre-aliases mc as "s3" using injected AWS_* env vars.
# This just confirms it — it does not create a new alias.
if command -v mc >/dev/null 2>&1; then
  mc alias list || echo "  -> no alias found yet; see SETUP.md for manual config"
else
  echo "  -> mc CLI not found on PATH; see SETUP.md for boto3 fallback"
fi

echo "Init complete. Repo at: ${REPO_DIR}"
echo "Project at:            ${PROJECT_DIR}"
