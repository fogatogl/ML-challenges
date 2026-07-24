# Phase 0 Setup — Onyxia / SSPCloud

One-time setup, then every future service boots ready in one click.

## 0. Authentication — GitHub token

GitHub no longer accepts your account password for git operations, so you
need a **Personal Access Token (PAT)**:

1. Go to `github.com/settings/personal-access-tokens/new` (fine-grained tokens).
2. Repository access -> *Only select repositories* -> `fogatogl/ML-challenges`.
3. Permissions -> **Contents: Read and write**.
4. Generate, and copy the token (`github_pat_...`) somewhere safe — GitHub
   only shows it once.

You have two ways to use it on Onyxia:

**A. Manual, once per service (default, simplest, no secrets in any script)**
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global credential.helper store   # caches it for this service's lifetime

git clone https://github.com/fogatogl/ML-challenges.git ~/work/ML-challenges
```
When prompted:
- Username: `fogatogl` (your GitHub username)
- Password: paste the PAT — not your GitHub account password

After this first clone, `git push`/`git pull` in the same service won't
prompt again (cached in `~/.git-credentials`). Because the service is
ephemeral, you'll re-enter it once per *fresh* service — that's expected,
not a bug, since we deliberately don't bake the token into `init.sh`.

**B. Fully automated via Onyxia's Vault secrets manager**
If retyping the token bothers you: in Onyxia, go to *My Account -> Vault*
(or the secrets manager in your deployment) and store the PAT as a secret,
e.g. key `GIT_TOKEN`. When launching the service, inject it as an
environment variable (Onyxia's service form has a "Vault secrets" /
environment variables section referencing your stored secret). Then change
`init.sh`'s clone line to use it instead of your PAT never touching the
script itself:
```bash
git clone "https://${GIT_USER_NAME}:${GIT_TOKEN}@github.com/fogatogl/ML-challenges.git" "${REPO_DIR}"
```
Only do this if you're comfortable with secret injection — for a solo
learning project, option A is fine and safer by default.

## 1. Put mnist-generative inside ML-challenges

Assumption: `ML-challenges` is your general repo for various projects, so
`mnist-generative` lives as a subfolder, not its own repo. If you'd rather
it be standalone, say so and the paths below just need `ML-challenges`
swapped for a dedicated repo.

```bash
cd ~/work/ML-challenges
mkdir -p mnist-generative
# copy in the project files (requirements.txt, init.sh, src/, scripts/, etc.)
git add mnist-generative
git commit -m "Add mnist-generative project skeleton"
git push
```

Then edit `init.sh` (already pointed at `fogatogl/ML-challenges` in this
copy):
- `GIT_USER_NAME` / `GIT_USER_EMAIL` -> your identity

Commit and push that edit too.

## 2. Launch the service with the init script

On the Onyxia UI, when configuring your `service-pytorch` (VSCode):
- GPU: request one (check the resource quota for your course/lab).
- **Init script**: paste the *raw* URL of `init.sh` from your pushed repo, e.g.
  `https://raw.githubusercontent.com/fogatogl/ML-challenges/main/mnist-generative/init.sh`
  (Onyxia runs this as a startup script — you don't need to run it by hand.)
  Note: if `ML-challenges` is private, a plain `raw.githubusercontent.com`
  URL won't be fetchable by Onyxia without auth — in that case skip the
  init-script field and just run it manually (below) after cloning once.

If you'd rather run it manually inside an already-running service instead of
setting the init script field:
```bash
cd ~/work/ML-challenges/mnist-generative   # after the manual clone in step 0
bash init.sh
```

## 3. Verify everything

```bash
cd ~/work/ML-challenges/mnist-generative
nvidia-smi
python scripts/check_env.py
```

You want to see: a GPU listed, all packages importing, torch reporting
`cuda available: True`, and a real matmul running on the GPU. If `mc alias
list` comes back empty, see step 4.

## 4. S3 / MinIO for checkpoints

Onyxia injects credentials as environment variables
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
and an S3 endpoint var — name varies slightly by deployment, check `env | grep AWS`).
The `mc` client is usually pre-aliased already; confirm with:

```bash
mc alias list
mc ls s3          # should list your personal/course bucket(s)
```

If no alias exists, configure one manually (adjust the alias name/endpoint
to match what `env | grep AWS` shows you):

```bash
mc alias set s3 https://minio.lab.sspcloud.fr \
  "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" \
  --api S3v4
```

Then push/pull checkpoints:
```bash
mc cp checkpoints/vae_epoch10.pt s3/<your-bucket>/mnist-generative/checkpoints/
mc cp s3/<your-bucket>/mnist-generative/checkpoints/vae_epoch10.pt checkpoints/
```

If you'd rather use Python directly instead of the CLI, `boto3` is already
in `requirements.txt` and points at the same injected env vars.

## 5. Daily workflow reminder

Because the service is ephemeral:
- Commit + push code after every meaningful change, not just at the end of a session.
- Push checkpoints to S3 at every eval, not just at the end of training.
- Everything under `data/`, `checkpoints/`, `runs/` is gitignored on purpose —
  those belong on S3, not in git history.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | Service launched without GPU resource | Relaunch requesting a GPU flavor |
| `pip install` reinstalls a CPU-only torch | `requirements.txt` pinned torch | Don't pin torch/torchvision — use the image's build |
| `mc alias list` is empty | Env vars not injected for this service type | Check `env \| grep AWS`, configure alias manually (step 4) |
| Init script didn't run | Pasted a non-raw GitHub URL | Must be the `raw.githubusercontent.com` link, not the `github.com/blob/...` page |
