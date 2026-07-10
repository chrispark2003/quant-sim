# Deploying quant-sim to EC2

## One-time instance setup (already done if you followed the manual setup)

See `deploy/setup_ec2.sh` -- installs system packages, builds the venv,
installs requirements, fetches historical data, and installs the systemd
units (`quant-sim-live`, `quant-sim-api`, `quant-sim-dashboard`).

## One-time: converting to a git-tracked deploy

`quant-sim` is a **private** repo, so the instance needs its own credential
to fetch -- it can't rely on password auth (GitHub disabled that for git
operations) and shouldn't share your personal machine's cached credentials.
Use a **deploy key**: a dedicated SSH keypair, read-only, scoped to just
this repo.

If `~/quant-sim` on the instance was set up via `rsync` (not `git clone`),
convert it to a real git checkout so the CI/CD workflow can `git pull`
instead of re-syncing files on every push:

```bash
ssh -i ssh/quant-sim-key.pem ubuntu@<EC2_HOST>

# 1. Generate a dedicated deploy key (no passphrase -- it must run unattended)
ssh-keygen -t ed25519 -f ~/.ssh/quant_sim_deploy -N "" -C "quant-sim-ec2-deploy"
cat ~/.ssh/quant_sim_deploy.pub
```

Copy that public key's output, then in the GitHub web UI: repo -> **Settings
-> Deploy keys -> Add deploy key** -> paste it -> leave "Allow write access"
**unchecked** (the instance only ever needs to fetch, never push).

Back on the instance, point SSH and git at the new key and switch the
remote from HTTPS to SSH:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/quant_sim_deploy
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

cd ~/quant-sim
git init
git remote remove origin 2>/dev/null   # in case an earlier attempt left one
git remote add origin git@github.com:chrispark2003/quant-sim.git
ssh -T git@github.com                  # accept the host-key prompt once; should greet you by repo
git fetch origin main
git reset --hard origin/main
```

`git reset --hard` only touches tracked files -- it will not delete `.env`,
`data/state/` (ledger + kill switch), `data/*.duckdb`, or `.venv/`, since
those are gitignored and untracked. Confirm afterward:

```bash
git status              # should show "working tree clean"
cat .env | grep -c '='   # your keys should still be there
ls data/state/           # ledger.json / kill_switch.json should still exist
```

## GitHub Actions secrets

In the repo: **Settings -> Secrets and variables -> Actions -> New repository secret**

| Secret | Value |
|---|---|
| `EC2_HOST` | the instance's public IP (e.g. `54.90.250.235`) |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | full contents of `ssh/quant-sim-key.pem`, including the `-----BEGIN ... KEY-----` / `-----END ... KEY-----` lines |

Add these via the GitHub web UI (or `gh secret set <name> < path/to/file`
for `EC2_SSH_KEY` if you have the GitHub CLI authenticated) -- never commit
the private key to the repo.

## What the workflow does

On every push to `main` (`.github/workflows/deploy.yml`):

1. **test job**: installs base requirements and runs the pytest suite on a
   throwaway GitHub-hosted runner. Deploy only proceeds if this passes.
2. **deploy job**: SSHes into the instance, `git reset --hard` to the new
   commit, reinstalls `requirements.txt` only if that file changed in the
   diff, then restarts all three systemd units and verifies they came back
   up with `systemctl is-active`.

If the EC2 instance's public IP changes (e.g. after a stop/start without an
Elastic IP), update the `EC2_HOST` secret before the next push.
