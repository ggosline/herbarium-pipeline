# Cloud Setup — Step-by-Step Guide

This guide takes you from "no cloud accounts" to "first successful training run" using the **Cloud** tab in the herbarium pipeline. Follow it once; afterwards every project just clicks **Provision** and runs.

You will create accounts at three services. None of them require a credit card to *start* (RunPod and Cloudflare both have free tiers; WandB is free). RunPod will need a card before you can actually launch a GPU.

| Service | Cost | What it gives you |
|---|---|---|
| **RunPod** | Pay-per-minute GPU (~$0.30–$1/hr) | The GPU that runs training |
| **WandB** *(optional)* | Free for academic use | Live training graphs in your browser |
| **Cloudflare R2** | 10 GB free, then ~$0.015/GB/month | Persistent storage for project archives + a shared cache that makes every fresh pod ~50× faster to set up |

Total time: **~30 minutes** the first time, **~2 minutes** for every subsequent project.

---

## 1. RunPod (the GPU host)

### 1a. Create an account and add billing

1. Go to <https://runpod.io>, sign up.
2. **Settings → Billing** → add a payment method and load $10–20 to start.

### 1b. Generate an SSH key on your computer

The pipeline talks to your pod over SSH. You need a passwordless key-pair that RunPod knows about.

On Linux/Mac/WSL:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_herbarium -N ""
```

That creates two files:
- `~/.ssh/id_ed25519_herbarium` — private key (stays on your computer)
- `~/.ssh/id_ed25519_herbarium.pub` — public key (gets uploaded to RunPod)

### 1c. Register the public key with RunPod

1. RunPod console → **Settings → SSH Public Keys → Add Key**
2. Paste the entire contents of `id_ed25519_herbarium.pub` (one long line starting with `ssh-ed25519`).
3. Save.

### 1d. Generate a RunPod API key

1. RunPod console → **Settings → API Keys → + Create API Key**
2. Permissions: **Read & Write** (default).
3. Copy the key — it starts with `rpa_…`. **You only see it once.**

---

## 2. WandB (optional, but recommended for training)

Without WandB, training metrics are written to a CSV on the pod. With WandB, you get live charts in your browser while training runs.

1. Sign up at <https://wandb.ai>.
2. Visit <https://wandb.ai/authorize> — copy the API key it shows.

That's it.

---

## 3. Cloudflare R2 (persistent storage + shared cache)

R2 is Cloudflare's S3-compatible object storage with **zero egress fees**, which makes it ideal for moving large files in and out of RunPod cheaply.

### 3a. Create a Cloudflare account

1. Sign up at <https://dash.cloudflare.com>. R2 is in the left-hand sidebar.
2. Click **R2** → **Subscribe to R2** (10 GB free; payment method required but you won't be charged at low usage).

### 3b. Create the two buckets

In the R2 dashboard, click **Create bucket** twice:

| Bucket name | Purpose |
|---|---|
| `herbarium-backup` | One subfolder per project; holds checkpoints, the resized image set, and predictions so you can delete the network volume between projects |
| `herbarium-cache` | Shared by **every** project: cached PyPI wheels (~2 GB) and Hugging Face model weights (~2 GB) so fresh pods skip the slow downloads |

Leave region and other settings at defaults.

### 3c. Generate an R2 API token

1. R2 dashboard → **Manage R2 API Tokens** → **Create Account API Token**
2. **Permissions**: *Object Read & Edit* — Cloudflare's UI labels write permission as "Edit", not "Write". Picking just "Read" leaves you with a read-only token and uploads will 403.
3. **Specify buckets**: *Apply to all buckets* (or limit to the two above)
4. **TTL**: Forever (or whatever you prefer)
5. Click **Create**. You will now see:
   - **Access Key ID** — copy
   - **Secret Access Key** — copy (only shown once)
   - **Endpoint URL** — looks like `https://<32-char-hex>.r2.cloudflarestorage.com`

The 32-char hex prefix in the endpoint is your **Account ID**. Copy that too (you'll also see it on the right-hand side of the R2 dashboard).

You should now have **four strings** noted down:
- Account ID (32-char hex)
- Access Key ID
- Secret Access Key
- (Bucket name — `herbarium-backup`)

---

## 4. Enter all credentials into the WebUI

1. Launch the WebUI (`python /path/to/Pipeline/herbarium_pipeline_webui.py`).
2. Open the **⚙ Setup** tab (the first one).
3. Each section has a status pill (✓ ready / not set) and its own *Save* button:

| Card | What to paste |
|---|---|
| **Local environment** | Nothing — just confirms your Python / NiceGUI / GPU / disk are usable. Click *Re-check* if the readout looks stale. |
| **RunPod** | The `rpa_…` API key from step 1d, plus the path to your **SSH private key** (`~/.ssh/id_ed25519_herbarium` — the file *without* `.pub`). |
| **WandB** | Key from step 2 (or leave blank to skip wandb). |
| **Cloudflare R2** | Account ID, Access Key, Secret, default bucket = `herbarium-backup`. |

Each *Save* button stores its secret in your **OS keyring** (Windows Credential Manager / macOS Keychain / Linux Secret Service). Nothing is written to plain files. When all four cards show ✓, you can switch to the ☁ Cloud tab and forget Setup exists.

---

## 5. First project — provision and run

1. **Top of the page** (above the tabs): set a **Project name** (e.g. `MyFirstProject`) and click **Apply paths**.
2. Open the **☁ Cloud** tab. It's organised as four cards:

| Card | What it does |
|---|---|
| **Pod** | Status row (pod id / cost / current step) + lifecycle buttons: *Provision + sync code*, *Cancel running step*, *Terminate (keep volume)*, *Terminate + delete volume*. Banner at the top warns if Setup credentials are missing. |
| **Configure run** | *Project* (mirrors the top-of-page field) and *Purpose*. The **Advanced** expander has GPU type override (a searchable dropdown — click ↻ to load the live list from RunPod), datacenter, volume size. **Download caps** is a separate expander. |
| **Pipeline** | The Upload-DwC-A button + the five step buttons in flow order: Setup → Download → Prep → Train → Identify. A yellow note above them explains that each step pulls its parameters from the matching stage tab (hover any step button — its tooltip names the exact source tab and fields). The **Repair R2 wheel cache** button and **⚠ Danger zone** (wipe pod-side data) live here too. |
| **Results & Archive** | *Download results*, *Pull images_1024*, *Archive project to R2*, *Restore project from R2*. |

3. In **Configure run**, set:
   - **Purpose**: `light` (NVIDIA L4 — cheap, for download / prep / identify) or `train` (RTX 4090 — long GPU run).
   - **Advanced → Datacenter**: pick one that's geographically near you. *Once chosen, it's locked* — your network volume can only live in that DC.
   - **Advanced → Volume size**: 80 GB is fine for ≤10k images.
   - **Advanced → GPU type** (optional override): leave blank to use the purpose default. If you want a specific GPU, click the **↻** button to load RunPod's current list, then pick from the dropdown — RunPod's web console shows marketing names ("A100 SXM 80GB") that are *not* what the API accepts ("NVIDIA A100-SXM4-80GB"). For a CLI version of the same list run `python tools/list_runpod_gpus.py`.

4. **Pod card → Provision + sync code**. Logs appear in the right-hand panel:

   ```
   Creating network volume (80 GB, EUR-IS-1)...
     volume xxx created
   Creating NVIDIA L4 pod in EUR-IS-1...
     pod yyy created ($0.43/hr), waiting for SSH...
     pod ready @ 1.2.3.4:22001
   Syncing 24 files → /workspace/Pipeline
   Pushed wandb key → /workspace/.wandb_key
   Pushed rclone.conf → /workspace/.config/rclone/
   ```

5. **Pipeline card → Setup**. **First time:** ~5–10 min (downloads torch + nvidia-* + DALI etc., then pushes them to the R2 cache). **Every subsequent setup** on any project: ~1 min (pulls from R2 cache).

6. **Pipeline card → Download → Prep → Train → Identify** in order. Configure each step's parameters on its own stage tab first — Train in particular pulls every knob (model, batch size, epochs, LRs, geo, hierarchy, WandB run name) from the **4 Train** tab.

7. When done: **Results & Archive → Archive project to R2** to back up checkpoints + images, then **Pod → Terminate** (or let the idle watchdog do it after 1 hr of inactivity).

---

## What gets saved where

| Where | What | Cost |
|---|---|---|
| OS keyring (your computer) | RunPod / WandB / R2 keys | free |
| RunPod network volume (per-project) | Your raw + processed images, checkpoints, specsin.csv | ~$0.07/GB/month while it exists |
| `r2:herbarium-cache` (shared) | pip wheels + HF model weights | a few cents/month total |
| `r2:herbarium-backup/<project>/` (per-project) | Latest checkpoint + tarred images for restore | a few cents/GB/month |
| RunPod pod itself | Nothing persistent — wiped on terminate | $0 when terminated |

The whole point of R2 is so you can **terminate volumes between projects** without losing work. Restore a project later with **Restore project from R2** on the Cloud tab.

---

## Troubleshooting

### "toomanyrequests: You have reached your unauthenticated pull rate limit" at provision time
RunPod's host hit Docker Hub's anonymous pull cap. The base image (`runpod/pytorch:…`) is on Docker Hub. Wait an hour and retry, or terminate and re-provision (you may land on a different host).

### Pod stuck "waiting for SSH" past 5 minutes
Check the RunPod console — is the pod's status `RUNNING`? If yes, look at "Connect → SSH over exposed TCP". If grayed out / proxy-only, the image you used doesn't ship `sshd`. Stick with the default image.

### `uv sync` is extremely slow (KB/s, not MB/s)
PyPI's CDN is sometimes slow from a particular RunPod datacenter. The first setup pays this cost; afterwards the R2 cache short-circuits it. If it never finishes:
- SSH into the pod and check `ss -i` — if a TCP connection to `151.101.x.x` (Fastly/PyPI) is alive but slow, just wait.
- If it's truly stuck, terminate and try a different datacenter (but remember: your network volume is locked to its DC, so you'll create a fresh volume).

### `cache_push` / Archive-to-R2 fails with `403 Forbidden / AccessDenied`
Your R2 API token doesn't have write permission on the target bucket. Cloudflare labels write permission as **"Object Read & Edit"** (not "Write") in the token-creation UI — easy to miss. Recreate the token with that permission and "Apply to all buckets" (or explicitly include both `herbarium-backup` and `herbarium-cache`), then paste the new keys into the WebUI's R2 section and re-Provision.

### "A cloud step is already running" when nothing seems to be running
The previous step crashed but the task tracker didn't clear. Click **Cancel running step** on the Cloud tab.

### Pod was created but provision crashed before SSH came up
The orchestrator now persists `pod_id` immediately, so the next **Provision** click will reuse the orphaned pod (or report it). If reuse fails, terminate the orphan in the RunPod console first.

### `400 Bad Request — value must be one of 'NVIDIA GeForce RTX 4090', …` on Provision
The GPU id you typed into the **GPU type** override doesn't match RunPod's API enum. Their web console shows marketing names ("A100 SXM 80GB"); the API only accepts model numbers ("NVIDIA A100-SXM4-80GB"). Two fixes:

1. Open Configure run → Advanced → click the **↻** next to the GPU dropdown and pick from the list, or
2. Clear the override entirely so the purpose default kicks in.

For a quick CLI listing of every accepted id:
```bash
python tools/list_runpod_gpus.py            # all 49
python tools/list_runpod_gpus.py --grep H100  # filter
```

The orchestrator also pre-validates whatever you submit and suggests the three closest matches if it doesn't recognise the id, so a typo gets a useful error rather than the truncated 400.

### Setup pushes wheels to R2 but the next pod still re-downloads everything
Symptom: in `r2:herbarium-cache/uv/` you only see a small `simple-v21/pypi/` tree (~25 entries) and no `wheels-v6/` or `archive-v0/` directory. This was a real bug in versions of `pod_bootstrap.sh` before commit `af57303`: `uv cache prune --ci` ran *before* `cache_push` and stripped the wheel binaries from the local cache, so only the index metadata reached R2. The fix removed the prune step.

To repair an existing pod whose R2 cache was emptied this way, click the **Repair R2 wheel cache** button (Pipeline card on the Cloud tab) — it forces `uv sync --frozen --reinstall` to re-download every wheel into the volume cache, then pushes them to R2. One painful PyPI fetch (~5–30 min depending on datacenter) buys back warm-cache pods for every future provision.

---

## Costs in practice

A typical 15k-image, 19-epoch training run on a single RTX 4090 costs **~$2–4** in pod time. A month of "between projects" R2 storage for one archived project is usually **<$0.50**. Network volume left running unused is the main money pit — use the **idle watchdog** (default: terminate after 1 hr idle) and the **Archive to R2** flow to keep this under control.
