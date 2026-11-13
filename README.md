# ACE-Step 1.5 XL on Modal: Music Generation Test

A small script that generates a full song (vocals, lyrics and arrangement) with
[ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) on a rented cloud GPU via
[Modal](https://modal.com). Run one command and get a FLAC and a 320 kbps MP3 back on your machine.

> ⚠️ **This is a test / proof of concept, not a production build.**
> It uses a single hard-coded song, runs one generation per invocation, has no
> queueing, retries, auth, monitoring or cost controls, and installs dependencies
> at image-build time from unpinned upstream sources. Treat it as a starting point
> for experimenting, not something to serve users from.

## 🎧 Sample output

The song the script generates ("No Slow Fade", original lyrics, alternative R&B / melodic hip-hop with Afro-fusion touches):

**[▶ samples/no_slow_fade.mp3](samples/no_slow_fade.mp3)**

(Click through and use GitHub's file preview to listen.)

## What it does

1. Builds a container image on Modal (PyTorch 2.10 / CUDA 12.8 base, ffmpeg, a venv, ACE-Step 1.5 and its dependencies).
2. Downloads the model weights into a persistent Modal Volume (first run only):
   - DiT: `ACE-Step/acestep-v15-xl-sft`
   - LM: `ACE-Step/acestep-5Hz-lm-4B`
3. Generates a 150-second song on an A100-80GB with a fixed seed.
4. Converts the output to a 320 kbps MP3 and downloads both files to `./generated/`.

## Why Modal?

[Modal](https://modal.com) is a good fit for this kind of experiment:

- **No GPU needed locally.** The XL model plus a 4B language model needs a big GPU (this script requests an A100-80GB). Modal rents one by the second.
- **Pay only while it runs.** Serverless billing means no idle instance sitting on your card. Modal has a free monthly credit allowance, but check [current pricing](https://modal.com/pricing), because A100-80GB time is not cheap.
- **Infrastructure as Python.** The container image, GPU choice, volumes and entrypoint are all defined in one Python file. No Dockerfile, Kubernetes or SSH required.
- **Persistent volumes.** Model weights (many GB) are downloaded once and cached, so later runs skip the download.

Any GPU host will work if you adapt the script. Modal just makes the test low-friction.

## Prerequisites

- Python 3.10+
- A free [Modal account](https://modal.com/signup)

## Setup and run

```bash
# 1. Clone
git clone https://github.com/edwardbudaza/ace-step-modal-test.git
cd ace-step-modal-test

# 2. Install the Modal client
pip install modal

# 3. Authenticate with Modal (opens a browser)
modal setup

# 4. Run the test
modal run ace_step_modal_test.py
```

Outputs land in `./generated/`:

- `no_slow_fade.flac` (lossless)
- `no_slow_fade.mp3` (320 kbps)

### What to expect

| Run        | Roughly what happens                                                        |
| ---------- | --------------------------------------------------------------------------- |
| First run  | Image build (a few minutes), then model download (many GB), then generation |
| Later runs | Image and weights are cached, so you only wait for generation               |

Exact times depend on Modal's load and the GPU you get. Generation time is printed at the end.

## Customising

Everything you'd want to change is near the top of `ace_step_modal_test.py`:

| Setting              | Where                                               | Notes                                                      |
| -------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| Song style           | `CAPTION`                                           | Free-text description of genre, instruments and vocals     |
| Lyrics               | `LYRICS`                                            | Uses `[Verse]`, `[Chorus]`, etc. section tags              |
| Tempo / key / length | `bpm`, `keyscale`, `duration` in `GenerationParams` | Duration is in seconds                                     |
| Quality vs. speed    | `inference_steps`                                   | Fewer steps is faster, with lower quality                  |
| Reproducibility      | `SEED`                                              | Same seed and settings give a similar result               |
| GPU                  | `gpu="A100-80GB"`                                   | Change to another Modal GPU type to compare cost and speed |

## Troubleshooting

**`/opt/venv/bin/uv: not found` during image build**
The base image already ships `uv` in system site-packages, so a plain `pip install uv` into a venv created with `--system-site-packages` is skipped as "already satisfied". The script fixes this with `pip install --ignore-installed uv` and runs `uv` as `python -m uv`.

**Torch/CUDA version conflicts after `uv sync`**
ACE-Step pins its own torch version. If it clashes with the base image's PyTorch 2.10, remove `--system-site-packages` from the venv command and let `uv` install the pinned torch itself.

**vllm fails when initialising the language model**
In `generate_song()`, change `backend="vllm"` to `backend="pt"`. It's slower, but has fewer version constraints.

**Image build errors**
Inspect the logs with `modal image logs <image-id>`, using the ID printed in the error.

## Cost warning

This script spins up an A100-80GB with 16 CPUs and 64 GB RAM. Model downloads and generation have timeouts of 2 hours each, so a stuck run can cost real money. Keep an eye on the [Modal dashboard](https://modal.com/apps), and stop runaway apps with `modal app stop`.

## Not for production

If you want to build on this, you'd at minimum want to:

- Pin the ACE-Step commit and all dependency versions
- Load the models once in a class-based Modal function (`@app.cls`) instead of on every call
- Expose it behind an authenticated web endpoint with request validation
- Add queueing, retries, logging and spend limits
- Store outputs in object storage rather than returning raw bytes

## Licences and credits

- Model and code: [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5). Check their repository and model cards for licence terms before using generated audio commercially.
- The lyrics and caption in this repo are original.
- `MIT LICENSE`
