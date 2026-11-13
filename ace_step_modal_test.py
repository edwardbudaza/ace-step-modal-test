import os
import time
import subprocess
from pathlib import Path

import modal


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "ace-step-music-test"
VOLUME_NAME = "ace-step-model-cache"

REPO_DIR = "/opt/ACE-Step-1.5"
CHECKPOINT_DIR = f"{REPO_DIR}/checkpoints"
OUTPUT_DIR = "/outputs"

VENV_DIR = "/opt/venv"
PYTHON = f"{VENV_DIR}/bin/python"

DIT_MODEL = "acestep-v15-xl-sft"
LM_MODEL = "acestep-5Hz-lm-4B"

SEED = 20261113

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)


# ============================================================
# ORIGINAL SONG: NO SLOW FADE
# ============================================================

CAPTION = """
Premium contemporary alternative R&B blended with melodic hip-hop
and subtle Afro-fusion. Soulful expressive male lead vocals, intimate
verses and a huge emotionally memorable melodic chorus. Deep warm
sub-heavy 808 bass, punchy syncopated hip-hop drums, crisp hi-hats,
subtle African percussion, lush Rhodes electric piano, atmospheric
synth pads, muted electric guitar accents, layered soulful backing
vocals, tasteful vocal runs and modern polished vocal production.
Sophisticated late-night atmosphere, dynamic arrangement, rich stereo
imaging, professional studio recording, warm analogue texture.
"""

LYRICS = """
[Intro]
Oh, yeah
No slow fade
We don't disappear tonight

[Verse 1]
City lights dancing on the glass
Another long night, moving way too fast
I used to chase things I couldn't hold
Trading my peace just to feel in control

Now I see clearly when the morning comes
Some things are better when you let them run
I got a heart that I won't betray
And I don't need to lose myself to find my way

[Pre-Chorus]
So if the world keeps turning
Let it turn around me
I've been waiting for a moment
Now the moment found me

[Chorus]
We don't do no slow fade
Burning like the lights on a summer highway
If we're going all in, let it feel this way
No looking back, no yesterday

We don't do no slow fade
Hands in the air while the whole world waits
If the night gets cold, we can make our own flame
No slow fade, no slow fade

[Verse 2]
Bassline rolling through the avenue
Every little heartbeat pulling me to you
No more pretending that I'm doing fine
I found a little freedom in the space of time

Old conversations in the rearview
New destination with a better view
I took the long road, learned the hard way
But every single scar got me here today

[Pre-Chorus]
So if the world keeps turning
Let it turn around me
I've been waiting for a moment
Now the moment found me

[Chorus]
We don't do no slow fade
Burning like the lights on a summer highway
If we're going all in, let it feel this way
No looking back, no yesterday

We don't do no slow fade
Hands in the air while the whole world waits
If the night gets cold, we can make our own flame
No slow fade, no slow fade

[Bridge]
Let the rhythm take control
Feel the fire in your soul
Even when the lights go down
We will never lose this sound

No fear, no chains
No more running from the rain
If we fall, we rise again
And we won't go quietly

[Final Chorus]
We don't do no slow fade
Burning like the lights on a summer highway
If we're going all in, let it feel this way
No looking back, no yesterday

We don't do no slow fade
Hands in the air while the whole world waits
If the night gets cold, we can make our own flame
No slow fade, no slow fade

[Outro]
No slow fade
No slow fade
We don't disappear tonight
"""


# ============================================================
# MODAL IMAGE
# ============================================================

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel",

        # Remove the Debian PEP 668 marker BEFORE Modal's
        # generated pip bootstrap layer runs.
        setup_dockerfile_commands=[
            "RUN find /usr -type f -name EXTERNALLY-MANAGED -delete"
        ],
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "python3-venv",
        "python3-full",
    )
    .run_commands(
        # Create an isolated environment while retaining the
        # preinstalled PyTorch/CUDA packages from the base image.
        "python -m venv --system-site-packages /opt/venv",

        # Upgrade pip/wheel inside the venv.
        "/opt/venv/bin/python -m pip install --upgrade pip wheel",

        # FIX: --ignore-installed forces uv INTO the venv even though a
        # copy already exists in the system site-packages. Without this,
        # pip says "already satisfied" and /opt/venv/bin/uv is never created.
        "/opt/venv/bin/python -m pip install --ignore-installed uv",

        # Sanity check that the uv binary now exists in the venv.
        "ls -l /opt/venv/bin/uv && /opt/venv/bin/uv --version",

        # Get ACE-Step source.
        "git clone https://github.com/ACE-Step/ACE-Step-1.5.git /opt/ACE-Step-1.5",

        # Install dependencies into /opt/venv, not the repo's default .venv.
        # Invoked as a module (python -m uv) so it works regardless of where
        # the uv executable lives. --inexact stops uv from pruning packages
        # already present (e.g. the base image's PyTorch/CUDA stack).
        "cd /opt/ACE-Step-1.5 && "
        "UV_PROJECT_ENVIRONMENT=/opt/venv "
        "/opt/venv/bin/python -m uv sync --inexact --python /opt/venv/bin/python",

        # Verify the environment and GPU libraries.
        "/opt/venv/bin/python -c \"import torch; print('Python environment OK'); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())\"",
    )
    .env({
        "VIRTUAL_ENV": VENV_DIR,
        "UV_PROJECT_ENVIRONMENT": VENV_DIR,
        "PATH": (
            "/opt/venv/bin:"
            "/usr/local/cuda/bin:"
            "/usr/local/sbin:"
            "/usr/local/bin:"
            "/usr/sbin:"
            "/usr/bin:"
            "/sbin:"
            "/bin"
        ),
        "PYTHONUNBUFFERED": "1",
    })
)


# ============================================================
# DOWNLOAD MODEL WEIGHTS
# ============================================================

@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={
        CHECKPOINT_DIR: volume,
    },
)
def download_models():

    from huggingface_hub import snapshot_download

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    models = [
        (
            "ACE-Step/acestep-v15-xl-sft",
            f"{CHECKPOINT_DIR}/{DIT_MODEL}",
        ),
        (
            "ACE-Step/acestep-5Hz-lm-4B",
            f"{CHECKPOINT_DIR}/{LM_MODEL}",
        ),
    ]

    for repo_id, local_dir in models:

        model_dir = Path(local_dir)

        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"Model already exists: {repo_id}")
            continue

        print(f"\nDownloading: {repo_id}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )

        print(f"Finished downloading: {repo_id}")

    volume.commit()

    print("\nAll model downloads completed.")


# ============================================================
# GENERATE MUSIC
# ============================================================

@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={
        CHECKPOINT_DIR: volume,
    },
)
def generate_song():

    import torch

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import (
        GenerationParams,
        GenerationConfig,
        generate_music,
    )

    print("\n" + "=" * 65)
    print("ACE-STEP 1.5 XL MUSIC GENERATION")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    dit_path = Path(CHECKPOINT_DIR) / DIT_MODEL
    lm_path = Path(CHECKPOINT_DIR) / LM_MODEL

    if not dit_path.exists() or not any(dit_path.iterdir()):
        raise FileNotFoundError(f"DiT model missing: {dit_path}")

    if not lm_path.exists() or not any(lm_path.iterdir()):
        raise FileNotFoundError(f"Language model missing: {lm_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # INITIALIZE DiT
    # --------------------------------------------------------

    print("\nInitializing XL SFT model...")

    dit_handler = AceStepHandler()

    dit_handler.initialize_service(
        project_root=REPO_DIR,
        config_path=DIT_MODEL,
        device="cuda",
    )

    print("XL SFT initialization complete.")

    # --------------------------------------------------------
    # INITIALIZE 4B LANGUAGE MODEL
    # --------------------------------------------------------

    print("\nInitializing 4B language model...")

    llm_handler = LLMHandler()

    # If vllm fails with a torch/CUDA version conflict, change
    # backend="vllm" to backend="pt" as a fallback.
    llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend="vllm",
        device="cuda",
    )

    print("4B language model initialization complete.")

    # --------------------------------------------------------
    # GENERATION SETTINGS
    # --------------------------------------------------------

    params = GenerationParams(
        task_type="text2music",
        caption=CAPTION,
        lyrics=LYRICS,
        instrumental=False,
        vocal_language="en",

        bpm=104,
        keyscale="F# minor",
        timesignature="4",
        duration=150.0,

        inference_steps=50,
        seed=SEED,
        guidance_scale=7.0,
        use_adg=True,

        cfg_interval_start=0,
        cfg_interval_end=1,

        shift=1.0,
        infer_method="ode",

        thinking=True,
        lm_temperature=0.85,
        lm_cfg_scale=2.5,
        lm_top_k=0,
        lm_top_p=0.9,

        use_cot_metas=True,
        use_cot_caption=True,
        use_cot_language=True,
        use_constrained_decoding=True,
    )

    config = GenerationConfig(
        batch_size=1,
        allow_lm_batch=False,
        use_random_seed=False,
        seeds=[SEED],
        audio_format="flac",
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print("\nStarting generation...")
    print("Title: NO SLOW FADE")
    print("Duration: 150 seconds")
    print("DiT: XL SFT")
    print("Language model: 4B")
    print("Inference steps: 50")
    print("Seed:", SEED)

    start_time = time.time()

    result = generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=OUTPUT_DIR,
    )

    generation_seconds = time.time() - start_time

    print(
        f"\nGeneration completed in "
        f"{generation_seconds:.1f} seconds."
    )

    if not result.audios:
        raise RuntimeError("ACE-Step returned no audio.")

    audio = result.audios[0]

    if isinstance(audio, dict):
        audio_path = audio.get("path")
        sample_rate = audio.get("sample_rate")
    else:
        audio_path = getattr(audio, "path", None)
        sample_rate = getattr(audio, "sample_rate", None)

    if not audio_path:
        raise RuntimeError(
            f"Could not determine generated audio path: {audio}"
        )

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Generated audio file not found: {audio_path}"
        )

    # --------------------------------------------------------
    # CREATE 320 KBPS MP3
    # --------------------------------------------------------

    mp3_path = Path(OUTPUT_DIR) / "no_slow_fade.mp3"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(mp3_path),
        ],
        check=True,
    )

    print("\nAudio generation and conversion complete.")

    return {
        "flac": audio_path.read_bytes(),
        "mp3": mp3_path.read_bytes(),
        "generation_seconds": generation_seconds,
        "sample_rate": sample_rate,
    }


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():

    output_dir = Path("./generated")
    output_dir.mkdir(exist_ok=True)

    print("\nACE-STEP 1.5 XL MUSIC TEST")
    print("=" * 55)

    print("\nSTEP 1: MODEL DOWNLOAD")
    download_models.remote()

    print("\nSTEP 2: MUSIC GENERATION")
    result = generate_song.remote()

    flac_file = output_dir / "no_slow_fade.flac"
    mp3_file = output_dir / "no_slow_fade.mp3"

    flac_file.write_bytes(result["flac"])
    mp3_file.write_bytes(result["mp3"])

    print("\n" + "=" * 55)
    print("GENERATION COMPLETE")
    print("=" * 55)

    print(f"\nFLAC: {flac_file.resolve()}")
    print(f"MP3:  {mp3_file.resolve()}")

    print(
        f"\nGeneration time: "
        f"{result['generation_seconds']:.1f} seconds"
    )

    if result.get("sample_rate"):
        print(f"Sample rate: {result['sample_rate']} Hz")

    print("\nYour song is ready to listen to.")