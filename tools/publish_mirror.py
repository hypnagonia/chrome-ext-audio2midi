"""Publish fp16 MuScriptor mirrors to a public Hugging Face repo.

Usage (needs a write token, e.g. from `hf auth login`):
    python tools/publish_mirror.py <weights_dir> <hf_user> small [medium]

<weights_dir>/<size>/ must hold the original model.safetensors and config.json
from MuScriptor/muscriptor-<size>. Each size is converted to fp16 (see
to_fp16.py) and uploaded to <hf_user>/muscriptor-<size>-fp16 with the CC BY-NC
licence, attribution and the authors' conditions of use.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

PARAMS = {"small": "103M", "medium": "307M", "large": "1.4B"}

CONDITIONS = (
    "MuScriptor is the result of a research collaboration between Mirelo and Kyutai whose purpose is to "
    "transcribe audio to MIDI/music sheet. It is provided primarily for research purposes under the CC BY-NC 4.0 "
    "licence supplemented by the below specific conditions of use.\n>\n> Specific conditions of use: MuScriptor "
    "and any generated content by MuScriptor are provided as is without any warranty of any kind, including but not "
    "limited to any warranty of non-infringement. Use of MuScriptor and its output must comply with all applicable "
    "laws and must not result in, involve, or facilitate any illegal or unauthorized activity. Prohibited uses "
    "include, without limitation, inputting music files and transcribing them to MIDI/music sheet without having all "
    "the necessary rights, including intellectual property rights, under applicable laws. Accordingly, users of "
    "MuScriptor undertake and warrant to have all the necessary rights, including intellectual property rights, in "
    "connection with their use of MuScriptor and its output. We disclaim all liability for any non-compliant use and "
    "users of MuScriptor shall indemnify, defend, and hold harmless Mirelo and Kyutai from and against any and all "
    "claims, damages, losses, liabilities, and expenses (including reasonable attorneys' fees) incurred by Mirelo "
    "and/or Kyutai arising out of or resulting from their failure to comply with the terms of the CC BY-NC 4.0 "
    "licence and/or these specific conditions of use."
)


def readme(size: str) -> str:
    return f"""---
license: cc-by-nc-4.0
library_name: muscriptor
base_model: MuScriptor/muscriptor-{size}
tags:
- music
- music-transcription
- audio-to-midi
- midi
- webgpu
---

# MuScriptor {size} (fp16 mirror)

An fp16 copy of [MuScriptor/muscriptor-{size}](https://huggingface.co/MuScriptor/muscriptor-{size}) ({PARAMS[size]} parameters), the multi-instrument music transcription model by [Mirelo](https://www.mirelo.ai/) and [Kyutai](https://kyutai.org/). It's mirrored here so the [byEar](https://github.com/hypnagonia/chrome-ext-audio2midi) Chrome extension by [jenyadoesapps](https://jenyadoesapps.com/) can download it without a Hugging Face login. It runs fully on your GPU with WebGPU.

**Changes from the original:** transformer weights converted from fp32 to fp16, which halves the download. The audio-conditioning tensors (`condition_provider.*`) are kept in fp32. Tensor names are unchanged, so the official [`muscriptor`](https://github.com/muscriptor/muscriptor) package loads this file as is.

All credit goes to the original authors. Paper: *MuScriptor: An Open Model for Multi-Instrument Music Transcription*, Rouard, Krause, Roebel, Simon-Gabriel, Défossez (2026), [arXiv:2607.08168](https://arxiv.org/abs/2607.08168).

## License and conditions of use

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): **non-commercial use only**, with attribution. The original authors add the following conditions, which also apply to this copy:

> {CONDITIONS}

## Citation

```bibtex
@misc{{rouard2026muscriptoropenmodelmultiinstrument,
      title={{MuScriptor: An Open Model for Multi-Instrument Music Transcription}},
      author={{Simon Rouard and Michael Krause and Axel Roebel and Carl-Johann Simon-Gabriel and Alexandre Défossez}},
      year={{2026}},
      eprint={{2607.08168}},
      archivePrefix={{arXiv}},
}}
```
"""


def main() -> None:
    weights_dir, user, *sizes = sys.argv[1:]
    api = HfApi()
    for size in sizes:
        src = Path(weights_dir) / size
        out = Path(weights_dir) / f"{size}-fp16"
        out.mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("to_fp16.py")), str(src / "model.safetensors"), str(out / "model.safetensors")],
            check=True,
        )
        shutil.copy(src / "config.json", out / "config.json")
        (out / "README.md").write_text(readme(size))
        repo = f"{user}/muscriptor-{size}-fp16"
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=str(out), repo_id=repo, commit_message=f"MuScriptor {size} fp16 mirror")
        print(f"published https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
