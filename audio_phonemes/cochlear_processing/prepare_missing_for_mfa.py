"""
Prepare missing words for MFA re-alignment.
Copies existing WAVs (handling known alternate/misspellings) into a staging folder
and creates simple transcripts. Then you can run MFA on just these words to 
generate TextGrids.
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
AUDIO_DIR = PROJECT_ROOT / "audio_phonemes" / "All_Recordings"
STAGING_DIR = PROJECT_ROOT / "audio_phonemes" / "MFA_missing"
WAV_OUT = STAGING_DIR / "wavs"
TXT_OUT = STAGING_DIR / "txt"

MISSING_WORDS = [
    "country", "defence", "fable", "flavour", "footwear", "haddock", "harbour",
    "hour", "message", "moustache", "neighbour", "pedal", "portion", "pupil",
    "racquet", "report", "result", "rumour", "satin", "segment", "sparrow",
    "traitor", "weapon",
]

# Known alternates/misspellings in All_Recordings -> map to canonical
ALT_NAMES = {
    "defence": "defense.wav",
    "fable": "fabel.wav",
    "flavour": "flavor.wav",
    "footwear": "footware.wav",
    "harbour": "harbor.wav",
    "moustache": "mustache.wav",
    "country": "country  .wav",  # extra spaces seen
    # If you have haddock audio under a different name, add here, e.g.:
    # "haddock": "havoc.wav",
}


def main():
    WAV_OUT.mkdir(parents=True, exist_ok=True)
    TXT_OUT.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for w in MISSING_WORDS:
        candidates = []
        # canonical
        candidates.append(AUDIO_DIR / f"{w}.wav")
        # alternate
        if w in ALT_NAMES:
            candidates.append(AUDIO_DIR / ALT_NAMES[w])
        wav_src = next((p for p in candidates if p.exists()), None)
        if wav_src is None:
            print(f"[WARN] missing wav for {w}: tried {[str(c) for c in candidates]}")
            fail += 1
            continue
        wav_dst = WAV_OUT / f"{w}.wav"  # normalized/canonical name
        shutil.copyfile(wav_src, wav_dst)
        # transcript: canonical word
        (TXT_OUT / f"{w}.txt").write_text(w + "\n", encoding="utf-8")
        ok += 1
    print(f"Copied {ok} wavs; missing {fail}")
    print(f"Staging wavs in: {WAV_OUT}")
    print(f"Transcripts in:  {TXT_OUT}")
    print("Run MFA on staging folder to produce TextGrids, e.g.:")
    print("  mfa align \"" + str(WAV_OUT) + "\" \"" + str(TXT_OUT) + "\" english_us_arpa \"" + str(PROJECT_ROOT / 'audio_phonemes' / 'MFA_Output_TextGrids') + "\"")


if __name__ == "__main__":
    main()
