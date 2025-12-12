from pathlib import Path
import pandas as pd

missing = [
    "country","defence","fable","flavour","footwear","haddock","harbour",
    "hour","message","moustache","neighbour","pedal","portion","pupil",
    "racquet","report","result","rumour","satin","segment","sparrow",
    "traitor","weapon",
]
alt = {
    "defence": "defense.wav",
    "fable": "fabel.wav",
    "flavour": "flavor.wav",
    "footwear": "footware.wav",
    "harbour": "harbor.wav",
    "moustache": "mustache.wav",
    "country": "country  .wav",
    "neighbour": "neighbor.wav",
    "pedal": "petal.wav",
}
root = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
audio_dir = root / "audio_phonemes" / "All_Recordings"
lex_path = root / "my_800_words.csv"
out_path = root / "audio_phonemes" / "cochlear_processing" / "check_wavs_output.txt"

try:
    df = pd.read_csv(lex_path)
    if 'word' in df.columns:
        lex = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        lex = df[df.columns[0]].astype(str).str.strip().str.lower().tolist()
    lex = set(lex)
except Exception:
    lex = set()

wav_files = list(audio_dir.glob('*.wav'))
wav_names = set(p.stem.lower().strip() for p in wav_files)

missing_wav = sorted(list(lex - wav_names))
extra_wav = sorted(list(wav_names - lex))

lines = []
lines.append(f"Lex words: {len(lex)}")
lines.append(f"WAV files: {len(wav_names)}")
lines.append(f"Missing WAVs: {len(missing_wav)}")
lines.append(str(missing_wav[:200]))
lines.append(f"Extra WAVs: {len(extra_wav)}")
lines.append(str(extra_wav[:200]))
lines.append("\nAlt-candidate hits:")
for w in missing:
    candidates = [audio_dir / f"{w}.wav"]
    if w in alt:
        candidates.append(audio_dir / alt[w])
    hit = next((c for c in candidates if c.exists()), None)
    lines.append(f"  {w}: {'FOUND ' + hit.name if hit else 'MISSING'}")

out_path.write_text('\n'.join(lines), encoding='utf-8')
print("Wrote", out_path)
