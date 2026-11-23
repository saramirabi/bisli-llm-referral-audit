"""Clean BiSLI CHAT transcripts so group/diagnosis cannot leak into prompts.

Reads a TalkBank BiSLI zip (or a directory of .cha files). Does not modify the
archive. Writes plain Adult:/Child: transcripts and an id map CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

ADULTS = {"EX1", "EXP", "DAD"}
BULLET = re.compile("\x15[^\x15]*\x15")
ANGLE = re.compile(r"<([^>]*)>")
BRACKET = re.compile(r"\[[^\]]*\]")
PAREN = re.compile(r"\([^)]*\)")
OMITTED = re.compile(r"(?<!\S)0\S*")
GESTURE = re.compile(r"&=\S+")
FRAGMENT = re.compile(r"&[-+]?")
PLUS = re.compile(r"(^|\s)\+\S*")
AT_CODE = re.compile(r"@\w+")
UNINTELLIGIBLE = re.compile(r"(?<!\S)(?:xxx|yyy|www|ххх)(?!\S)")
SPACE = re.compile(r"\s+")

FOLDER_META = {
    "Biling-Dutch": ("bilingual", "Dutch", "TD"),
    "Biling-Russian": ("bilingual", "Russian", "TD"),
    "Mono-Dutch": ("monolingual", "Dutch", "TD"),
    "Mono-Russian": ("monolingual", "Russian", "TD"),
    "Mono-RussianSLI": ("monolingual", "Russian", "SLI"),
}


def clean_utterance(text: str) -> str:
    text = BULLET.sub(" ", text)
    text = ANGLE.sub(r"\1", text)
    text = BRACKET.sub(" ", text)
    text = PAREN.sub(" ", text)
    text = OMITTED.sub(" ", text)
    text = GESTURE.sub(" ", text)
    text = FRAGMENT.sub("", text)
    text = PLUS.sub(" ", text)
    text = AT_CODE.sub("", text)
    text = UNINTELLIGIBLE.sub("[unintelligible]", text)
    text = SPACE.sub(" ", text).strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    return text


def turns(chat: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    speaker = ""
    for raw in chat.splitlines():
        line = raw.strip()
        if not line or line.startswith("@"):
            continue
        if line.startswith("%"):
            continue
        if ":" in line and line.split(":", 1)[0].isupper():
            code, rest = line.split(":", 1)
            speaker = "Adult" if code in ADULTS else "Child"
            text = clean_utterance(rest)
            if text:
                result.append((speaker, text))
        elif speaker and result:
            text = clean_utterance(line)
            if text:
                prev_sp, prev_tx = result[-1]
                if prev_sp == speaker:
                    result[-1] = (speaker, f"{prev_tx} {text}")
                else:
                    result.append((speaker, text))
    return result


def age_months_from_cha(chat: str) -> str:
    for line in chat.splitlines():
        if line.startswith("@ID:") and "Target_Child" in line:
            parts = line.split("|")
            if len(parts) > 3 and parts[3].strip():
                age = parts[3].strip()
                # CHAT ages like 5;03
                if ";" in age:
                    y, m = age.split(";", 1)
                    try:
                        return str(int(y) * 12 + int(m.split(".")[0]))
                    except ValueError:
                        return ""
                return age
    return ""


def child_code_from_path(path: str) -> str:
    name = Path(path).stem
    return name


def iter_cha_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".cha"):
                continue
            parts = Path(info.filename).parts
            folder = next((p for p in parts if p in FOLDER_META), None)
            if folder is None:
                continue
            yield info.filename, folder, zf.read(info).decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, required=True, help="BiSLI.zip from TalkBank")
    ap.add_argument("--transcripts-out", type=Path, required=True)
    ap.add_argument("--id-map-out", type=Path, required=True)
    args = ap.parse_args()

    args.transcripts_out.mkdir(parents=True, exist_ok=True)
    args.id_map_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (path, folder, chat) in enumerate(iter_cha_from_zip(args.zip), start=1):
        group, language, diagnosis = FOLDER_META[folder]
        item_id = f"item_{i:04d}"
        child_id = child_code_from_path(path)
        age = age_months_from_cha(chat)
        lines = [f"{sp}: {tx}" for sp, tx in turns(chat)]
        (args.transcripts_out / f"{item_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        rows.append(
            {
                "item_id": item_id,
                "child_id": child_id,
                "source_path": path,
                "folder": folder,
                "group": group,
                "language": language,
                "diagnosis": diagnosis,
                "age_months": age,
            }
        )

    with args.id_map_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "item_id",
                "child_id",
                "source_path",
                "folder",
                "group",
                "language",
                "diagnosis",
                "age_months",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} transcripts -> {args.transcripts_out}")
    print(f"wrote id map -> {args.id_map_out}")


if __name__ == "__main__":
    main()
