from __future__ import annotations

from pathlib import Path
import re
import json
from datetime import datetime


def main() -> int:
    out_dir = Path("doc_extract")
    out_dir.mkdir(exist_ok=True)

    docx_path = Path("playing with RH.docx")
    meta = {
        "path": str(docx_path),
        "exists": docx_path.exists(),
        "size": docx_path.stat().st_size if docx_path.exists() else None,
        "mtime": datetime.fromtimestamp(docx_path.stat().st_mtime).isoformat() if docx_path.exists() else None,
    }

    from docx import Document

    doc = Document(str(docx_path))
    paras: list[dict] = []
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if not txt:
            continue
        style = getattr(getattr(p, "style", None), "name", "") or ""
        paras.append({"text": txt, "style": style})

    meta["nonempty_paragraphs"] = len(paras)

    last_n = 220
    start_idx = max(0, len(paras) - last_n)
    end_slice = paras[start_idx:]

    end_path = out_dir / "playing_with_RH__end.md"
    with end_path.open("w", encoding="utf-8") as f:
        f.write("# Extract: playing with RH (end)\n\n")
        f.write(json.dumps(meta, indent=2) + "\n\n")
        f.write(f"## Last {len(end_slice)} nonempty paragraphs (index, style, text)\n\n")
        for i, item in enumerate(end_slice, start=start_idx):
            t = re.sub(r"\s+", " ", item["text"])
            f.write(f"[{i}] ({item['style']}) {t}\n")

    keys = [
        "validation track",
        "validate",
        "validation",
        "directions",
        "instructions",
        "checklist",
        "cell 1",
        "cell 2",
        "cell 3",
        "cell 4",
        "cell 5",
        "vsgpt",
        "other gpt",
        "communicate",
        "non-negotiable",
        "must",
        "do not",
        "required",
        "sanity",
        "unitary",
        "unitarity",
        "functional equation",
        "argument principle",
        "divisor",
        "normalization",
        "normalize",
        "cayley",
        "dn map",
        "schur",
        "completion",
        "holomorphic",
        "1-s",
        "conjugate",
    ]

    hits: list[int] = []
    for idx, item in enumerate(paras):
        low = item["text"].lower()
        if any(k in low for k in keys):
            hits.append(idx)

    hits_path = out_dir / "playing_with_RH__keyword_hits.md"
    with hits_path.open("w", encoding="utf-8") as f:
        f.write("# Keyword hits: playing with RH\n\n")
        f.write(json.dumps(meta, indent=2) + "\n\n")
        f.write(f"Total hits: {len(hits)}\n\n")
        window = 2
        for idx in hits[-120:]:
            f.write(f"## Hit at [{idx}]\n")
            for j in range(max(0, idx - window), min(len(paras), idx + window + 1)):
                prefix = ">>" if j == idx else "  "
                t = re.sub(r"\s+", " ", paras[j]["text"])
                style = paras[j]["style"]
                f.write(f"{prefix} [{j}] ({style}) {t}\n")
            f.write("\n")

    print("Wrote:", end_path.as_posix())
    print("Wrote:", hits_path.as_posix())
    print("Nonempty paragraphs:", len(paras))
    print("Tail start index:", start_idx)
    print("Keyword hits:", len(hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
