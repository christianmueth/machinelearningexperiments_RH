import os
import re
import pandas as pd


BEGIN = "<!-- BEGIN KEY_NUMBERS -->"
END = "<!-- END KEY_NUMBERS -->"


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _render_key_numbers_markdown(df: pd.DataFrame) -> str:
    # Expect two stages: preregistered_seedblock_32_63 and confirm_preregistered_rejecting_seeds_N16384
    out_lines: list[str] = []

    # Preregistered block
    prereg = df[df["stage"] == "preregistered_seedblock_32_63"].copy()
    if len(prereg):
        prereg = prereg.sort_values(["wlo", "whi", "backend_label"], kind="stable")
        out_lines.append("**Preregistered generalization (seeds 32–63, N_null=4096)**")
        out_lines.append("")
        out_lines.append("| window | backend | n_seeds | seeds w/ any reject | frac | best_p_overall | median_best_p |")
        out_lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for _, r in prereg.iterrows():
            w = f"({r['wlo']:.1f}, {r['whi']:.1f})"
            backend = str(r["backend_label"])
            n_units = int(r["n_units"]) if pd.notna(r.get("n_units")) else ""
            n_any = int(r["n_units_any_reject"]) if pd.notna(r.get("n_units_any_reject")) else ""
            frac = float(r["frac_units_any_reject"]) if pd.notna(r.get("frac_units_any_reject")) else float("nan")
            best_p_overall = float(r["best_p_overall"]) if pd.notna(r.get("best_p_overall")) else float("nan")
            median_best_p = float(r["median_best_p"]) if pd.notna(r.get("median_best_p")) else float("nan")
            out_lines.append(
                f"| {w} | {backend} | {n_units} | {n_any} | {frac:.5f} | {best_p_overall:.6g} | {median_best_p:.6g} |"
            )
        out_lines.append("")

    # Confirm block
    confirm = df[df["stage"] == "confirm_preregistered_rejecting_seeds_N16384"].copy()
    if len(confirm):
        confirm = confirm.sort_values(["wlo", "whi", "backend_label"], kind="stable")
        seed_set = ""
        if "seed_set" in confirm.columns:
            # Should be identical across backends
            seed_set = str(confirm["seed_set"].dropna().iloc[0]) if confirm["seed_set"].dropna().size else ""
        wlo = float(confirm["wlo"].dropna().iloc[0]) if confirm["wlo"].dropna().size else float("nan")
        whi = float(confirm["whi"].dropna().iloc[0]) if confirm["whi"].dropna().size else float("nan")
        out_lines.append(f"**High-null confirm (N_null=16384) on preregistered rejecting seeds**")
        if seed_set:
            out_lines.append(f"- Seeds: `{seed_set}`")
        out_lines.append(f"- Window: ({wlo:.1f}, {whi:.1f})")
        out_lines.append("")
        out_lines.append("| backend | rows | reject_rate | best_p | median_p | worst_p |")
        out_lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in confirm.iterrows():
            backend = str(r["backend_label"])
            rows = int(r["rows"]) if pd.notna(r.get("rows")) else ""
            reject_rate = float(r["reject_rate"]) if pd.notna(r.get("reject_rate")) else float("nan")
            best_p = float(r["best_p"]) if pd.notna(r.get("best_p")) else float("nan")
            median_p = float(r["median_p"]) if pd.notna(r.get("median_p")) else float("nan")
            worst_p = float(r["worst_p"]) if pd.notna(r.get("worst_p")) else float("nan")
            out_lines.append(
                f"| {backend} | {rows} | {reject_rate:.3f} | {best_p:.6g} | {median_p:.6g} | {worst_p:.6g} |"
            )
        out_lines.append("")

    if not out_lines:
        out_lines = ["(No rows found in out_stage_gate_figure_table.csv)"]

    return "\n".join(out_lines).rstrip() + "\n"


def _upsert_block(text: str, block_md: str) -> str:
    if BEGIN in text and END in text:
        # Replace existing
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        return pattern.sub(BEGIN + "\n" + block_md + END, text)
    # Insert after the first H1 header paragraph.
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            insert_at = i + 1
            break
    # Skip immediate blank lines
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        insert_at += 1
    section = [
        "## Key numbers (auto-generated)",
        "",
        "Generated from [out_stage_gate_figure_table.csv](out_stage_gate_figure_table.csv) via `update_key_numbers_blocks.py`.",
        "",
        BEGIN,
        block_md.rstrip(),
        END,
        "",
    ]
    return "\n".join(lines[:insert_at] + [""] + section + lines[insert_at:]).rstrip() + "\n"


def update_files(
    csv_path: str = "out_stage_gate_figure_table.csv",
    files: tuple[str, ...] = ("POSITIONING_STATEMENT.md", "THEOREM_ROADMAP.md"),
) -> None:
    df = _read_csv(csv_path)
    block_md = _render_key_numbers_markdown(df)

    for path in files:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        new_text = _upsert_block(text, block_md)
        if new_text != text:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_text)
            print(f"updated {path}")
        else:
            print(f"no change {path}")


if __name__ == "__main__":
    update_files()
