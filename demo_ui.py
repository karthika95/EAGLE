#!/usr/bin/env python3
"""
demo_ui.py — Gradio UI for Samanvaya-Constrained Generation (SCG).

GPU assignment (both preloaded at startup):
  Airavata 7B  → cuda:0
  Param-1 2.9B → cuda:4

Three modes shown simultaneously: Baseline | SCG | AdaSCG
Token-level highlights (hover for details):
  ✓ OPP_SAVE    — greeen — model's greedy was already LWG-valid (DOMINO saved)
  ⚡ MASK_APPLIED — orange — vocabulary mask forced a valid token
  ⊘ ENTROPY_SKIP — yellow — AdaSCG entropy gate suppressed mask
  ▲ DIFF        — red    — this token differs from baseline (in SCG/AdaSCG panels)
"""

from __future__ import annotations
import difflib
import json
import time
from pathlib import Path
from typing import Optional

import torch
import gradio as gr

WORKSPACE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# GPU assignment
# ─────────────────────────────────────────────────────────────────────────────

MODEL_GPU = {
    "airavata": "cuda:0",
    "param":    "cuda:4",
}

MODEL_LABELS = {
    "Airavata 7B (ai4bharat)": "airavata",
    "Param-1 2.9B (bharatgenai)": "param",
}

# ─────────────────────────────────────────────────────────────────────────────
# Startup model preloading
# ─────────────────────────────────────────────────────────────────────────────

def preload_all() -> None:
    """Load both models sequentially at startup so switching is instant."""
    from scg_generate import load_model
    for key, dev in MODEL_GPU.items():
        print(f"[SCG-UI] Loading {key} on {dev} …", flush=True)
        load_model(key, device=dev)
        print(f"[SCG-UI] {key} ready.", flush=True)
    print("[SCG-UI] Both models loaded. Starting UI …", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

def run_all_modes(
    prompt: str,
    model_key: str,
    max_new_tokens: int,
    T_G_init: float = 0.5,
) -> tuple[dict, object]:
    """
    Run baseline, constrained, adaptive generation.
    Returns (results_dict, tokenizer).

    Trace token alignment (verified from code):
      trace_log[0]  = INIT step  (controls output_token_0 but always passes)
      trace_log[j]  → action that controls output_token_j

    So: output token at index j → trace_log[j].action.
    """
    from scg_generate import load_model, MODELS
    from samanvaya_logits_processor import SamanvayaLogitsProcessor
    from samanvaya_parser import SamanvayaParser

    device = MODEL_GPU[model_key]
    cfg = MODELS[model_key]
    model, tokenizer = load_model(model_key, device=device)

    full_prompt = cfg["prompt_template"].format(instruction=prompt)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    results = {}
    for mode in ("baseline", "constrained", "adaptive"):
        processor: Optional[SamanvayaLogitsProcessor] = None
        procs = []

        if mode in ("constrained", "adaptive"):
            processor = SamanvayaLogitsProcessor(
                tokenizer=tokenizer,
                vocab_scan=WORKSPACE / cfg["vocab_scan"],
                use_entropy_gate=(mode == "adaptive"),
                T_G_init=T_G_init,
                eos_token_id=tokenizer.eos_token_id,
                trace=True,
            )
            processor.reset_stats()
            procs.append(processor)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                logits_processor=procs if procs else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_ids = out_ids[0, input_ids.shape[1]:].tolist()
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        n_tok = len(new_ids)

        if processor is not None:
            processor._flush_buffer()
            gen_stats = processor.generation_stats()
            trace_log = processor.get_trace_log()
            compliance = processor.parser.compliance_rate()
            parser_stats = processor.parser.stats()
        else:
            gen_stats = {}
            trace_log = []
            pp = SamanvayaParser()
            for w in response.split():
                pp.update(w)
            compliance = pp.compliance_rate()
            parser_stats = pp.stats()

        results[mode] = {
            "response": response,
            "new_ids": new_ids,
            "n_tokens": n_tok,
            "elapsed": round(elapsed, 2),
            "tps": round(n_tok / elapsed, 1) if elapsed > 0 else 0.0,
            "compliance": compliance or 0.0,
            "parser_stats": parser_stats,
            "gen_stats": gen_stats,
            "trace_log": trace_log,
        }

    return results, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

# (bg_color, text_color, icon, default tooltip)
ACTION_STYLE: dict[str, tuple[str, str, str, str]] = {
    "OPP_SAVE":          ("#c8f7c5", "#1a6b1a", "✓", "Model chose correctly without intervention"),
    "MASK_APPLIED":      ("#ffd9a0", "#7a3b00", "⚡", "Vocabulary mask forced a valid token"),
    "ENTROPY_SKIP":      ("#fffacc", "#6b5e00", "⊘", "Entropy gate skipped constraint"),
    "PASS_NO_VALID_IDS": ("#dce9f7", "#1a4d6b", "?", "Constrained state but no valid token IDs"),
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _display(tok_str: str) -> str:
    """Strip leading ▁ (SentencePiece word-start marker → becomes a space in context)."""
    return _esc(tok_str.replace("▁", " "))


def render_action_html(new_ids: list[int], trace_log: list[dict], tokenizer) -> str:
    """
    Wrap each output token in a colored span based on the action that controlled
    its generation.

    Alignment: output_token[j] ↔ trace_log[j]
      • trace_log[0] = INIT (always PASS, no color)
      • trace_log[j].action controls what output_token[j] will be, for j ≥ 0

    NOTE: use convert_ids_to_tokens (not decode) for single tokens so that
    the SentencePiece ▁ word-boundary prefix is preserved.  decode([id])
    strips ▁ without inserting the corresponding space, causing words to jam.
    """
    if not new_ids:
        return "<em style='color:#999'>No tokens generated.</em>"

    raw_tokens = tokenizer.convert_ids_to_tokens(new_ids)

    html: list[str] = []
    for j, (tid, tok_str) in enumerate(zip(new_ids, raw_tokens)):
        display = _display(tok_str or "")

        entry = trace_log[j] if j < len(trace_log) else None
        action = entry.get("action", "PASS") if entry else "PASS"

        if action in ACTION_STYLE:
            bg, fg, icon, tip = ACTION_STYLE[action]
            if action == "MASK_APPLIED" and entry:
                g = _esc(entry.get("greedy_str", "?").replace("▁", ""))
                f = _esc(entry.get("forced_str", "?").replace("▁", ""))
                gp = entry.get("greedy_prob", 0)
                fp = entry.get("forced_prob", 0)
                tip = f"Wanted &#39;{g}&#39; ({gp:.2f}) → Forced &#39;{f}&#39; ({fp:.2f})"
            elif action == "OPP_SAVE" and entry:
                g = _esc(entry.get("greedy_str", "?").replace("▁", ""))
                gp = entry.get("greedy_prob", 0)
                tip = f"Greedy &#39;{g}&#39; ({gp:.2f}) was already LWG-valid"
            elif action == "ENTROPY_SKIP" and entry:
                H = entry.get("entropy")
                TG = entry.get("T_G")
                if H is not None and TG is not None:
                    tip = f"H={float(H):.3f} ≤ T_G={float(TG):.3f} → mask skipped"
            style = (
                f"background:{bg};color:{fg};"
                "border-radius:3px;padding:0 2px;font-weight:500;"
            )
            html.append(f'<span style="{style}" title="{tip}">{display}</span>')
        else:
            html.append(display)

    return "".join(html)


def render_diff_html(new_ids: list[int], base_ids: list[int], tokenizer) -> str:
    """
    Show new_ids with tokens that differ from base_ids highlighted in red.
    Uses SequenceMatcher so shifts after a divergence are handled properly.

    Uses convert_ids_to_tokens (not decode) for single-token display so the
    SentencePiece ▁ prefix is preserved and converted to a space by _display().
    """
    if not new_ids:
        return ""

    new_raw = tokenizer.convert_ids_to_tokens(new_ids)
    base_raw = tokenizer.convert_ids_to_tokens(base_ids) if base_ids else []

    sm = difflib.SequenceMatcher(None, base_ids, new_ids, autojunk=False)
    html: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for tok_str in new_raw[j1:j2]:
                html.append(_display(tok_str or ""))

        elif tag == "replace":
            base_text = _esc("".join(
                (t or "").replace("▁", " ") for t in base_raw[i1:i2]
            ).strip())
            for tok_str in new_raw[j1:j2]:
                d = _display(tok_str or "")
                style = (
                    "background:#ffe0e0;color:#8b0000;border-radius:3px;"
                    "padding:0 2px;border:1px solid #c00;"
                )
                html.append(f'<span style="{style}" title="Baseline: &#39;{base_text}&#39;">{d}</span>')

        elif tag == "insert":
            for tok_str in new_raw[j1:j2]:
                d = _display(tok_str or "")
                style = (
                    "background:#fff0cc;color:#7a5000;border-radius:3px;"
                    "padding:0 2px;border:1px solid #f90;border-style:dashed;"
                )
                html.append(f'<span style="{style}" title="Inserted (not in baseline)">{d}</span>')
        # "delete" → tokens only in baseline → skip (we display new_ids only)

    return "".join(html)


def stats_html(mode: str, r: dict, base_compliance: float = 0.0) -> str:
    """Styled stats panel for one generation mode."""
    c = (r.get("compliance") or 0.0) * 100
    gs = r.get("gen_stats") or {}
    ps = r.get("parser_stats") or {}

    ir  = gs.get("intervention_rate", 0) * 100
    osr = gs.get("opp_save_rate", 0) * 100
    ma  = gs.get("mask_applied", 0)
    opp = gs.get("opportunistic_saves", 0)
    ent = gs.get("entropy_gate_skips", 0)
    cs  = gs.get("constrained_steps", 0)

    obl  = ps.get("obligations", 0)
    comp = ps.get("completions", 0)
    viol = ps.get("violations", 0)

    bar_color = "#27ae60" if c >= 75 else "#e67e22" if c >= 50 else "#e74c3c"
    bar_w = int(c)

    delta_html = ""
    if mode != "baseline" and base_compliance > 0:
        delta = c - base_compliance * 100
        sign = "+" if delta >= 0 else ""
        delta_color = "#27ae60" if delta >= 0 else "#e74c3c"
        delta_html = f'<span style="font-size:11px;color:{delta_color};margin-left:6px">{sign}{delta:.1f} pp vs base</span>'

    TD_L = 'style="color:#333;padding:3px 4px;font-size:12px;white-space:nowrap;"'
    TD_R = 'style="color:#111;padding:3px 4px;font-size:12px;font-weight:600;"'

    rows = f"""
      <tr><td {TD_L}>Tokens</td>
          <td {TD_R}>{r['n_tokens']} @ {r['tps']} tok/s</td></tr>
      <tr><td {TD_L}>Time</td>
          <td {TD_R}>{r['elapsed']}s</td></tr>
      <tr><td {TD_L}>LWG obligations</td>
          <td {TD_R}>{obl} ({comp} ✓ · {viol} ✗)</td></tr>
    """
    if mode != "baseline":
        rows += f"""
      <tr><td style="color:#b95b00;padding:3px 4px;font-size:12px;">IR (masks / step)</td>
          <td style="color:#b95b00;padding:3px 4px;font-size:12px;font-weight:700;">{ir:.1f}%</td></tr>
      <tr><td style="color:#1a6b2e;padding:3px 4px;font-size:12px;">OSR (saves / constrained)</td>
          <td style="color:#1a6b2e;padding:3px 4px;font-size:12px;font-weight:700;">{osr:.1f}%</td></tr>
      <tr><td {TD_L}>Constrained steps</td>
          <td {TD_R}>{cs} ({ma} masked · {opp} saved)</td></tr>
        """
    if mode == "adaptive":
        rows += f"""
      <tr><td style="color:#7a6000;padding:3px 4px;font-size:12px;">Entropy gate skips</td>
          <td style="color:#7a6000;padding:3px 4px;font-size:12px;font-weight:700;">{ent}</td></tr>
        """

    return f"""
    <div style="padding:10px;border:1px solid #ccc;border-radius:6px;
                background:#f8f8f8;color:#111;">
      <div style="font-weight:700;font-size:13px;margin-bottom:5px;color:#111;">
        LWG Compliance {delta_html}</div>
      <div style="background:#ddd;border-radius:3px;height:12px;margin-bottom:5px;">
        <div style="background:{bar_color};width:{bar_w}%;height:12px;border-radius:3px;"></div>
      </div>
      <div style="font-size:22px;font-weight:800;color:{bar_color};margin-bottom:8px;">
        {c:.1f}%</div>
      <table style="width:100%;border-collapse:collapse;">
        {rows}
      </table>
    </div>"""


def trace_table_html(trace_log: list[dict], tokenizer) -> str:
    """Decode trace table showing only non-FREE steps."""
    ACTION_BG = {
        "OPP_SAVE":          "#eafaea",
        "MASK_APPLIED":      "#fff3e0",
        "ENTROPY_SKIP":      "#fefce8",
        "PASS_NO_VALID_IDS": "#e8f4f8",
    }

    rows: list[str] = []
    for entry in trace_log:
        action = entry.get("action", "PASS")
        if action in ("PASS", "INIT"):
            continue

        bg   = ACTION_BG.get(action, "white")
        step = entry.get("step", "")
        ct   = _esc(entry.get("committed_token", "").replace("▁", "·"))
        eff  = entry.get("eff_state", "")
        fsm_b = entry.get("fsm_before", "")
        fsm_str = f"{fsm_b} → {eff}" if (fsm_b and fsm_b != eff) else eff

        detail = ""
        if action == "MASK_APPLIED":
            g  = _esc(entry.get("greedy_str", "?").replace("▁", "·"))
            f  = _esc(entry.get("forced_str", "?").replace("▁", "·"))
            gp = entry.get("greedy_prob", 0)
            fp = entry.get("forced_prob", 0)
            detail = (
                f'<b style="color:#c00">{g}</b> ({gp:.2f}) '
                f'→ <b style="color:#060">{f}</b> ({fp:.2f})'
            )
        elif action == "OPP_SAVE":
            g  = _esc(entry.get("greedy_str", "?").replace("▁", "·"))
            gp = entry.get("greedy_prob", 0)
            detail = f'✓ <b>{g}</b> ({gp:.2f}) valid'
        elif action == "ENTROPY_SKIP":
            H  = entry.get("entropy", "")
            TG = entry.get("T_G", "")
            try:
                detail = f"H={float(H):.3f} ≤ T_G={float(TG):.3f}"
            except (ValueError, TypeError):
                detail = f"H={H}"

        _td = 'style="padding:4px 7px;border-bottom:1px solid #e8e8e8;color:#111;font-size:12px;"'
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td style="padding:4px 7px;border-bottom:1px solid #e8e8e8;color:#555;font-size:12px;">{step}</td>'
            f'<td {_td}><code style="color:#111;font-size:12px;">{ct}</code></td>'
            f'<td style="padding:4px 7px;border-bottom:1px solid #e8e8e8;color:#333;font-size:11px;">{fsm_str}</td>'
            f'<td style="padding:4px 7px;border-bottom:1px solid #e8e8e8;font-weight:800;font-size:12px;color:#111;">{action}</td>'
            f'<td {_td}><code style="font-size:12px;">{detail}</code></td>'
            f'</tr>'
        )

    if not rows:
        return (
            '<p style="color:#444;font-size:13px;padding:8px;">'
            'No constrained steps — model was fully LWG-compliant without any intervention.'
            '</p>'
        )

    header = (
        '<tr style="background:#eeeeee;">'
        '<th style="padding:5px 7px;text-align:left;color:#111;font-size:12px;">Step</th>'
        '<th style="padding:5px 7px;text-align:left;color:#111;font-size:12px;">Token in</th>'
        '<th style="padding:5px 7px;text-align:left;color:#111;font-size:12px;">FSM state</th>'
        '<th style="padding:5px 7px;text-align:left;color:#111;font-size:12px;">Action</th>'
        '<th style="padding:5px 7px;text-align:left;color:#111;font-size:12px;">Detail</th>'
        '</tr>'
    )
    return (
        '<div style="max-height:380px;overflow-y:auto;border:1px solid #ccc;'
        'border-radius:6px;background:#fff;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead style="position:sticky;top:0;z-index:1;">{header}</thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table></div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gradio event handler
# ─────────────────────────────────────────────────────────────────────────────

_FONT = (
    "font-family:'Noto Sans Devanagari',sans-serif;"
    "font-size:15px;line-height:1.85;color:#111111;font-weight:500;"
)
_OUT_WRAP = f'<div style="{_FONT}padding:6px 4px;">'

_COL_HEADERS_HTML = {
    "baseline":    ('<b style="color:#1a2942;font-size:14px;">📄 Baseline</b>'
                    '<span style="font-size:11px;color:#666;margin-left:6px;">'
                    'autoregressive, no constraint</span>'),
    "constrained": ('<b style="color:#1a5c2e;font-size:14px;">🔒 SCG</b>'
                    '<span style="font-size:11px;color:#666;margin-left:6px;">'
                    'opportunistic masking (DOMINO)</span>'),
    "adaptive":    ('<b style="color:#1a3d6e;font-size:14px;">🎯 AdaSCG</b>'
                    '<span style="font-size:11px;color:#666;margin-left:6px;">'
                    'entropy-gated (AdaSD)</span>'),
}

_COL_BORDER = {
    "baseline":    "#95a5a6",
    "constrained": "#27ae60",
    "adaptive":    "#2980b9",
}

_EMPTY_COMBINED = """
<div style="display:flex;gap:14px;overflow-x:auto;padding-bottom:4px;">
  <div style="flex:1;min-width:340px;border:1px solid #ddd;border-radius:8px;
              padding:14px;background:#fff;color:#888;font-style:italic;">
    Baseline output will appear here.
  </div>
  <div style="flex:1;min-width:340px;border:1px solid #ddd;border-radius:8px;
              padding:14px;background:#fff;color:#888;font-style:italic;">
    SCG output will appear here.
  </div>
  <div style="flex:1;min-width:340px;border:1px solid #ddd;border-radius:8px;
              padding:14px;background:#fff;color:#888;font-style:italic;">
    AdaSCG output will appear here.
  </div>
</div>
"""


def _col_panel(mode: str, output_html: str, diff_html: str, s_html: str) -> str:
    """Build one column panel (header + output + optional diff + stats)."""
    border = _COL_BORDER[mode]
    header = _COL_HEADERS_HTML[mode]

    diff_section = ""
    if diff_html:
        diff_section = (
            '<div style="margin-top:10px;padding-top:8px;border-top:1px dashed #bbb;">'
            '<span style="font-size:11px;font-weight:700;color:#555;letter-spacing:.5px;">'
            '▶ VS BASELINE</span><br>'
            f'{_OUT_WRAP}{diff_html}</div>'
            '</div>'
        )

    return f"""
    <div style="flex:1;min-width:340px;border:2px solid {border};border-radius:8px;
                padding:14px;background:#ffffff;display:flex;flex-direction:column;gap:10px;">
      <div style="border-bottom:1px solid #e8e8e8;padding-bottom:8px;">{header}</div>
      <div style="max-height:380px;overflow-y:auto;">
        {_OUT_WRAP}{output_html}</div>
        {diff_section}
      </div>
      {s_html}
    </div>"""


def on_generate(
    prompt: str,
    model_label: str,
    max_tokens: int,
    T_G_val: float,
) -> tuple:
    if not prompt or not prompt.strip():
        return (_EMPTY_COMBINED, "", "")

    model_key = MODEL_LABELS.get(model_label, "airavata")

    try:
        results, tok = run_all_modes(prompt, model_key, int(max_tokens), T_G_init=T_G_val)
    except Exception as exc:
        import traceback
        tb = _esc(traceback.format_exc())
        err_panel = (
            f'<div style="color:#cc0000;font-weight:700;margin-bottom:6px;">'
            f'Error: {_esc(str(exc))}</div>'
            f'<pre style="font-size:11px;color:#555;white-space:pre-wrap;">{tb}</pre>'
        )
        combined = (
            '<div style="display:flex;gap:14px;overflow-x:auto;">'
            f'<div style="flex:1;min-width:340px;padding:14px;border:2px solid #e74c3c;'
            f'border-radius:8px;background:#fff;">{err_panel}</div>'
            '</div>'
        )
        return (combined, "", "")

    base_r = results["baseline"]
    scg_r  = results["constrained"]
    ada_r  = results["adaptive"]
    base_c = base_r["compliance"]

    # ── Build per-column content ─────────────────────────────────────────
    base_text = _esc(base_r["response"]).replace("\n", "<br>")
    base_s    = stats_html("baseline", base_r, base_c)
    base_col  = _col_panel("baseline", base_text, "", base_s)

    scg_act  = render_action_html(scg_r["new_ids"], scg_r["trace_log"], tok)
    scg_diff = render_diff_html(scg_r["new_ids"], base_r["new_ids"], tok)
    scg_s    = stats_html("constrained", scg_r, base_c)
    scg_col  = _col_panel("constrained", scg_act, scg_diff, scg_s)

    ada_act  = render_action_html(ada_r["new_ids"], ada_r["trace_log"], tok)
    ada_diff = render_diff_html(ada_r["new_ids"], base_r["new_ids"], tok)
    ada_s    = stats_html("adaptive", ada_r, base_c)
    ada_col  = _col_panel("adaptive", ada_act, ada_diff, ada_s)

    combined = (
        '<div style="display:flex;gap:14px;overflow-x:auto;overflow-y:visible;'
        'padding-bottom:6px;align-items:flex-start;">'
        f'{base_col}{scg_col}{ada_col}'
        '</div>'
    )

    scg_tr = trace_table_html(scg_r["trace_log"], tok)
    ada_tr = trace_table_html(ada_r["trace_log"], tok)

    return combined, scg_tr, ada_tr

# ─────────────────────────────────────────────────────────────────────────────
# Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    ["महात्मा गांधी के जीवन और स्वतंत्रता संग्राम में उनके योगदान के बारे में बताइए।"],
    ["भारत में पर्यावरण प्रदूषण की समस्या और उसके समाधान पर एक निबंध लिखिए।"],
    ["एक छोटी हिंदी कहानी लिखिए जिसमें एक बच्चा जंगल में खो जाता है।"],
    ["आधुनिक तकनीक ने हमारे जीवन को कैसे बदला है? इस पर अपने विचार प्रस्तुत करें।"],
    ["रामायण के प्रमुख पात्रों का परिचय दीजिए और उनके गुण-दोष बताइए।"],
    ["भारत की संसदीय प्रणाली कैसे काम करती है? विस्तार से समझाइए।"],
    ["जल संरक्षण के महत्व पर एक भाषण तैयार करें।"],
    ["भारत के प्रमुख त्योहारों का वर्णन कीजिए और उनके सांस्कृतिक महत्व को समझाइए।"],
]

LEGEND_HTML = """
<div style="display:flex;gap:10px;flex-wrap:wrap;padding:8px 12px;
     background:#f7f7f7;border-radius:6px;font-size:12px;border:1px solid #e8e8e8;">
  <b style="color:#555;align-self:center;">Token highlights:</b>
  <span style="background:#c8f7c5;color:#1a6b1a;padding:2px 9px;border-radius:4px;">
    ✓ Opportunistic Save — greedy already LWG-valid</span>
  <span style="background:#ffd9a0;color:#7a3b00;padding:2px 9px;border-radius:4px;">
    ⚡ Mask Applied — constraint forced a valid token</span>
  <span style="background:#fffacc;color:#6b5e00;padding:2px 9px;border-radius:4px;">
    ⊘ Entropy Skip — AdaSCG gate suppressed constraint</span>
  <span style="background:#ffe0e0;color:#8b0000;padding:2px 9px;border-radius:4px;
        border:1px solid #c00;">▲ Differs from baseline</span>
  <span style="color:#999;font-style:italic;align-self:center;">
    hover any highlighted token for details</span>
</div>
"""

TITLE_HTML = """
<div style="padding:12px 0 4px;">
  <h1 style="margin:0;font-size:22px;color:#1a2942;">
    🔤 Samanvaya-Constrained Generation (SCG) — Interactive Demo
  </h1>
  <p style="margin:6px 0 0;color:#555;font-size:13px;">
    Compare Baseline · SCG (opportunistic masking) · AdaSCG (entropy-gated) side-by-side.<br>
    Both models are preloaded on separate GPUs — switching between them is instant.
  </p>
</div>
"""

def build_demo() -> gr.Blocks:
    google_font = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari'
        ':wght@400;500;600&display=swap" rel="stylesheet">'
    )

    css = """
    /* Force dark text inside HTML components — Gradio Soft theme can wash these out */
    .scg-output, .scg-output * {
        color: #111111 !important;
    }
    .scg-output em { color: #888 !important; font-style: italic; }
    /* Remove default Gradio padding around HTML blocks */
    .scg-output > div { padding: 0 !important; }
    """

    with gr.Blocks(
        title="SCG Demo — Samanvaya-Constrained Hindi Generation",
        theme=gr.themes.Default(),
        head=google_font,
        css=css,
    ) as demo:

        gr.HTML(TITLE_HTML)

        # ── Controls ───────────────────────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODEL_LABELS.keys()),
                value="Airavata 7B (ai4bharat)",
                label="Model (both preloaded)",
                scale=3,
            )
            max_tok = gr.Slider(
                minimum=50, maximum=400, value=200, step=10,
                label="Max new tokens",
                scale=2,
            )
            T_G_sl = gr.Slider(
                minimum=0.1, maximum=3.0, value=0.5, step=0.05,
                label="AdaSCG initial T_G (entropy threshold)",
                scale=2,
            )

        prompt_tb = gr.Textbox(
            label="Hindi Prompt",
            placeholder="हिंदी में प्रश्न या विषय लिखें... (Enter to generate)",
            lines=3,
        )

        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=[prompt_tb],
            label="Example prompts (click to load)",
        )

        gen_btn = gr.Button("⚡  Generate all three modes", variant="primary", size="lg")

        gr.HTML(LEGEND_HTML)

        # ── Combined 3-column output (single HTML, horizontal scroll inside) ──
        # Using one gr.HTML avoids Gradio's page-level horizontal overflow.
        # Each column has min-width so they never squish; the wrapper scrolls.
        combined_out = gr.HTML(
            value=_EMPTY_COMBINED,
            elem_classes=["scg-output"],
        )

        # ── Decode trace accordions ────────────────────────────────────────
        with gr.Accordion("🔍 SCG Decode Trace — constrained steps only", open=False):
            gr.HTML(
                '<p style="font-size:12px;color:#444;margin:4px 0 8px;">'
                'PASS and INIT steps are hidden. '
                '<b>MASK_APPLIED</b>: rejected greedy → forced token. '
                '<b>OPP_SAVE</b>: greedy was already LWG-valid (no intervention needed).'
                '</p>'
            )
            scg_trace = gr.HTML(elem_classes=["scg-output"])

        with gr.Accordion("🔍 AdaSCG Decode Trace — constrained steps only", open=False):
            gr.HTML(
                '<p style="font-size:12px;color:#444;margin:4px 0 8px;">'
                '<b>ENTROPY_SKIP</b>: H ≤ T_G — model was confident, mask suppressed. '
                'T_G updates online as rolling mean of entropies at constrained steps.'
                '</p>'
            )
            ada_trace = gr.HTML(elem_classes=["scg-output"])

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML(
            '<div style="margin-top:16px;padding-top:12px;border-top:1px solid #ddd;'
            'font-size:11px;color:#666;text-align:center;">'
            'SCG · IIT Bombay MTP Stage II · Samanvaya LWG rules injected at inference time · '
            'No training changes · DOMINO opportunistic masking + AdaSD entropy gate'
            '</div>'
        )

        # ── Wire ──────────────────────────────────────────────────────────
        _inputs  = [prompt_tb, model_dd, max_tok, T_G_sl]
        _outputs = [combined_out, scg_trace, ada_trace]

        gen_btn.click(fn=on_generate, inputs=_inputs, outputs=_outputs)
        prompt_tb.submit(fn=on_generate, inputs=_inputs, outputs=_outputs)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preload_all()
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
