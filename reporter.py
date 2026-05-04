"""
╔══════════════════════════════════════════════════════════════╗
║              HTML REPORT GENERATOR                           ║
║   Generates beautiful self-contained HTML reports           ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from .engine import CheckResult, DocumentPair


VERDICT_COLOR = {
    "HIGHLY_PLAGIARISED": "#ef4444",
    "LIKELY_PLAGIARISED":  "#f97316",
    "SUSPICIOUS":          "#eab308",
    "MINOR_SIMILARITY":    "#22d3ee",
    "ORIGINAL":            "#22c55e",
}

VERDICT_LABEL = {
    "HIGHLY_PLAGIARISED": "🔴 Highly Plagiarised",
    "LIKELY_PLAGIARISED":  "🟠 Likely Plagiarised",
    "SUSPICIOUS":          "🟡 Suspicious",
    "MINOR_SIMILARITY":    "🟢 Minor Similarity",
    "ORIGINAL":            "✅ Original",
}


def _score_bar(score: float) -> str:
    pct = score * 100
    color = VERDICT_COLOR.get(
        "HIGHLY_PLAGIARISED" if pct >= 90 else
        "LIKELY_PLAGIARISED"  if pct >= 75 else
        "SUSPICIOUS"          if pct >= 50 else
        "MINOR_SIMILARITY"    if pct >= 30 else "ORIGINAL"
    )
    return f"""
        <div class="bar-wrap">
            <div class="bar" style="width:{pct:.1f}%; background:{color};"></div>
            <span class="bar-label">{pct:.1f}%</span>
        </div>"""


def generate_html_report(result: CheckResult, output_path: str = "plagiarism_report.html") -> str:
    """Generate a fully self-contained, interactive HTML report."""

    pairs_json = json.dumps([p.to_dict() for p in result.pairs], indent=2)

    pair_cards = ""
    sorted_pairs = sorted(result.pairs, key=lambda p: p.combined_score, reverse=True)
    for pair in sorted_pairs:
        color = VERDICT_COLOR.get(pair.verdict, "#6b7280")
        label = VERDICT_LABEL.get(pair.verdict, pair.verdict)
        phrases_html = ""
        if pair.highlighted_sections:
            phrases_html = "<div class='phrases'><strong>Matching Phrases:</strong><ul>" + \
                "".join(f'<li>&#8220;{p}&#8221;</li>' for p in pair.highlighted_sections[:5]) + \
                "</ul></div>"

        pair_cards += f"""
        <div class="card" data-score="{pair.combined_score:.4f}" data-verdict="{pair.verdict}">
            <div class="card-header" style="border-left: 4px solid {color};">
                <span class="verdict-badge" style="background:{color}20; color:{color}; border:1px solid {color};">{label}</span>
                <span class="doc-names">📄 {pair.doc1} <span class="arrow">↔</span> {pair.doc2}</span>
            </div>
            <div class="card-body">
                <div class="scores">
                    <div class="score-row"><span>Cosine Similarity</span>{_score_bar(pair.cosine_score)}</div>
                    <div class="score-row"><span>Jaccard Similarity</span>{_score_bar(pair.jaccard_score)}</div>
                    <div class="score-row"><span>N-gram Similarity</span>{_score_bar(pair.ngram_score)}</div>
                    <div class="score-row combined"><span><strong>Combined Score</strong></span>{_score_bar(pair.combined_score)}</div>
                </div>
                <div class="meta">
                    <span>🔡 Shared words: <strong>{len(pair.word_overlap)}</strong></span>
                    <span>⏱ {pair.execution_time*1000:.1f}ms</span>
                </div>
                {phrases_html}
            </div>
        </div>"""

    total_original = result.total_pairs - result.plagiarised_pairs
    plag_rate = result.plagiarism_rate

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Plagiarism Report — {result.timestamp[:10]}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface2: #242740;
    --border: #2d3158;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #6366f1;
    --red: #ef4444;
    --orange: #f97316;
    --yellow: #eab308;
    --cyan: #22d3ee;
    --green: #22c55e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}
  header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    padding: 40px 24px 32px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.3) 0%, transparent 70%);
  }}
  h1 {{
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
  }}
  .subtitle {{ color: var(--muted); margin-top: 8px; font-size: 0.95rem; position: relative; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    padding: 24px;
    max-width: 1100px;
    margin: 0 auto;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
  }}
  .stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }}
  .stat-num {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
  .stat-label {{ color: var(--muted); font-size: 0.8rem; margin-top: 6px; text-transform: uppercase; letter-spacing: .05em; }}
  .filter-bar {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px 16px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-btn {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all .2s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }}
  .search-box {{
    margin-left: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 14px;
    border-radius: 8px;
    font-size: 0.9rem;
    width: 220px;
    outline: none;
  }}
  .search-box:focus {{ border-color: var(--accent); }}
  .pairs-container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px 48px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    transition: box-shadow .2s, transform .2s;
  }}
  .card:hover {{ box-shadow: 0 6px 28px rgba(0,0,0,.5); transform: translateY(-1px); }}
  .card-header {{
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface2);
    cursor: pointer;
    user-select: none;
  }}
  .verdict-badge {{
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
  }}
  .doc-names {{ font-size: 0.9rem; color: var(--muted); }}
  .arrow {{ color: var(--accent); font-weight: bold; margin: 0 6px; }}
  .card-body {{ padding: 18px 20px; }}
  .scores {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }}
  .score-row {{ display: flex; align-items: center; gap: 12px; font-size: 0.88rem; }}
  .score-row span:first-child {{ width: 160px; flex-shrink: 0; color: var(--muted); }}
  .score-row.combined span:first-child {{ color: var(--text); }}
  .bar-wrap {{ flex: 1; display: flex; align-items: center; gap: 8px; }}
  .bar {{
    height: 8px;
    border-radius: 4px;
    transition: width .6s ease;
    min-width: 2px;
  }}
  .bar-label {{ font-size: 0.85rem; font-weight: 700; white-space: nowrap; }}
  .meta {{ display: flex; gap: 20px; font-size: 0.8rem; color: var(--muted); margin-bottom: 10px; }}
  .phrases {{ margin-top: 12px; padding: 12px; background: var(--bg); border-radius: 8px; font-size: 0.85rem; }}
  .phrases ul {{ margin-top: 6px; padding-left: 20px; }}
  .phrases li {{ margin-top: 4px; color: var(--muted); font-style: italic; }}
  .donut-wrap {{
    display: flex;
    justify-content: center;
    padding: 20px 0 10px;
  }}
  canvas {{ max-width: 200px; }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 24px;
    border-top: 1px solid var(--border);
  }}
  .no-results {{ text-align:center; padding:60px; color:var(--muted); }}
</style>
</head>
<body>

<header>
  <h1>🔍 Plagiarism Check Report</h1>
  <p class="subtitle">Generated: {result.timestamp} &nbsp;·&nbsp; Powered by TF-IDF + Cosine Similarity</p>
</header>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-num" style="color:var(--cyan)">{result.total_documents}</div>
    <div class="stat-label">Documents</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:var(--accent)">{result.total_pairs}</div>
    <div class="stat-label">Pairs Compared</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:var(--red)">{result.plagiarised_pairs}</div>
    <div class="stat-label">Flagged Pairs</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:{'var(--red)' if plag_rate>50 else 'var(--yellow)' if plag_rate>20 else 'var(--green)'}">{plag_rate:.1f}%</div>
    <div class="stat-label">Plagiarism Rate</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:var(--green)">{total_original}</div>
    <div class="stat-label">Original Pairs</div>
  </div>
  <div class="stat-card">
    <div class="stat-num" style="color:var(--muted)">{result.duration:.2f}s</div>
    <div class="stat-label">Analysis Time</div>
  </div>
</div>

<div class="filter-bar">
  <span style="color:var(--muted); font-size:.85rem; font-weight:600;">FILTER:</span>
  <button class="filter-btn active" onclick="filterVerdict('ALL')">All</button>
  <button class="filter-btn" onclick="filterVerdict('HIGHLY_PLAGIARISED')" style="border-color:#ef4444">🔴 High</button>
  <button class="filter-btn" onclick="filterVerdict('LIKELY_PLAGIARISED')" style="border-color:#f97316">🟠 Likely</button>
  <button class="filter-btn" onclick="filterVerdict('SUSPICIOUS')" style="border-color:#eab308">🟡 Suspicious</button>
  <button class="filter-btn" onclick="filterVerdict('ORIGINAL')" style="border-color:#22c55e">✅ Original</button>
  <input class="search-box" type="text" placeholder="🔍 Search documents..." oninput="searchCards(this.value)"/>
</div>

<div class="pairs-container" id="pairs">
{pair_cards if pair_cards else '<div class="no-results">✅ No pairs to display.</div>'}
</div>

<footer>
  Plagiarism Checker Pro v2.0 &nbsp;·&nbsp; github.com/Aranya2801/Plagiarism-checker-Python &nbsp;·&nbsp;
  Built with Python · scikit-learn · TF-IDF
</footer>

<script>
const allCards = document.querySelectorAll('.card');
function filterVerdict(v) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  allCards.forEach(c => {{
    c.style.display = (v === 'ALL' || c.dataset.verdict === v) ? '' : 'none';
  }});
}}
function searchCards(q) {{
  const lq = q.toLowerCase();
  allCards.forEach(c => {{
    c.style.display = c.textContent.toLowerCase().includes(lq) ? '' : 'none';
  }});
}}
// Animate bars on load
setTimeout(() => {{
  document.querySelectorAll('.bar').forEach(b => {{
    b.style.transition = 'width 0.8s cubic-bezier(.4,0,.2,1)';
  }});
}}, 100);
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
