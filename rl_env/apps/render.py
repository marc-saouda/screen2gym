"""Server-side HTML rendering for the three simulated apps.

Pages are fully server-rendered and reflect current state on every GET, so the
environment works identically whether driven by a real browser (Playwright) or
by plain HTTP requests in the headless fallback path. Forms use standard POST +
303 redirect so both a human and an automated agent can complete the task.
"""
from __future__ import annotations

import html
import re
from typing import Optional

from state import SimState, normalize_channel

EMOJI = {
    ":rotating_light:": "\U0001F6A8",
    ":anchor:": "\u2693",
    ":ship:": "\U0001F6A2",
    ":sailboat:": "\u26F5",
    ":wave:": "\U0001F44B",
    ":calendar:": "\U0001F4C5",
    ":bell:": "\U0001F514",
}

STYLE = """
:root{--bg:#f6f8fa;--ink:#1b1f24;--muted:#5b6470;--line:#d8dee4;--accent:#1264a3;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);}
a{color:var(--accent);text-decoration:none;}
.topnav{display:flex;align-items:center;gap:4px;background:#0b1f33;padding:0 14px;height:48px;color:#fff;}
.topnav .brand{font-weight:700;margin-right:14px;letter-spacing:.2px;}
.topnav a{color:#cdd9e5;padding:8px 14px;border-radius:6px;font-size:14px;}
.topnav a.active{background:#1264a3;color:#fff;}
.topnav a:hover{background:#16324f;color:#fff;}
.wrap{max-width:1100px;margin:18px auto;padding:0 16px;}
.brief{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:14px 16px;margin-bottom:16px;}
.brief h2{margin:0 0 6px;font-size:15px;}
.brief p{margin:4px 0;color:var(--muted);font-size:13px;line-height:1.5;}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px;}
.card h3{margin:0 0 12px;font-size:14px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);}
/* Spreadsheet */
.sheet-title{display:flex;align-items:center;gap:10px;font-weight:600;margin-bottom:10px;}
.sheet-title .tag{background:#188038;color:#fff;font-size:11px;padding:2px 8px;border-radius:4px;}
table.grid{border-collapse:collapse;width:100%;font-size:13px;}
table.grid th,table.grid td{border:1px solid #e1e4e8;padding:6px 9px;text-align:left;white-space:nowrap;}
table.grid thead th{background:#f1f3f5;position:sticky;top:0;}
table.grid th.rownum,table.grid td.rownum{background:#f8f9fa;color:#98a0a8;text-align:center;width:38px;}
table.grid tr.newrow td{background:#fff8e1;}
.formgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.formgrid label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px;}
input,textarea,select{font-family:inherit;font-size:13px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);}
textarea{min-height:130px;resize:vertical;}
.btn{display:inline-flex;align-items:center;gap:6px;background:var(--accent);color:#fff;border:none;padding:9px 16px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;}
.btn:hover{filter:brightness(1.07);}
.btn.secondary{background:#fff;color:var(--accent);border:1px solid var(--accent);}
.btn.ghost{background:#eef1f4;color:var(--ink);}
.row-actions{margin-top:12px;display:flex;gap:10px;align-items:center;}
/* Slack */
.slack{display:grid;grid-template-columns:230px 1fr;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;min-height:520px;}
.slack .side{background:#3f0e40;color:#cfc3cf;padding:14px 10px;}
.slack .side .ws{font-weight:700;color:#fff;font-size:15px;padding:4px 8px 12px;border-bottom:1px solid #522653;margin-bottom:10px;}
.slack .side .chan{display:block;padding:6px 10px;border-radius:6px;color:#d9c9d9;font-size:14px;margin-bottom:2px;}
.slack .side .chan.active{background:#1164a3;color:#fff;}
.slack .side .chan:hover{background:#52275420;}
.slack .main{display:flex;flex-direction:column;}
.slack .chanhead{border-bottom:1px solid var(--line);padding:12px 18px;font-weight:700;font-size:16px;}
.slack .msgs{padding:14px 18px;flex:1;overflow:auto;}
.msg{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #f0f0f0;}
.msg .avatar{width:36px;height:36px;border-radius:8px;background:#1264a3;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;}
.msg.bot .avatar{background:#611f69;}
.msg .body{font-size:14px;line-height:1.5;}
.msg .who{font-weight:700;margin-right:8px;}
.msg .bot-badge{background:#e8d9ee;color:#611f69;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;margin-left:4px;vertical-align:middle;}
.msg .when{color:var(--muted);font-size:11px;}
.msg .text{margin-top:3px;white-space:pre-wrap;}
.empty{color:var(--muted);padding:30px 0;text-align:center;font-size:14px;}
/* Zapier */
.zstep{border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:12px;}
.zstep .hd{display:flex;align-items:center;gap:10px;font-weight:700;margin-bottom:8px;}
.zstep .num{width:24px;height:24px;border-radius:50%;background:#ff4f00;color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;}
.pill{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:700;}
.pill.on{background:#e6f4ea;color:#188038;}
.pill.off{background:#fdeceb;color:#c5221f;}
.status{font-size:13px;margin-bottom:10px;}
.help{color:var(--muted);font-size:12px;margin:4px 0 0;}
"""


def _esc(text: str) -> str:
    return html.escape(text or "")


def _fmt_ts(ts: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")


def render_message_text(text: str) -> str:
    out = _esc(text)
    for code, ch in EMOJI.items():
        out = out.replace(_esc(code), ch)
    out = re.sub(r"\*([^*\n]+)\*", r"<strong>\1</strong>", out)
    return out


def page_shell(title: str, active: str, body: str) -> str:
    def nav(href: str, key: str, label: str) -> str:
        cls = "active" if key == active else ""
        return f'<a class="{cls}" href="{href}">{label}</a>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_esc(title)}</title><style>{STYLE}</style></head>
<body>
<div class="topnav">
  <span class="brand">Salt &amp; Stone Workspace</span>
  {nav('/spreadsheet','spreadsheet','Sheets')}
  {nav('/zapier','zapier','Zapier')}
  {nav('/slack','slack','Slack')}
</div>
<div class="wrap">{body}</div>
</body></html>"""


def _next_booking_id(state: SimState) -> str:
    if not state.columns:
        return ""
    id_col = state.columns[0]
    best = 0
    prefix = "SST-"
    for row in state.rows:
        val = row.get(id_col, "")
        m = re.search(r"(\d+)\s*$", val)
        if m:
            best = max(best, int(m.group(1)))
    return f"{prefix}{best + 1:05d}"


def _readonly_grid(columns: list, rows: list) -> str:
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = []
    for i, row in enumerate(rows, start=1):
        cells = "".join(f"<td>{_esc(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f'<tr><td class="rownum">{i}</td>{cells}</tr>')
    body_html = "\n".join(body) or (
        f'<tr><td class="rownum"></td><td colspan="{max(1, len(columns))}" '
        f'style="color:#98a0a8">No rows.</td></tr>')
    return f"""<div style="overflow:auto;max-height:300px;border:1px solid #eef0f2;border-radius:6px">
        <table class="grid"><thead><tr><th class="rownum">#</th>{head}</tr></thead>
        <tbody>{body_html}</tbody></table></div>"""


def render_source(state: SimState) -> str:
    """Read-only source dataset (the web-relabeled WPS Office file)."""
    grid = _readonly_grid(state.source_columns, state.source_rows)
    return page_shell(
        "Source data - Sheets", "spreadsheet",
        f"""
    <div class="card">
      <div class="sheet-title"><span class="tag" style="background:#5b6470">SOURCE</span>
        Source booking data <span style="color:var(--muted);font-weight:400;font-size:12px">
        (read-only &middot; {len(state.source_rows)} rows)</span></div>
      <p class="help">This is the original booking data to copy into your working sheet.</p>
      {grid}
    </div>""")


def render_brief(state: SimState) -> str:
    """The task brief, presented as an in-app instructions panel."""
    crit = "".join(f"<li>{_esc(c)}</li>" for c in state.success_criteria)
    return page_shell(
        "Task brief", "spreadsheet",
        f"""
    <div class="brief">
      <h2>Task brief &mdash; Real-time Booking Alerts for Salt &amp; Stone Coastal Tours</h2>
      <p>{_esc(state.business_context)}</p>
      <p><strong>Goal:</strong> {_esc(state.task_goal)}</p>
      {'<p><strong>Success criteria:</strong></p><ul class="help">' + crit + '</ul>' if crit else ''}
    </div>""")


def render_spreadsheet(state: SimState) -> str:
    cols = state.columns
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body_rows = []
    for i, row in enumerate(state.rows, start=1):
        is_new = i > state.seed_row_count
        cells = "".join(f"<td>{_esc(row.get(c, ''))}</td>" for c in cols)
        cls = ' class="newrow"' if is_new else ""
        body_rows.append(f'<tr{cls}><td class="rownum">{i}</td>{cells}</tr>')
    rows_html = "\n".join(body_rows) or (
        f'<tr><td class="rownum"></td><td colspan="{max(1, len(cols))}" '
        f'style="color:#98a0a8">Empty sheet &mdash; add the booking rows below.</td></tr>')

    suggested = {c: "" for c in cols}
    if cols:
        suggested[cols[0]] = _next_booking_id(state)
    inputs = []
    for c in cols:
        inputs.append(
            f'<label>{_esc(c)}'
            f'<input name="{_esc(c)}" id="cell-{_esc(c)}" value="{_esc(suggested.get(c, ""))}" '
            f'autocomplete="off"/></label>'
        )
    inputs_html = "\n".join(inputs)

    brief = f"""
    <div class="brief">
      <h2>Task brief &mdash; Real-time Booking Alerts for Salt &amp; Stone Coastal Tours</h2>
      <p>{_esc(state.business_context)}</p>
      <p><strong>Your goal:</strong> {_esc(state.task_goal)}</p>
    </div>"""

    return page_shell(
        "Salt and Stone Booking - Sheets",
        "spreadsheet",
        f"""
    {brief}
    <div class="card">
      <div class="sheet-title"><span class="tag">SHEET</span> {_esc(state.sheet_title)}
        <span style="color:var(--muted);font-weight:400;font-size:12px">({len(state.rows)} rows)</span></div>
      <div style="overflow:auto;max-height:360px;border:1px solid #eef0f2;border-radius:6px">
        <table class="grid">
          <thead><tr><th class="rownum">#</th>{head}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>Add a new booking row</h3>
      <form method="POST" action="/api/spreadsheet/rows" id="add-row-form">
        <div class="formgrid">{inputs_html}</div>
        <div class="row-actions">
          <button class="btn" type="submit" id="add-row-btn">Add row</button>
          <span class="help">Submitting appends a row to the sheet. If the Zap is live, it fires the Slack alert.</span>
        </div>
      </form>
    </div>
    <div class="card">
      <h3>Source data (read-only)</h3>
      <p class="help" style="margin-bottom:10px">Copy these bookings into the working sheet above.
        Also at <a href="/source">/source</a>.</p>
      {_readonly_grid(state.source_columns, state.source_rows)}
    </div>""",
    )


def render_slack(state: SimState, channel: Optional[str] = None) -> str:
    current = normalize_channel(channel or state.target_channel) or state.target_channel
    exists = current in state.slack_channels

    side_links = []
    for ch in state.slack_channels:
        cls = "chan active" if ch == current else "chan"
        label = _esc(ch.lstrip("#"))
        side_links.append(f'<a class="{cls}" href="/slack?channel={_esc(ch.lstrip("#"))}"># {label}</a>')
    side_html = "\n".join(side_links)

    # Always-available "create channel" control. Prefilled with the requested
    # channel name when it does not exist yet, so the create flow is one click.
    create_prefill = "" if exists else _esc(current.lstrip("#"))
    create_form = f"""
        <form method="POST" action="/api/slack/channels" id="create-channel-form"
              style="margin-top:14px;padding:0 8px">
          <input name="name" id="create-channel-input" value="{create_prefill}"
                 placeholder="new-channel" autocomplete="off"
                 style="width:100%;background:#52275420;color:#fff;border:1px solid #6b3a6c;margin-bottom:6px"/>
          <button class="btn" id="create-channel-btn" type="submit"
                  style="width:100%;background:#1164a3">+ Create channel</button>
        </form>"""

    if not exists:
        msgs_html = (f'<div class="empty">#{_esc(current.lstrip("#"))} does not exist yet. '
                     f'Create it from the sidebar.</div>')
        return page_shell(
            f"Slack - {current}", "slack",
            f"""
    <div class="slack">
      <div class="side">
        <div class="ws">Salt &amp; Stone</div>
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:4px 8px;color:#b9a7b9">Channels</div>
        {side_html}
        {create_form}
      </div>
      <div class="main">
        <div class="chanhead"># {_esc(current.lstrip('#'))} <span class="help">(not created)</span></div>
        <div class="msgs" id="slack-messages">{msgs_html}</div>
      </div>
    </div>""")

    msgs = state.messages_for(current)
    if msgs:
        items = []
        for m in msgs:
            cls = "msg bot" if m.is_bot else "msg"
            initial = _esc((m.author or "?")[:1].upper())
            badge = '<span class="bot-badge">APP</span>' if m.is_bot else ""
            items.append(
                f'<div class="{cls}" data-source="{_esc(m.source)}">'
                f'<div class="avatar">{initial}</div>'
                f'<div class="body"><span class="who">{_esc(m.author)}</span>{badge} '
                f'<span class="when">{_fmt_ts(m.ts)}</span>'
                f'<div class="text">{render_message_text(m.text)}</div></div></div>'
            )
        msgs_html = "\n".join(items)
    else:
        msgs_html = '<div class="empty">No messages in this channel yet.</div>'

    return page_shell(
        f"Slack - {current}",
        "slack",
        f"""
    <div class="slack">
      <div class="side">
        <div class="ws">Salt &amp; Stone</div>
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:4px 8px;color:#b9a7b9">Channels</div>
        {side_html}
        {create_form}
      </div>
      <div class="main">
        <div class="chanhead"># {_esc(current.lstrip('#'))}</div>
        <div class="msgs" id="slack-messages">{msgs_html}</div>
      </div>
    </div>""",
    )


def _opt(value: str, current: str, label: str) -> str:
    sel = " selected" if (current or "") == value else ""
    return f'<option value="{_esc(value)}"{sel}>{_esc(label)}</option>'


def render_zapier(state: SimState) -> str:
    auto = state.automation
    configured = auto.is_configured()
    live = auto.enabled and configured
    status_pill = (
        '<span class="pill on">ON</span>' if live else '<span class="pill off">OFF / DRAFT</span>'
    )
    trig_pill = ('<span class="pill on">SET</span>' if auto.trigger_configured()
                 else '<span class="pill off">NOT SET</span>')
    template_val = auto.message_template or ""
    channel_val = auto.channel or ""

    app_opts = _opt("", auto.trigger_app, "Choose app...") + _opt("google_sheets", auto.trigger_app, "Google Sheets")
    evt_opts = _opt("", auto.trigger_event, "Choose event...") + _opt("new_row", auto.trigger_event, "New Spreadsheet Row")

    return page_shell(
        "Zapier - Zap editor",
        "zapier",
        f"""
    <div class="card">
      <h3>Zap editor</h3>
      <div class="status">Status: {status_pill} &nbsp; Trigger: {trig_pill}
        &nbsp;|&nbsp; Target channel for this task: <strong>{_esc(state.target_channel)}</strong></div>

      <form method="POST" action="/api/automation" id="zap-form">
        <div class="zstep">
          <div class="hd"><span class="num">1</span> Trigger &mdash; Google Sheets</div>
          <div class="formgrid" style="grid-template-columns:1fr 1fr 1fr">
            <label>Trigger app
              <select name="trigger_app" id="zap-trigger-app">{app_opts}</select>
            </label>
            <label>Trigger event
              <select name="trigger_event" id="zap-trigger-event">{evt_opts}</select>
            </label>
            <label>Spreadsheet
              <input name="trigger_sheet" id="zap-trigger-sheet" value="{_esc(auto.trigger_sheet)}"
                     placeholder="{_esc(state.sheet_title)}" autocomplete="off"/>
            </label>
          </div>
        </div>

        <div class="zstep">
          <div class="hd"><span class="num">2</span> Action &mdash; Slack: Send Channel Message</div>
          <div class="formgrid" style="grid-template-columns:1fr 1fr">
            <label>Channel
              <input name="channel" id="zap-channel" value="{_esc(channel_val)}" placeholder="#crew-alerts" autocomplete="off"/>
            </label>
            <label>Send as bot name
              <input name="bot_name" id="zap-bot" value="{_esc(auto.bot_name)}" autocomplete="off"/>
            </label>
          </div>
          <label style="display:block;margin-top:10px">Message template
            <textarea name="message_template" id="zap-template" placeholder="Compose the Slack message. Use {{Guest_Name}} style placeholders.">{_esc(template_val)}</textarea>
          </label>
          <p class="help">Available fields: {_esc(', '.join('{' + c + '}' for c in state.columns))}</p>
        </div>

        <div class="row-actions">
          <label style="display:flex;align-items:center;gap:8px;font-size:13px">
            <input type="checkbox" name="enabled" id="zap-enabled" value="true" {'checked' if auto.enabled else ''} style="width:auto"/>
            Turn Zap on
          </label>
          <button class="btn" type="submit" id="zap-save">Save &amp; publish Zap</button>
          <button class="btn secondary" type="submit" name="load_recommended" value="1" id="zap-load-template" formnovalidate>Load recommended template</button>
        </div>
      </form>
    </div>""",
    )
