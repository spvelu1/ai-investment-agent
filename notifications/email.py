from __future__ import annotations

import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_daily_report(
    account: dict,
    positions: list[dict],
    signals_summary: str = "",
    *,
    email_from: str,
    email_to: str,
    app_password: str,
) -> None:
    """Send formatted daily portfolio report via Gmail SMTP."""
    if not email_from or not app_password:
        raise ValueError(
            "EMAIL_FROM and GMAIL_APP_PASSWORD must be set in .env to send reports. "
            "Create a Gmail App Password at https://myaccount.google.com/apppasswords"
        )

    subject = f"📈 Investment Agent Report — {date.today()}"
    html = _build_html(account, positions, signals_summary)
    text = _build_text(account, positions, signals_summary)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(email_from, app_password)
        smtp.sendmail(email_from, email_to, msg.as_string())

    logger.info("Daily report sent to %s", email_to)


def _build_text(account: dict, positions: list[dict], signals_summary: str) -> str:
    lines = [
        f"AI Investment Agent — Daily Report ({date.today()})",
        "=" * 52,
        "",
        "ACCOUNT",
    ]
    for k, v in account.items():
        lines.append(f"  {k}: ${v:,.2f}")

    if positions:
        lines += ["", f"OPEN POSITIONS ({len(positions)})"]
        lines.append(f"  {'Symbol':<8} {'Qty':>6} {'Entry':>9} {'Current':>9} {'P&L%':>7}")
        lines.append("  " + "-" * 45)
        for p in sorted(positions, key=lambda x: -x["market_value"]):
            pnl = p["unrealized_plpc"] * 100
            lines.append(
                f"  {p['symbol']:<8} {p['qty']:>6} "
                f"${p['avg_entry_price']:>8,.2f} "
                f"${p['current_price']:>8,.2f} "
                f"{pnl:>+6.2f}%"
            )
        total_pnl = sum(p["unrealized_pl"] for p in positions)
        lines.append(f"\n  Total Unrealized P&L: ${total_pnl:+,.2f}")
    else:
        lines += ["", "No open positions (all cash)."]

    if signals_summary:
        lines += ["", "NOTES", signals_summary]

    lines += ["", "—", "Sent by ai-investment-agent"]
    return "\n".join(lines)


def _build_html(account: dict, positions: list[dict], signals_summary: str) -> str:
    acct_rows = "".join(
        f"<tr><td>{k}</td><td><b>${v:,.2f}</b></td></tr>" for k, v in account.items()
    )

    if positions:
        pos_rows = ""
        for p in sorted(positions, key=lambda x: -x["market_value"]):
            pnl = p["unrealized_plpc"] * 100
            color = "#16a34a" if pnl >= 0 else "#dc2626"
            pos_rows += (
                f"<tr>"
                f"<td><b>{p['symbol']}</b></td>"
                f"<td style='text-align:right'>{p['qty']}</td>"
                f"<td style='text-align:right'>${p['avg_entry_price']:,.2f}</td>"
                f"<td style='text-align:right'>${p['current_price']:,.2f}</td>"
                f"<td style='text-align:right'>${p['market_value']:,.2f}</td>"
                f"<td style='text-align:right;color:{color}'><b>{pnl:+.2f}%</b></td>"
                f"</tr>"
            )
        total_pnl = sum(p["unrealized_pl"] for p in positions)
        color = "#16a34a" if total_pnl >= 0 else "#dc2626"
        positions_html = f"""
        <h2 style="color:#1e293b">Open Positions ({len(positions)})</h2>
        <table style="border-collapse:collapse;width:100%;font-size:14px">
          <thead>
            <tr style="background:#f1f5f9">
              <th style="padding:8px;text-align:left">Symbol</th>
              <th style="padding:8px;text-align:right">Qty</th>
              <th style="padding:8px;text-align:right">Entry</th>
              <th style="padding:8px;text-align:right">Current</th>
              <th style="padding:8px;text-align:right">Mkt Val</th>
              <th style="padding:8px;text-align:right">P&amp;L %</th>
            </tr>
          </thead>
          <tbody>{pos_rows}</tbody>
        </table>
        <p style="font-size:14px">
          Total Unrealized P&amp;L:
          <b style="color:{color}">${total_pnl:+,.2f}</b>
        </p>
        """
    else:
        positions_html = "<p>No open positions &mdash; all cash.</p>"

    notes_html = (
        f"<h2 style='color:#1e293b'>Notes</h2><p style='font-size:14px'>{signals_summary}</p>"
        if signals_summary else ""
    )

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;color:#0f172a">
      <h1 style="color:#1e3a8a">AI Investment Agent</h1>
      <p style="color:#64748b">{date.today()} &mdash; Daily Report</p>
      <hr style="border:1px solid #e2e8f0"/>
      <h2 style="color:#1e293b">Account Summary</h2>
      <table style="border-collapse:collapse;font-size:14px">
        <tbody>{acct_rows}</tbody>
      </table>
      {positions_html}
      {notes_html}
      <hr style="border:1px solid #e2e8f0"/>
      <p style="color:#94a3b8;font-size:12px">Sent by ai-investment-agent</p>
    </body></html>
    """
