"""
utils/alerts.py — Email + SMS repeat-violator alert system (Sprint 3).

Tracks per-plate violation counts and fires SMTP email + Twilio SMS
when a plate reaches the configured threshold within a session.

Design:
  • No external state — counts reset on process restart (stateless between sessions)
  • Alert fires exactly once per plate per session (no duplicate spam)
  • SMTP and SMS are independent — one can succeed while the other fails
  • All credentials loaded from Settings (which reads env vars)
  • Fully testable without credentials (threshold logic is pure Python)
"""

from __future__ import annotations
import logging
import smtplib
import threading
from collections import defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("AlertSystem")


class AlertSystem:
    """
    Tracks violations per plate and fires alerts at threshold.

    Parameters
    ----------
    threshold : int
        Number of confirmed violations before an alert is sent.
    settings  : Settings | None
        Runtime config containing SMTP/Twilio credentials.
        If None, alerts are logged but not actually sent (test mode).
    """

    def __init__(self, threshold: int = 3, settings=None):
        self.threshold = threshold
        self.cfg       = settings
        self._counts:   dict[str, int] = defaultdict(int)
        self._alerted:  set[str]       = set()
        self._lock      = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────

    def record(self, plate: str, violation: str) -> bool:
        """
        Record a violation event for plate.

        Returns True the first time the threshold is reached (alert fires).
        Returns False for all subsequent calls after the alert has fired.
        """
        key = f"{plate}:{violation}"
        with self._lock:
            if key in self._alerted:
                return False
            self._counts[key] += 1
            if self._counts[key] >= self.threshold:
                self._alerted.add(key)
                self._dispatch_alert(plate, violation, self._counts[key])
                return True
        return False

    def reset_plate(self, plate: str):
        """Clear all violation history for a plate (e.g. vehicle leaves scene)."""
        with self._lock:
            keys = [k for k in self._counts if k.startswith(f"{plate}:")]
            for k in keys:
                del self._counts[k]
                self._alerted.discard(k)

    def get_stats(self) -> dict:
        """Return current counts dict (copy) for API/reporting."""
        with self._lock:
            return {
                "tracked_plates":  len(set(k.split(":")[0] for k in self._counts)),
                "alerted_plates":  len(self._alerted),
                "violation_counts": dict(self._counts),
            }

    # ── Internal dispatch ───────────────────────────────────────────

    def _dispatch_alert(self, plate: str, violation: str, count: int):
        """Dispatch email and/or SMS in background threads."""
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"[ANPR ALERT] Repeat Violator — {plate}"
        body    = (
            f"⚠️  Repeat Violation Alert\n\n"
            f"  Plate     : {plate}\n"
            f"  Violation : {violation}\n"
            f"  Count     : {count} times this session\n"
            f"  Timestamp : {ts}\n\n"
            f"This plate has been flagged {count} times. "
            f"Please take appropriate action.\n\n"
            f"— Smart City ANPR System (SRM IST)"
        )

        log.warning(f"ALERT triggered: {plate} | {violation} × {count}")

        if self.cfg is None:
            return   # test mode — no actual sends

        # Email
        if self.cfg.alert_email_to and self.cfg.alert_smtp_user:
            t = threading.Thread(
                target=self._send_email,
                args=(subject, body),
                daemon=True,
            )
            t.start()

        # SMS (Twilio)
        if self.cfg.alert_twilio_sid and self.cfg.alert_sms_to:
            sms_body = (
                f"[ANPR ALERT] Repeat violator {plate}: "
                f"{violation} x{count} at {ts}"
            )
            t = threading.Thread(
                target=self._send_sms,
                args=(sms_body,),
                daemon=True,
            )
            t.start()

    def _send_email(self, subject: str, body: str):
        """Send SMTP email (TLS on port 587)."""
        cfg = self.cfg
        try:
            msg                       = MIMEMultipart()
            msg["From"]               = cfg.alert_email_from or cfg.alert_smtp_user
            msg["To"]                 = cfg.alert_email_to
            msg["Subject"]            = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(cfg.alert_smtp_host, cfg.alert_smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(cfg.alert_smtp_user, cfg.alert_smtp_pass)
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())

            log.info(f"  ✓ Email sent to {cfg.alert_email_to}")
        except Exception as e:
            log.error(f"Email send failed: {e}")

    def _send_sms(self, body: str):
        """Send SMS via Twilio REST API."""
        cfg = self.cfg
        try:
            from twilio.rest import Client
            client = Client(cfg.alert_twilio_sid, cfg.alert_twilio_token)
            message = client.messages.create(
                body = body,
                from_= cfg.alert_sms_from,
                to   = cfg.alert_sms_to,
            )
            log.info(f"  ✓ SMS sent: {message.sid}")
        except ImportError:
            log.warning("twilio library not installed — SMS skipped. "
                        "Install with: pip install twilio")
        except Exception as e:
            log.error(f"SMS send failed: {e}")
