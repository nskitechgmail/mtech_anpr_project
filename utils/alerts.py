"""
utils/alerts.py — Email and SMS alert system for repeat violators.

Sends an alert when the same plate accumulates >= THRESHOLD violations
within a rolling time window.  Uses SMTP for email (supports Gmail/Outlook)
and Twilio REST API for SMS.  All credentials are read from environment
variables or the Settings object — never hardcoded.

Environment variables (optional):
    ANPR_ALERT_EMAIL_FROM    sender email address
    ANPR_ALERT_EMAIL_TO      comma-separated recipient emails
    ANPR_ALERT_SMTP_HOST     default: smtp.gmail.com
    ANPR_ALERT_SMTP_PORT     default: 587
    ANPR_ALERT_SMTP_USER     SMTP auth username
    ANPR_ALERT_SMTP_PASS     SMTP auth password

    ANPR_ALERT_SMS_TO        Twilio: recipient phone (+91...)
    ANPR_ALERT_SMS_FROM      Twilio: sender phone
    ANPR_TWILIO_SID          Twilio account SID
    ANPR_TWILIO_TOKEN        Twilio auth token
"""
from __future__ import annotations
import os, logging, smtplib, threading
from email.mime.text    import MIMEText
from email.mime.multipart import MIMEMultipart
from collections        import defaultdict, deque
from datetime           import datetime, timedelta

log = logging.getLogger("AlertSystem")

# Repeat-violator threshold: trigger alert after this many violations
_REPEAT_THRESHOLD = 3
# Rolling window in minutes
_WINDOW_MINUTES   = 60


class AlertSystem:
    """
    Tracks violations per plate and fires email / SMS alerts for repeat offenders.

    Usage::
        alerts = AlertSystem()
        alerts.record(plate="MH12AB1234", violation="No Helmet", camera="CCTV-001")
    """

    def __init__(self, threshold: int = _REPEAT_THRESHOLD,
                 window_minutes: int = _WINDOW_MINUTES):
        self.threshold = threshold
        self.window    = timedelta(minutes=window_minutes)
        # plate → deque of (datetime, violation_str)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._alerted: set[str] = set()
        self._lock = threading.Lock()

    def record(self, plate: str, violation: str, camera: str = "") -> bool:
        """
        Record a violation event. Returns True if an alert was triggered.
        Thread-safe.
        """
        plate = plate.upper().strip()
        now   = datetime.now()
        with self._lock:
            self._history[plate].append((now, violation))
            # Count events inside the rolling window
            recent = [
                (t, v) for t, v in self._history[plate]
                if now - t <= self.window
            ]
            if len(recent) >= self.threshold and plate not in self._alerted:
                self._alerted.add(plate)
                log.warning(
                    f"REPEAT VIOLATOR: {plate} — {len(recent)} violations "
                    f"in {self.window.seconds//60} min. Sending alert."
                )
                # Fire alert in background thread
                threading.Thread(
                    target=self._send_alert,
                    args=(plate, recent, camera),
                    daemon=True,
                ).start()
                return True
        return False

    def reset_plate(self, plate: str) -> None:
        """Clear violation history for a plate (e.g., after manual review)."""
        with self._lock:
            self._history.pop(plate, None)
            self._alerted.discard(plate)

    # ── Alert dispatch ─────────────────────────────────────────────────────

    def _send_alert(self, plate: str, events: list, camera: str) -> None:
        subject = f"⚠ Repeat Traffic Violator Detected — {plate}"
        body    = self._build_message_body(plate, events, camera)
        self._send_email(subject, body)
        self._send_sms(f"ANPR ALERT: Repeat violator {plate} — {len(events)} violations "
                       f"at {camera}. Check dashboard.")

    @staticmethod
    def _build_message_body(plate: str, events: list, camera: str) -> str:
        lines = [
            f"Smart City ANPR System — Repeat Violator Alert",
            f"{'='*48}",
            f"Plate         : {plate}",
            f"Camera        : {camera}",
            f"Generated at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Violation History (last {len(events)} events):",
        ]
        for i, (ts, viol) in enumerate(events, 1):
            lines.append(f"  {i:2d}. [{ts.strftime('%H:%M:%S')}]  {viol}")
        lines += ["", "Please review CCTV footage and take appropriate action.",
                  "— SRM Smart City ANPR System"]
        return "\n".join(lines)

    def _send_email(self, subject: str, body: str) -> None:
        """Send alert email via SMTP. No-op if credentials not configured."""
        smtp_host = os.getenv("ANPR_ALERT_SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("ANPR_ALERT_SMTP_PORT", "587"))
        smtp_user = os.getenv("ANPR_ALERT_SMTP_USER", "")
        smtp_pass = os.getenv("ANPR_ALERT_SMTP_PASS", "")
        from_addr = os.getenv("ANPR_ALERT_EMAIL_FROM", smtp_user)
        to_addrs  = [a.strip() for a in
                     os.getenv("ANPR_ALERT_EMAIL_TO", "").split(",") if a.strip()]

        if not smtp_user or not to_addrs:
            log.debug("Email alert skipped — SMTP credentials not configured.")
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = from_addr
            msg["To"]      = ", ".join(to_addrs)
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            log.info(f"Alert email sent to: {to_addrs}")
        except Exception as e:
            log.error(f"Email alert failed: {e}")

    def _send_sms(self, message: str) -> None:
        """Send SMS via Twilio. No-op if credentials not configured."""
        sid   = os.getenv("ANPR_TWILIO_SID", "")
        token = os.getenv("ANPR_TWILIO_TOKEN", "")
        to    = os.getenv("ANPR_ALERT_SMS_TO", "")
        from_ = os.getenv("ANPR_ALERT_SMS_FROM", "")

        if not all([sid, token, to, from_]):
            log.debug("SMS alert skipped — Twilio credentials not configured.")
            return

        try:
            from twilio.rest import Client
            client = Client(sid, token)
            client.messages.create(body=message, from_=from_, to=to)
            log.info(f"SMS alert sent to {to}")
        except ImportError:
            log.debug("twilio package not installed — SMS skipped.")
        except Exception as e:
            log.error(f"SMS alert failed: {e}")
