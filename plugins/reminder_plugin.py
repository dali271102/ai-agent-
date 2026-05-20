"""
Reminder Plugin — call-like Telegram notification with snooze/dismiss.
"""
from typing import Optional
import threading, time, re, json
from datetime import datetime, timedelta
from pathlib import Path
from plugins import Plugin

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
except ImportError:
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None

REMINDERS_FILE = Path("memory/reminders.json")


class ReminderPlugin(Plugin):
    name        = "reminder"
    description = "Set reminders with call-like Telegram notifications + snooze"
    version     = "2.0"
    author      = "upgraded"

    def __init__(self, agent_ref):
        super().__init__(agent_ref)
        self.pending  = {}
        self._counter = 0
        self._lock    = threading.Lock()
        self._load_saved()

    def _load_saved(self):
        if not REMINDERS_FILE.exists():
            return
        try:
            saved = json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
            now   = datetime.now().timestamp()
            for r in saved:
                delay = r["fire_at"] - now
                if delay > 0:
                    rid = r.get("id", self._next_id())
                    with self._lock:
                        self.pending[rid] = {"task": r["task"], "fire_at": r["fire_at"]}
                    self._schedule(r["task"], delay, rid)
        except Exception:
            pass

    def _save(self):
        REMINDERS_FILE.parent.mkdir(exist_ok=True)
        data = [
            {"id": rid, "task": info["task"], "fire_at": info["fire_at"]}
            for rid, info in self.pending.items()
        ]
        try:
            REMINDERS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def get_commands(self):
        return {
            "remind":          "remind me in [N] seconds/minutes/hours to [task]",
            "list reminders":  "show all pending reminders",
            "cancel reminder": "cancel reminder [id]",
        }

    def handle(self, command: str) -> Optional[str]:
        cmd = command.lower().strip()

        if "list reminders" in cmd or "show reminders" in cmd or "my reminders" in cmd:
            return self._list_reminders()

        if "cancel reminder" in cmd or "delete reminder" in cmd:
            return self._cancel_reminder(cmd)

        if not (cmd.startswith("remind") or "remind me" in cmd):
            return None

        m = re.search(r"in\s+(\d+)\s*(second|minute|hour|sec|min|hr)s?", cmd)
        task_match = re.search(r"\bto\s+(.+)$", cmd)

        if not m:
            return "❓ Format: remind me in 10 seconds to drink water"

        amount = int(m.group(1))
        unit   = m.group(2)
        task   = task_match.group(1).strip() if task_match else "something"

        mult  = {"second":1,"sec":1,"minute":60,"min":60,"hour":3600,"hr":3600}
        delay = amount * mult.get(unit, 60)

        fire_at = datetime.now() + timedelta(seconds=delay)
        rid     = self._next_id()

        with self._lock:
            self.pending[rid] = {"task": task, "fire_at": fire_at.timestamp()}
        self._save()
        self._schedule(task, delay, rid)

        return (
            f"⏰ Reminder #{rid} set!\n"
            f"   Task: {task}\n"
            f"   Fires at: {fire_at.strftime('%H:%M:%S')}"
        )

    def _next_id(self):
        self._counter += 1
        return self._counter

    def _schedule(self, task, delay, rid):
        def _fire():
            time.sleep(delay)
            self._fire_reminder(task, rid)
        threading.Thread(target=_fire, daemon=True).start()

    def _fire_reminder(self, task, rid):
        with self._lock:
            self.pending.pop(rid, None)
        self._save()

        agent = self.agent

        # XP chat popup
        if hasattr(agent, "root") and hasattr(agent, "_system_msg"):
            try:
                agent.root.after(0, lambda: agent._system_msg(
                    f"📞 ══════════════════════\n"
                    f"   REMINDER #{rid}\n"
                    f"   🔔 {task}\n"
                    f"   ══════════════════════", "error"
                ))
            except Exception:
                pass

        # Telegram call-like notification
        tg = getattr(agent, "tg_bot", None)
        if not (tg and getattr(tg, "running", False) and getattr(tg, "_loop", None)):
            return

        chat_id = getattr(tg, "_last_chat_id", None)
        if not chat_id:
            return

        import asyncio

        if InlineKeyboardMarkup is None:
            try:
                asyncio.run_coroutine_threadsafe(
                    tg.app.bot.send_message(chat_id, f"⏰ REMINDER: {task}"),
                    tg._loop
                )
            except Exception:
                pass
            return

        text = (
            "📞 *INCOMING REMINDER*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🔔  *{task}*\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Dismiss",      callback_data=f"rem_dismiss:{rid}"),
            InlineKeyboardButton("⏰ Snooze 5min", callback_data=f"rem_snooze:{rid}:{task}"),
        ]])

        async def _send():
            await tg.app.bot.send_message(
                chat_id, text,
                parse_mode="Markdown",
                reply_markup=keyboard
            )
        try:
            asyncio.run_coroutine_threadsafe(_send(), tg._loop)
        except Exception:
            pass

    def _list_reminders(self):
        with self._lock:
            items = list(self.pending.items())
        if not items:
            return "📋 No pending reminders."
        lines = [f"📋 Pending reminders ({len(items)}):"]
        for rid, info in items:
            fire = datetime.fromtimestamp(info["fire_at"]).strftime("%H:%M:%S")
            lines.append(f"  #{rid} — {info['task']}  (at {fire})")
        lines.append("\nCancel: cancel reminder [id]")
        return "\n".join(lines)

    def _cancel_reminder(self, cmd):
        m = re.search(r"(\d+)", cmd)
        if not m:
            return "❓ Format: cancel reminder 1"
        rid = int(m.group(1))
        with self._lock:
            if rid in self.pending:
                del self.pending[rid]
                self._save()
                return f"✅ Reminder #{rid} cancelled."
        return f"❌ Reminder #{rid} not found."
