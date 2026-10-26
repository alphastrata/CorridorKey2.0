"""Stats for nerds: the telemetry as a translucent panel over the viewer. Clicks pass through."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

MARGIN = 12


class StatsOverlay(QFrame):
    def __init__(self, host: QWidget):
        super().__init__(host)
        self.setObjectName("statsOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet("#statsOverlay { background: rgba(8, 10, 12, 205); border: 1px solid "
                           "rgba(255, 255, 255, 40); border-radius: 6px; }"
                           "#statsOverlay QLabel { color: #d8f5d0; background: transparent; }")
        self.text = QLabel()
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPixelSize(11)
        self.text.setFont(mono)
        self.text.setTextFormat(Qt.TextFormat.PlainText)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addWidget(self.text)
        host.installEventFilter(self)
        self.hide()

    def set_text(self, text: str) -> None:
        self.text.setText(text)
        self._place()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self._place()
        return False

    def _place(self) -> None:
        host = self.parentWidget()
        self.setMaximumWidth(max(host.width() - 2 * MARGIN, 200))
        self.adjustSize()
        self.move(MARGIN, MARGIN)
        self.raise_()
