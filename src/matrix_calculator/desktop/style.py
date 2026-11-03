"""桌面端 Apple 风格“液态玻璃”样式表。"""

STYLESHEET = """
* {
    font-family: "SF Pro Display", "SF Pro Text", "Segoe UI Variable Display",
        "Segoe UI", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 14px;
    color: #1d1d1f;
}
#Root {
    background: qlineargradient(x1:0, y1:0, x2:0.55, y2:1,
        stop:0 #fafbff, stop:1 #edf1f9);
}
#Sidebar {
    background: rgba(255, 255, 255, 0.52);
    border: 1px solid rgba(255, 255, 255, 0.75);
    border-radius: 22px;
}
#Card {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(255, 255, 255, 0.85);
    border-radius: 18px;
}
#Segment {
    background: rgba(120, 120, 128, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.55);
    border-radius: 12px;
}
#SegmentButton {
    background: transparent;
    border: none;
    border-radius: 9px;
    padding: 6px 16px;
}
#SegmentButton:hover { background: rgba(255, 255, 255, 0.45); }
#SegmentButton:checked {
    background: #ffffff;
    color: #0071e3;
    font-weight: 600;
}
#NavButton {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 10px 14px;
    text-align: left;
}
#NavButton:hover { background: rgba(0, 0, 0, 0.045); }
#NavButton:checked {
    background: #0071e3;
    color: #ffffff;
    font-weight: 600;
}
#Accent {
    background: #0071e3;
    color: #ffffff;
    border: none;
    border-radius: 20px;
    padding: 8px 26px;
    font-size: 15px;
    font-weight: 600;
}
#Accent:hover { background: #0a7cff; }
#Accent:pressed { background: #0060c2; }
#Ghost {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(29, 29, 31, 0.08);
    border-radius: 11px;
    padding: 5px 14px;
}
#Ghost:hover { background: rgba(255, 255, 255, 0.92); }
#Ghost:pressed { background: rgba(255, 255, 255, 1.0); }
#Ghost:disabled { color: rgba(29, 29, 31, 0.28); }
QLineEdit#Cell {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(29, 29, 31, 0.08);
    border-radius: 9px;
    padding: 4px 6px;
    selection-background-color: rgba(0, 113, 227, 0.22);
}
QLineEdit#Cell:focus {
    border: 1px solid #0071e3;
    background: #ffffff;
}
#BracketL {
    border-left: 2px solid rgba(29, 29, 31, 0.42);
    border-top: 2px solid rgba(29, 29, 31, 0.42);
    border-bottom: 2px solid rgba(29, 29, 31, 0.42);
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
    background: transparent;
}
#BracketR {
    border-right: 2px solid rgba(29, 29, 31, 0.42);
    border-top: 2px solid rgba(29, 29, 31, 0.42);
    border-bottom: 2px solid rgba(29, 29, 31, 0.42);
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
    background: transparent;
}
QGroupBox {
    background: rgba(255, 255, 255, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 16px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px;
    color: #3c3c43;
    font-size: 13px;
}
/* iOS 拨动开关风格的折叠指示器 */
QGroupBox::indicator {
    width: 32px;
    height: 19px;
    border-radius: 10px;
    background: rgba(120, 120, 128, 0.26);
    border: none;
}
QGroupBox::indicator:hover { background: rgba(120, 120, 128, 0.38); }
QGroupBox::indicator:checked { background: #34c759; }
QLabel#Title { font-size: 24px; font-weight: 700; }
QLabel#H2 { font-size: 15px; font-weight: 600; }
QLabel#Caption { color: #6e6e73; font-size: 12px; }
QLabel#Error { color: #ff3b30; }
QMenu {
    background: rgba(250, 250, 252, 0.98);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    padding: 5px;
}
QMenu::item { padding: 6px 20px; border-radius: 8px; }
QMenu::item:selected { background: rgba(0, 113, 227, 0.12); color: #0071e3; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px; }
QScrollBar::handle:vertical {
    background: rgba(29, 29, 31, 0.16);
    border-radius: 3px;
    min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: rgba(29, 29, 31, 0.28); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""
