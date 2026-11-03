"""桌面端 Apple 风格“液态玻璃”样式表。"""

STYLESHEET = """
* {
    font-family: "PingFang SC", "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 14px;
    color: #1d1d1f;
}
#Root {
    background: qlineargradient(x1:0, y1:0, x2:0.55, y2:1,
        stop:0 #f8faff, stop:1 #eef3fb);
}
#Sidebar {
    background: rgba(255, 255, 255, 0.56);
    border: 1px solid rgba(255, 255, 255, 0.78);
    border-radius: 20px;
}
#Card {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(255, 255, 255, 0.82);
    border-radius: 16px;
}
#NavButton {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 10px 14px;
    text-align: left;
}
#NavButton:hover { background: rgba(255, 255, 255, 0.55); }
#NavButton:checked {
    background: rgba(255, 255, 255, 0.88);
    color: #0071e3;
    font-weight: 600;
}
#Pill {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.88);
    border-radius: 17px;
    padding: 6px 16px;
}
#Pill:hover { background: rgba(255, 255, 255, 0.85); }
#Pill:checked {
    background: #ffffff;
    color: #0071e3;
    border: 1px solid rgba(0, 113, 227, 0.45);
    font-weight: 600;
}
#Accent {
    background: #0071e3;
    color: #ffffff;
    border: none;
    border-radius: 18px;
    padding: 8px 22px;
    font-weight: 600;
}
#Accent:hover { background: #0077ed; }
#Ghost {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.92);
    border-radius: 15px;
    padding: 5px 14px;
}
#Ghost:hover { background: rgba(255, 255, 255, 0.88); }
#Ghost:disabled { color: rgba(29, 29, 31, 0.3); }
QTableWidget {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.92);
    border-radius: 12px;
    gridline-color: rgba(29, 29, 31, 0.08);
    selection-background-color: rgba(0, 113, 227, 0.16);
}
QHeaderView::section {
    background: rgba(255, 255, 255, 0.72);
    border: none;
    border-bottom: 1px solid rgba(29, 29, 31, 0.08);
    padding: 4px;
}
QGroupBox {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.82);
    border-radius: 14px;
    margin-top: 10px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #6e6e73;
}
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#H2 { font-size: 15px; font-weight: 650; }
QLabel#Caption { color: #6e6e73; font-size: 12px; }
QLabel#Error { color: #d13438; }
QMenu {
    background: rgba(250, 250, 252, 0.98);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 10px;
    padding: 4px;
}
QMenu::item { padding: 6px 18px; border-radius: 6px; }
QMenu::item:selected { background: rgba(0, 113, 227, 0.12); color: #0071e3; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: rgba(29, 29, 31, 0.18); border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
