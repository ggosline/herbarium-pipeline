"""Reusable presentational widgets and small helpers.

Pulled out of the monolithic webui module so each tab builder can import
exactly what it needs without dragging in pipeline-runner state. None of
these functions touch process state, network, or the filesystem beyond
what a file picker has to do.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import app, ui


# ── tiny helpers ──────────────────────────────────────────────────────────

def _v(widget) -> str:
    """Return widget value as a stripped string, handling None from clearable inputs."""
    return (widget.value or "").strip()


def _section(title: str) -> None:
    """Teal-bordered uppercase section header — visual rhythm marker."""
    (ui.label(title)
     .classes("w-full font-bold")
     .style("color:#00695c; border-left:3px solid #00897b; background:#f0f7f6;"
            "padding:3px 10px; margin:8px 0 3px; font-size:11px;"
            "letter-spacing:.7px; text-transform:uppercase;"
            "border-radius:0 3px 3px 0; display:block"))


def _accordion(title: str, *, opened: bool = True):
    """Collapsible section that styles its header like _section().

    Use as a context manager:

        with _accordion("Model & batch"):
            ui.label(...)

    Pass ``opened=False`` for a section collapsed by default — useful for
    advanced / rarely-touched options. Returns the ui.expansion so the
    caller can ``with`` it.
    """
    return (ui.expansion(title, value=opened)
            .classes("w-full")
            .props("dense expand-separator header-class=accordion-header")
            .style("margin:8px 0 4px;border-radius:0 3px 3px 0;"
                   "border-left:3px solid #00897b;background:#f0f7f6"))


# ── status pills (Setup tab) ──────────────────────────────────────────────

_PILL_PALETTE = {
    "ok":      ("#e8f5e9", "#2e7d32"),
    "warn":    ("#fff3e0", "#e65100"),
    "err":     ("#ffebee", "#c62828"),
    "neutral": ("#eceff1", "#455a64"),
}


def _pill(text: str, kind: str) -> "ui.label":
    """Status pill rendered as a coloured Quasar chip-style label.

    kind: 'ok' (green), 'warn' (amber), 'err' (red), 'neutral' (grey).
    Returns the label so callers can update its text/style later via _set_pill.
    """
    bg, fg = _PILL_PALETTE.get(kind, _PILL_PALETTE["neutral"])
    return (ui.label(text)
            .classes("rounded text-caption")
            .style(f"background:{bg};color:{fg};padding:2px 10px;"
                   f"font-weight:600;letter-spacing:.2px"))


def _set_pill(lbl: "ui.label", text: str, kind: str) -> None:
    bg, fg = _PILL_PALETTE.get(kind, _PILL_PALETTE["neutral"])
    lbl.set_text(text)
    lbl.style(f"background:{bg};color:{fg};padding:2px 10px;"
              f"font-weight:600;letter-spacing:.2px")


def _setup_card(icon: str, title: str, subtitle: str = ""):
    """Open a setup-section card returning (card, pill_label) so the caller
    can fill the body and update the pill when state changes."""
    card = ui.card().classes("w-full mt-2").style("border-left:3px solid #00897b")
    with card:
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon(icon).style("color:#00897b;font-size:22px")
            ui.label(title).classes("text-subtitle1 font-bold").style("color:#00695c")
            if subtitle:
                ui.label(subtitle).classes("text-caption text-grey-6")
            pill = _pill("…", "neutral")
            pill.classes("ml-auto")
    return card, pill


# ── file picker ───────────────────────────────────────────────────────────

class FilePicker(ui.dialog):
    """Navigable local-filesystem picker for files or directories."""

    def __init__(self, initial: str = "", mode: str = "dir"):
        super().__init__()
        self.mode = mode          # "dir" | "file" | "save"
        start = Path(initial or Path.home()).expanduser().resolve()
        self._cur = start.parent if start.is_file() else start
        # Walk up to the nearest directory that actually exists
        while not self._cur.is_dir() and self._cur != self._cur.parent:
            self._cur = self._cur.parent

        # For "save" mode, derive default filename from initial path
        default_name = Path(initial).name if initial and not Path(initial).is_dir() else ""

        with self, ui.card().style("min-width:560px"):
            with ui.row().classes("items-center w-full gap-1"):
                ui.button(icon="arrow_upward", on_click=self._up).props("flat round dense")
                self._loc = (ui.input("")
                             .classes("flex-1")
                             .props("dense outlined")
                             .on("keydown.enter", lambda e: self._goto(self._loc.value)))
            ui.separator()
            self._area = ui.scroll_area().style("height:380px; width:100%")
            ui.separator()
            if mode == "save":
                self._fname_inp = (ui.input(value=default_name, placeholder="filename.csv")
                                   .classes("w-full").props("dense outlined"))
            with ui.row().classes("w-full justify-end gap-2 pt-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                if mode == "dir":
                    ui.button("Select this folder",
                              on_click=lambda: self.submit(str(self._cur))
                              ).props("color=primary unelevated")
                elif mode == "save":
                    ui.button("Save here",
                              on_click=lambda: self.submit(
                                  str(self._cur / self._fname_inp.value.strip())
                              ) if self._fname_inp.value.strip() else ui.notify(
                                  "Enter a filename.", type="warning")
                              ).props("color=primary unelevated")

        self._render()

    def _goto(self, path: str) -> None:
        p = Path(path.strip()).expanduser().resolve()
        if p.is_dir():
            self._cur = p
            self._render()
        elif p.is_file() and self.mode in ("file", "save"):
            self.submit(str(p))

    def _render(self) -> None:
        self._loc.value = str(self._cur)
        entries: list[Path] = []
        try:
            entries = sorted(self._cur.iterdir(),
                             key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            pass

        self._area.clear()
        # Build the list inside the scroll area's own slot explicitly.
        # NiceGUI 3.x requires the target container to be entered via `with`
        # so element creation lands in the right slot after clear().
        with self._area:
            lst = ui.list().props("dense separator").classes("w-full")
        with lst:
            for e in entries:
                if e.is_dir():
                    with ui.item(on_click=lambda d=e: self._into(d)).props("clickable v-ripple"):
                        with ui.item_section().props("avatar"):
                            ui.icon("folder").classes("text-amber-6")
                        with ui.item_section():
                            ui.item_label(e.name)
                elif e.is_file() and self.mode in ("file", "save"):
                    if self.mode == "save":
                        action = lambda f=e: setattr(self._fname_inp, "value", f.name)
                    else:
                        action = lambda f=e: self.submit(str(f))
                    with ui.item(on_click=action).props("clickable v-ripple"):
                        with ui.item_section().props("avatar"):
                            ui.icon("description").classes("text-blue-grey-4")
                        with ui.item_section():
                            ui.item_label(e.name)

    def _up(self) -> None:
        self._cur = self._cur.parent
        self._render()

    def _into(self, d: Path) -> None:
        self._cur = d
        self._render()


# ── label + input + browse row ────────────────────────────────────────────

def _path_input(label: str, value: str = "", mode: str = "dir",
                hint: str = "") -> ui.input:
    """Label + text input + browse button row. Returns the input."""
    with ui.row().classes("w-full items-center gap-2"):
        ui.label(label).classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
        inp = (ui.input(value=value, placeholder=hint or "")
               .classes("flex-1").props("dense outlined clearable"))

        async def _browse():
            cur = _v(inp) or str(Path.home())
            result = await FilePicker(cur, mode=mode)
            if result:
                inp.value = result

        ui.button(icon="folder_open", on_click=_browse
                  ).props("flat dense round").tooltip("Browse")
    return inp


def _text_row(label: str, value: str = "", width: str = "w-36") -> ui.input:
    with ui.row().classes("w-full items-center gap-2"):
        ui.label(label).classes("w-36 text-right shrink-0 font-medium").style("color:#455a64")
        inp = ui.input(value=value).classes(width).props("dense outlined")
    return inp


def _inline(*items):
    """Render several (label, widget_factory) pairs in one row."""
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
        for label, factory in items:
            with ui.row().classes("items-center gap-1"):
                ui.label(label).classes("text-sm")
                factory()


# ── editable list of  specsin.csv : images_dir  pairs ─────────────────────

class SourcesPanel:
    """Editable list of  specsin.csv : images_dir  pairs."""

    def __init__(self, config_key: str = "") -> None:
        self._config_key = config_key
        self._sources: list[str] = (
            app.storage.general.get(config_key, []) if config_key else []
        )
        self._container = ui.column().classes("w-full gap-1")
        ui.button("Add Source…", icon="add", on_click=self._add).props("flat dense")
        self._refresh()

    def _persist(self) -> None:
        if self._config_key:
            app.storage.general[self._config_key] = self._sources

    async def _add(self) -> None:
        with ui.dialog() as dlg, ui.card().classes("w-full").style("min-width:480px"):
            ui.label("Add data source").classes("text-subtitle1 font-bold")
            sv = _path_input("specsin CSV:", mode="file")
            iv = _path_input("Images dir:", mode="dir")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                def _ok():
                    s, i = _v(sv), _v(iv)
                    if s and i:
                        self._sources.append(f"{s}:{i}")
                        self._refresh()
                    dlg.close()
                ui.button("Add", on_click=_ok).props("color=primary unelevated")
        await dlg

    def _refresh(self) -> None:
        self._persist()
        self._container.clear()
        with self._container:
            for idx, src in enumerate(self._sources):
                with ui.row().classes("w-full items-center gap-1"):
                    ui.label(src).classes(
                        "text-caption font-mono flex-1 bg-grey-2 px-2 py-1 rounded")
                    ui.button(icon="close",
                              on_click=lambda i=idx: self._remove(i)
                              ).props("flat dense round").tooltip("Remove")

    def _remove(self, idx: int) -> None:
        if 0 <= idx < len(self._sources):
            self._sources.pop(idx)
            self._refresh()

    def get_sources(self) -> list[str]:
        return list(self._sources)

    def set_source(self, pair: str) -> None:
        self._sources = [pair]
        self._refresh()
