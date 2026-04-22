"""
Herbarium Classification Pipeline — GUI frontend.

Wraps the five pipeline steps in a tkinter interface:
  1. Download      — download_gbif_images.py
  2. Filter & Crop — filter_and_crop_herbarium.py
  3. Resize        — resize_images.py
  4. Train         — train_herbarium.py
  5. Identify      — identify_herbarium.py

Run:
  python herbarium_pipeline_gui.py
"""

import json
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from tkinter import font as tkfont

# Locate sibling scripts relative to this file
_HERE = Path(__file__).parent
SCRIPTS = {
    "download":        _HERE / "download_gbif_images.py",
    "filter_and_crop": _HERE / "filter_and_crop_herbarium.py",
    "resize":          _HERE / "resize_images.py",
    "train":           _HERE / "train_herbarium.py",
    "identify":        _HERE / "identify_herbarium.py",
}

CONTINENTS = ["", "AFRICA", "ASIA", "EUROPE", "NORTH_AMERICA",
              "SOUTH_AMERICA", "OCEANIA", "ANTARCTICA"]

TIMM_MODELS = [
    "vit_large_patch16_dinov3.lvd1689m",
    "convnext_base_384_in22ft1k",
    "convnext_large_384_in22ft1k",
    "efficientnet_b4",
    "resnet50",
]


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

class ProcessRunner:
    """Runs a subprocess and streams its stdout to a tkinter Text widget."""

    def __init__(self, log_widget: scrolledtext.ScrolledText,
                 status_var: tk.StringVar, on_finish=None):
        self.log     = log_widget
        self.status  = status_var
        self.on_finish = on_finish
        self._q: queue.Queue = queue.Queue()
        self._proc = None

    def run(self, cmd: list[str]):
        self.log.insert(tk.END, f"\n$ {shlex.join(cmd)}\n", "cmd")
        self.log.see(tk.END)
        self.status.set("Running…")

        def worker():
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in self._proc.stdout:
                    self._q.put(line)
                self._proc.wait()
                self._q.put(None)  # sentinel
                returncode = self._proc.returncode
            except Exception as exc:
                self._q.put(f"ERROR launching process: {exc}\n")
                self._q.put(None)
                returncode = -1
            self._q.put(("__done__", returncode))

        threading.Thread(target=worker, daemon=True).start()
        self._poll()

    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                if item is None:
                    break
                if isinstance(item, tuple) and item[0] == "__done__":
                    rc = item[1]
                    msg = "Finished OK" if rc == 0 else f"Exited with code {rc}"
                    self.log.insert(tk.END, f"\n[{msg}]\n", "done" if rc == 0 else "err")
                    self.log.see(tk.END)
                    self.status.set(msg)
                    if self.on_finish:
                        self.on_finish(rc)
                    return
                self.log.insert(tk.END, item)
                self.log.see(tk.END)
        except queue.Empty:
            pass
        self.log.after(80, self._poll)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def terminate(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

def labeled_entry(parent, label: str, default: str = "", width: int = 45) -> tk.StringVar:
    row = ttk.Frame(parent)
    row.pack(fill="x", padx=8, pady=2)
    ttk.Label(row, text=label, width=22, anchor="e").pack(side="left")
    var = tk.StringVar(value=default)
    ttk.Entry(row, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
    return var


def browse_entry(parent, label: str, default: str = "",
                 mode: str = "dir") -> tk.StringVar:
    """Entry + Browse button for file or directory.

    The dialog opens at whichever of these exists first:
      1. Parent of the current entry value (for file/save modes) or the
         entry value itself (for dir mode).
      2. The most-recently-used browse directory (_last_dir).
    After a selection, _last_dir is updated for future dialogs.
    """
    row = ttk.Frame(parent)
    row.pack(fill="x", padx=8, pady=2)
    ttk.Label(row, text=label, width=22, anchor="e").pack(side="left")
    var = tk.StringVar(value=default)
    ttk.Entry(row, textvariable=var, width=38).pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _initial_dir() -> str:
        cur = var.get().strip()
        if cur:
            p = Path(cur)
            candidate = p.parent if mode in ("file", "save") else p
            if candidate.exists():
                return str(candidate)
        return str(_last_dir)

    def browse():
        global _last_dir
        idir = _initial_dir()
        if mode == "dir":
            path = filedialog.askdirectory(initialdir=idir)
        elif mode == "file":
            path = filedialog.askopenfilename(initialdir=idir)
        else:
            path = filedialog.asksaveasfilename(initialdir=idir)
        if path:
            var.set(path)
            _last_dir = Path(path).parent if mode in ("file", "save") else Path(path)

    ttk.Button(row, text="Browse…", command=browse, width=8).pack(side="left")
    return var


def section_header(parent, text: str):
    ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(10, 2))
    ttk.Label(parent, text=text, font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=8)


# ---------------------------------------------------------------------------
# Source list widget  (for train / identify)
# ---------------------------------------------------------------------------

class SourceList(ttk.Frame):
    """A listbox of 'specsin.csv : images_dir' pairs with Add/Remove buttons."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        row = ttk.Frame(self)
        row.pack(fill="both", expand=True)

        self._lb = tk.Listbox(row, height=5, selectmode="single")
        self._lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(row, orient="vertical", command=self._lb.yview)
        sb.pack(side="left", fill="y")
        self._lb.config(yscrollcommand=sb.set)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=2)
        ttk.Button(btns, text="Add Source…", command=self._add).pack(side="left", padx=2)
        ttk.Button(btns, text="Remove",      command=self._remove).pack(side="left", padx=2)

    def _add(self):
        dlg = tk.Toplevel(self)
        dlg.title("Add Source")
        dlg.grab_set()
        sv = browse_entry(dlg, "specsin CSV:", mode="file")
        iv = browse_entry(dlg, "Images dir:", mode="dir")

        def ok():
            s, i = sv.get().strip(), iv.get().strip()
            if s and i:
                self._lb.insert(tk.END, f"{s}:{i}")
            dlg.destroy()

        ttk.Button(dlg, text="Add", command=ok).pack(pady=6)
        dlg.wait_window()

    def _remove(self):
        sel = self._lb.curselection()
        if sel:
            self._lb.delete(sel[0])

    def get_sources(self) -> list[str]:
        return list(self._lb.get(0, tk.END))

    def set_source(self, source_pair: str):
        """Replace all entries with a single source pair (from project auto-fill)."""
        self._lb.delete(0, tk.END)
        self._lb.insert(tk.END, source_pair)

    def clear(self):
        self._lb.delete(0, tk.END)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_DEFAULT_BASE = Path.home()
_last_dir: Path = _DEFAULT_BASE  # updated after every browse dialog

# ---------------------------------------------------------------------------
# Persistent config  (~/.config/herbarium_pipeline.json)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".config" / "herbarium_pipeline.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Herbarium Classification Pipeline")
        self.geometry("1100x900")
        self.resizable(True, True)

        # Request antialiased font rendering via Xft
        try:
            self.tk.call("tk", "scaling", 1.5)
            self.option_add("*font", "sans-serif 11")
        except Exception:
            pass

        # Use clam theme — much cleaner than the default Motif-style theme
        style = ttk.Style()
        style.theme_use("clam")

        # Try to use a modern font; if the system Tk only sees X11 bitmap fonts
        # (run: conda install -c conda-forge tk  to fix permanently), fall back
        # gracefully to whatever TkDefaultFont resolves to.
        available = set(f.lower() for f in tkfont.families())
        ui_font = next(
            (f for f in ("Segoe UI", "Ubuntu", "Noto Sans", "DejaVu Sans",
                         "Liberation Sans", "Cantarell", "Arial",
                         "Bitstream Charter")
             if f.lower() in available),
            None,
        )
        mono_font = next(
            (f for f in ("JetBrains Mono", "Cascadia Code", "Ubuntu Mono",
                         "DejaVu Sans Mono", "Liberation Mono", "Noto Mono",
                         "Courier New", "Courier 10 Pitch")
             if f.lower() in available),
            None,
        )

        for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                      "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont"):
            try:
                kw = {"size": 13}
                if ui_font:
                    kw["family"] = ui_font
                tkfont.nametofont(fname).configure(**kw)
            except Exception:
                pass
        try:
            kw = {"size": 12}
            if mono_font:
                kw["family"] = mono_font
            tkfont.nametofont("TkFixedFont").configure(**kw)
        except Exception:
            pass

        # Apply to ttk widgets
        style.configure(".", font=("TkDefaultFont", 13))

        self._runner: ProcessRunner | None = None
        self._pipeline_steps: list | None = None
        self._project_name = tk.StringVar()

        cfg = _load_config()
        self._base_dir_var = tk.StringVar(value=cfg.get("base_dir", str(_DEFAULT_BASE)))

        self._build_ui()
        self._apply_tags()

        # Seed the browse dialog default to the saved base dir
        global _last_dir
        _last_dir = Path(self._base_dir_var.get())

        # Fire path updates whenever the project name changes
        self._project_name.trace_add("write", lambda *_: self._on_project_changed())

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Label(top, text="Herbarium Classification Pipeline",
                  font=("TkDefaultFont", 17, "bold")).pack(side="left")
        self._stop_btn = ttk.Button(top, text="Stop", command=self._stop,
                                    state="disabled")
        self._stop_btn.pack(side="right", padx=4)

        # Projects root + project name — drives all paths
        proj_frame = ttk.Frame(self, relief="groove", borderwidth=1)
        proj_frame.pack(fill="x", padx=8, pady=(0, 4))

        base_row = ttk.Frame(proj_frame)
        base_row.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(base_row, text="Projects root:", font=("TkDefaultFont", 12, "bold"),
                  width=14, anchor="e").pack(side="left", padx=(0, 4))
        ttk.Entry(base_row, textvariable=self._base_dir_var, width=40,
                  font=("TkDefaultFont", 12)).pack(side="left", fill="x", expand=True, padx=(0, 4))

        def _browse_base():
            global _last_dir
            p = filedialog.askdirectory(initialdir=self._base_dir_var.get() or str(_last_dir))
            if p:
                self._base_dir_var.set(p)
                _last_dir = Path(p)
        ttk.Button(base_row, text="Browse…", command=_browse_base, width=8).pack(side="left")

        name_row = ttk.Frame(proj_frame)
        name_row.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Label(name_row, text="Project name:", font=("TkDefaultFont", 12, "bold"),
                  width=14, anchor="e").pack(side="left", padx=(0, 4))
        ttk.Entry(name_row, textvariable=self._project_name, width=30,
                  font=("TkDefaultFont", 12)).pack(side="left", pady=4)
        ttk.Label(name_row, text="  →  ", foreground="grey").pack(side="left")
        self._proj_path_lbl = ttk.Label(name_row, text="", foreground="#0055cc",
                                        font=("TkDefaultFont", 12, "bold"))
        self._proj_path_lbl.pack(side="left")
        ttk.Button(name_row, text="Apply paths", width=10,
                   command=self._on_project_changed).pack(side="left", padx=8, pady=4)

        # Notebook (tabs per step)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=4)

        self._tab_download     = ttk.Frame(nb)
        self._tab_filter_crop  = ttk.Frame(nb)
        self._tab_resize       = ttk.Frame(nb)
        self._tab_train        = ttk.Frame(nb)
        self._tab_identify     = ttk.Frame(nb)
        self._tab_all          = ttk.Frame(nb)

        nb.add(self._tab_download,    text="1  Download")
        nb.add(self._tab_filter_crop, text="2  Filter & Crop")
        nb.add(self._tab_resize,      text="3  Resize")
        nb.add(self._tab_train,       text="4  Train")
        nb.add(self._tab_identify,    text="5  Identify")
        nb.add(self._tab_all,         text="Run All")

        self._build_download(self._tab_download)
        self._build_filter_crop(self._tab_filter_crop)
        self._build_resize(self._tab_resize)
        self._build_train(self._tab_train)
        self._build_identify(self._tab_identify)
        self._build_run_all(self._tab_all)

        # Log area
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._log = scrolledtext.ScrolledText(log_frame, height=12, font=("TkFixedFont", 11),
                                               wrap="word", state="normal")
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

        log_btns = ttk.Frame(log_frame)
        log_btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(log_btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=4)
        ttk.Button(log_btns, text="Save Log…", command=self._save_log).pack(side="left", padx=4)

        # Status bar
        self._status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self._status, relief="sunken",
                  anchor="w").pack(fill="x", padx=8, pady=(0, 4))

    def _apply_tags(self):
        self._log.tag_config("cmd",  foreground="#00aa00", font=("TkFixedFont", 11, "bold"))
        self._log.tag_config("done", foreground="#0055cc")
        self._log.tag_config("err",  foreground="#cc0000")

    # -------------------------------------------------- Project path wiring --

    def _on_project_changed(self):
        global _last_dir
        name    = self._project_name.get().strip()
        base    = self._base_dir_var.get().strip()
        if not base:
            messagebox.showerror("Missing", "Enter a Projects root directory first.")
            return
        if not name:
            self._proj_path_lbl.config(text="")
            return

        proj = Path(base) / name
        self._proj_path_lbl.config(text=str(proj))
        _last_dir = proj  # seed browse dialogs to the project folder

        # Persist the base dir choice
        cfg = _load_config()
        cfg["base_dir"] = base
        _save_config(cfg)

        images_dir  = str(proj / "images")
        specsin_csv = str(proj / "specsin.csv")
        runs_dir    = str(proj / "runs")
        review_dir  = str(proj / "review")
        ckpt_path   = str(proj / "runs" / "checkpoints" / "last.ckpt")
        nameslist   = str(proj / "runs" / "nameslist.json")
        source_pair = f"{specsin_csv}:{images_dir}"

        # Download
        self._dl_output_dir.set(images_dir)
        self._dl_specsin.set(specsin_csv)

        # Filter & Crop  (input = downloaded images, output = same dir by default)
        self._fc_input.set(images_dir)
        self._fc_output.set(images_dir)
        self._fc_specsin.set(specsin_csv)

        # Resize  (in-place by default — output left blank)
        self._rs_input.set(images_dir)
        self._rs_output.set("")

        # Train
        self._tr_output.set(runs_dir)
        self._tr_wandb_name.set(name)
        self._tr_sources.set_source(source_pair)

        # Identify
        self._id_checkpoint.set(ckpt_path)
        self._id_nameslist.set(nameslist)
        self._id_output.set(review_dir)
        self._id_sources.set_source(source_pair)

    # -------------------------------------------------------- Download tab --

    def _build_download(self, parent):
        section_header(parent, "Taxon")
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="Taxon rank:", width=22, anchor="e").pack(side="left")
        self._dl_rank = tk.StringVar(value="family")
        for val, txt in [("family","Family"), ("genus","Genus"), ("order","Order (client-side)")]:
            ttk.Radiobutton(row, text=txt, variable=self._dl_rank, value=val).pack(side="left", padx=6)

        self._dl_taxon    = labeled_entry(parent, "Taxon name:", "Ebenaceae")
        self._dl_continent = labeled_entry(parent, "Continent:", "AFRICA")

        section_header(parent, "Country filter  (mutually exclusive)")
        self._dl_countries         = labeled_entry(parent, "Include countries:", "")
        self._dl_exclude_countries = labeled_entry(parent, "Exclude countries:", "MG")
        ttk.Label(parent, text="  (space-separated ISO-2 codes, e.g.  ZA NG TZ)",
                  foreground="grey").pack(anchor="w", padx=30)

        section_header(parent, "Source  (DwC-A ZIP or live API)")
        self._dl_dwca = browse_entry(parent, "Local DwC-A ZIP:", "", mode="file")
        ttk.Label(parent, text="  (select a downloaded GBIF DwC-A ZIP to skip the API entirely)",
                  foreground="grey").pack(anchor="w", padx=30)

        section_header(parent, "Output")
        self._dl_output_dir = browse_entry(parent, "Output images dir:", "")
        self._dl_specsin    = browse_entry(parent, "specsin CSV path:",  "", mode="save")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Workers:", width=22, anchor="e").pack(side="left")
        self._dl_workers = tk.StringVar(value="8")
        ttk.Entry(row2, textvariable=self._dl_workers, width=6).pack(side="left")
        ttk.Label(row2, text="  Limit (0=all):", width=16, anchor="e").pack(side="left")
        self._dl_limit = tk.StringVar(value="0")
        ttk.Entry(row2, textvariable=self._dl_limit, width=8).pack(side="left")

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="IIIF image size:", width=22, anchor="e").pack(side="left")
        self._dl_iiif_size = tk.StringVar(value="")
        cb = ttk.Combobox(row3, textvariable=self._dl_iiif_size,
                          values=["", "1024", "2048", "4096", "max"], width=8)
        cb.pack(side="left")
        ttk.Label(row3, text="  px fit-box (or 'max') — blank = GBIF default thumbnail",
                  foreground="grey").pack(side="left")

        ttk.Button(parent, text="▶  Run Download",
                   command=self._run_download).pack(pady=10)

    def _run_download(self):
        cmd = self._build_download_cmd()
        self._launch(cmd)

    def _build_download_cmd(self) -> list[str]:
        dwca  = self._dl_dwca.get().strip()
        rank  = self._dl_rank.get()
        taxon = self._dl_taxon.get().strip()
        if not dwca and not taxon:
            messagebox.showerror("Missing", "Enter a taxon name or select a DwC-A ZIP."); return []

        cmd = [sys.executable, str(SCRIPTS["download"])]
        if dwca:
            cmd += ["--dwca", dwca]
        if taxon:
            cmd += [f"--{rank}", taxon]
        cont = self._dl_continent.get().strip()
        if cont:
            cmd += ["--continent", cont]

        inc = self._dl_countries.get().strip().split()
        exc = self._dl_exclude_countries.get().strip().split()
        if inc and exc:
            messagebox.showerror("Conflict",
                "Use either Include countries OR Exclude countries, not both."); return
        if inc:
            cmd += ["--countries"] + inc
        if exc:
            cmd += ["--exclude-countries"] + exc

        out_dir  = self._dl_output_dir.get().strip()
        specsin  = self._dl_specsin.get().strip()
        workers  = self._dl_workers.get().strip()
        limit    = self._dl_limit.get().strip()

        if out_dir:  cmd += ["--output-dir", out_dir]
        if specsin:  cmd += ["--specsin", specsin]
        if workers:  cmd += ["--workers", workers]
        if limit and limit != "0": cmd += ["--limit", limit]
        iiif = self._dl_iiif_size.get().strip()
        if iiif: cmd += ["--iiif-size", iiif]

        return cmd

    # ------------------------------------------------- Filter & Crop tab --

    def _build_filter_crop(self, parent):
        section_header(parent, "Paths")
        self._fc_input   = browse_entry(parent, "Input images dir:", "")
        self._fc_output  = browse_entry(parent, "Output images dir:", "")
        ttk.Label(parent, text="  (set same as input to overwrite originals in-place)",
                  foreground="grey").pack(anchor="w", padx=30)
        self._fc_specsin = browse_entry(parent, "specsin CSV (optional):", "", mode="file")
        ttk.Label(parent, text="  (if set, rejected images are marked hasfile=False in CSV)",
                  foreground="grey").pack(anchor="w", padx=30)

        section_header(parent, "Steps")
        chk_row = ttk.Frame(parent)
        chk_row.pack(fill="x", padx=8, pady=2)
        self._fc_do_filter = tk.BooleanVar(value=True)
        self._fc_do_crop   = tk.BooleanVar(value=True)
        ttk.Checkbutton(chk_row, text="Filter non-herbarium images",
                        variable=self._fc_do_filter).pack(side="left", padx=8)
        ttk.Checkbutton(chk_row, text="Crop white borders",
                        variable=self._fc_do_crop).pack(side="left", padx=8)

        section_header(parent, "Filter options")
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="Filter method:", width=22, anchor="e").pack(side="left")
        self._fc_method = ttk.Combobox(row, values=["clip", "hsv"], width=8, state="readonly")
        self._fc_method.set("clip")
        self._fc_method.pack(side="left")
        ttk.Label(row, text="  Confidence:", anchor="e").pack(side="left", padx=(12, 0))
        self._fc_confidence = tk.StringVar(value="0.6")
        ttk.Entry(row, textvariable=self._fc_confidence, width=6).pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="HSV white ratio:", width=22, anchor="e").pack(side="left")
        self._fc_hsv_white = tk.StringVar(value="0.25")
        ttk.Entry(row2, textvariable=self._fc_hsv_white, width=6).pack(side="left")
        ttk.Label(row2, text="  HSV saturation:", anchor="e").pack(side="left", padx=(12, 0))
        self._fc_hsv_sat = tk.StringVar(value="40")
        ttk.Entry(row2, textvariable=self._fc_hsv_sat, width=6).pack(side="left")

        section_header(parent, "Crop / performance options")
        row3 = ttk.Frame(parent)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="Crop padding (px):", width=22, anchor="e").pack(side="left")
        self._fc_padding = tk.StringVar(value="10")
        ttk.Entry(row3, textvariable=self._fc_padding, width=6).pack(side="left")
        ttk.Label(row3, text="  Batch size:", anchor="e").pack(side="left", padx=(12, 0))
        self._fc_batch = tk.StringVar(value="32")
        ttk.Entry(row3, textvariable=self._fc_batch, width=6).pack(side="left")
        ttk.Label(row3, text="  Workers:", anchor="e").pack(side="left", padx=(12, 0))
        self._fc_workers = tk.StringVar(value="8")
        ttk.Entry(row3, textvariable=self._fc_workers, width=6).pack(side="left")

        row4 = ttk.Frame(parent)
        row4.pack(fill="x", padx=8, pady=2)
        self._fc_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(row4, text="Force reprocess (ignore already-processed images)",
                        variable=self._fc_force).pack(side="left", padx=(130, 0))
        ttk.Label(parent, text="  (by default, images already in the output folder are skipped on re-run)",
                  foreground="grey").pack(anchor="w", padx=30)

        ttk.Button(parent, text="▶  Run Filter & Crop",
                   command=self._run_filter_crop).pack(pady=10)

    def _run_filter_crop(self):
        cmd = self._build_filter_crop_cmd()
        if cmd:
            self._launch(cmd)

    def _build_filter_crop_cmd(self) -> list[str]:
        inp = self._fc_input.get().strip()
        if not inp:
            messagebox.showerror("Missing", "Enter an input directory."); return []

        inp_path = Path(inp)
        out = self._fc_output.get().strip()
        out_path = Path(out) if out else None

        cmd = [sys.executable, str(SCRIPTS["filter_and_crop"]), "--input-dir", inp]

        if out_path and out_path.resolve() == inp_path.resolve():
            cmd += ["--in-place"]
        elif out:
            cmd += ["--output-dir", out]
        else:
            cmd += ["--in-place"]

        if not self._fc_do_filter.get():
            cmd += ["--no-filter"]
        else:
            cmd += [
                "--filter-method", self._fc_method.get(),
                "--confidence",    self._fc_confidence.get(),
                "--hsv-white-ratio", self._fc_hsv_white.get(),
                "--hsv-saturation",  self._fc_hsv_sat.get(),
                "--batch-size",    self._fc_batch.get(),
            ]

        if not self._fc_do_crop.get():
            cmd += ["--no-crop"]
        else:
            cmd += ["--crop-padding", self._fc_padding.get()]

        cmd += ["--workers", self._fc_workers.get()]

        if self._fc_force.get():
            cmd += ["--force"]

        specsin = self._fc_specsin.get().strip()
        if specsin:
            cmd += ["--specsin", specsin]

        return cmd

    # ---------------------------------------------------------- Resize tab --

    def _build_resize(self, parent):
        section_header(parent, "Paths")
        self._rs_input  = browse_entry(parent, "Input images dir:", "")
        self._rs_output = browse_entry(parent, "Output images dir:", "")
        ttk.Label(parent, text="  (leave blank to use --in-place)",
                  foreground="grey").pack(anchor="w", padx=30)

        section_header(parent, "Options")
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="Max size (px):", width=22, anchor="e").pack(side="left")
        self._rs_maxsize = tk.StringVar(value="1024")
        ttk.Entry(row, textvariable=self._rs_maxsize, width=8).pack(side="left")
        self._rs_noupscale = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="No upscale", variable=self._rs_noupscale).pack(side="left", padx=12)
        self._rs_nodali = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Force PIL (no DALI)", variable=self._rs_nodali).pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Batch size:", width=22, anchor="e").pack(side="left")
        self._rs_batch = tk.StringVar(value="64")
        ttk.Entry(row2, textvariable=self._rs_batch, width=6).pack(side="left")
        ttk.Label(row2, text="  PIL workers:", width=14, anchor="e").pack(side="left")
        self._rs_workers = tk.StringVar(value="8")
        ttk.Entry(row2, textvariable=self._rs_workers, width=6).pack(side="left")

        ttk.Button(parent, text="▶  Run Resize",
                   command=self._run_resize).pack(pady=10)

    def _run_resize(self):
        inp = self._rs_input.get().strip()
        if not inp:
            messagebox.showerror("Missing", "Enter an input directory."); return

        cmd = [sys.executable, str(SCRIPTS["resize"]), "--input-dir", inp]

        out = self._rs_output.get().strip()
        if out:
            cmd += ["--output-dir", out]
        else:
            cmd += ["--in-place"]

        cmd += ["--max-size", self._rs_maxsize.get()]
        if self._rs_noupscale.get(): cmd += ["--no-upscale"]
        if self._rs_nodali.get():    cmd += ["--no-dali"]
        cmd += ["--batch-size", self._rs_batch.get()]
        cmd += ["--workers",    self._rs_workers.get()]

        self._launch(cmd)

    # ----------------------------------------------------------- Train tab --

    def _build_train(self, parent):
        section_header(parent, "Data sources  (specsin CSV : images directory)")
        fr = ttk.Frame(parent)
        fr.pack(fill="x", padx=8, pady=2)
        self._tr_sources = SourceList(fr)
        self._tr_sources.pack(fill="x")

        section_header(parent, "Output")
        self._tr_output = browse_entry(parent, "Output / run dir:", "")

        section_header(parent, "Model")
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="timm model:", width=22, anchor="e").pack(side="left")
        self._tr_model = tk.StringVar(value=TIMM_MODELS[0])
        cb = ttk.Combobox(row, textvariable=self._tr_model, values=TIMM_MODELS, width=40)
        cb.pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Image size (px):", width=22, anchor="e").pack(side="left")
        self._tr_imgsz   = tk.StringVar(value="640")
        ttk.Entry(row2, textvariable=self._tr_imgsz, width=6).pack(side="left")
        ttk.Label(row2, text="  Batch size:", width=12, anchor="e").pack(side="left")
        self._tr_batch   = tk.StringVar(value="4")
        ttk.Entry(row2, textvariable=self._tr_batch, width=6).pack(side="left")
        ttk.Label(row2, text="  Grad accum:", width=12, anchor="e").pack(side="left")
        self._tr_accum   = tk.StringVar(value="2")
        ttk.Entry(row2, textvariable=self._tr_accum, width=6).pack(side="left")
        ttk.Label(row2, text="  GPUs:", width=6, anchor="e").pack(side="left")
        self._tr_gpus    = tk.StringVar(value="2")
        ttk.Entry(row2, textvariable=self._tr_gpus, width=4).pack(side="left")

        section_header(parent, "Training stages")
        row3 = ttk.Frame(parent)
        row3.pack(fill="x", padx=8, pady=2)
        for label, var_name, default in [
            ("Stage1 epochs:", "_tr_s1ep", "4"),
            ("Stage1 LR:",     "_tr_s1lr", "0.005"),
            ("Stage2 epochs:", "_tr_s2ep", "15"),
            ("Stage2 LR:",     "_tr_s2lr", "0.0001"),
        ]:
            ttk.Label(row3, text=label, anchor="e", width=13).pack(side="left")
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(row3, textvariable=var, width=8).pack(side="left", padx=(0, 6))

        section_header(parent, "Classification target")
        cls_row = ttk.Frame(parent)
        cls_row.pack(fill="x", padx=8, pady=2)
        ttk.Label(cls_row, text="Label level:", width=22, anchor="e").pack(side="left")
        self._tr_label_level = tk.StringVar(value="species")
        for val, txt in [("species", "Species"), ("genus", "Genus"), ("family", "Family")]:
            ttk.Radiobutton(cls_row, text=txt, variable=self._tr_label_level,
                            value=val).pack(side="left", padx=6)

        hier_row = ttk.Frame(parent)
        hier_row.pack(fill="x", padx=8, pady=2)
        self._tr_hierarchical = tk.BooleanVar(value=False)
        ttk.Checkbutton(hier_row, text="Hierarchical multi-head (joint loss)",
                        variable=self._tr_hierarchical).pack(side="left", padx=(130, 12))
        ttk.Label(hier_row, text="Species w:", anchor="e").pack(side="left")
        self._tr_w_species = tk.StringVar(value="1.0")
        ttk.Entry(hier_row, textvariable=self._tr_w_species, width=5).pack(side="left", padx=2)
        ttk.Label(hier_row, text="Genus w:", anchor="e").pack(side="left", padx=(8, 0))
        self._tr_w_genus = tk.StringVar(value="0.5")
        ttk.Entry(hier_row, textvariable=self._tr_w_genus, width=5).pack(side="left", padx=2)
        ttk.Label(hier_row, text="Family w:", anchor="e").pack(side="left", padx=(8, 0))
        self._tr_w_family = tk.StringVar(value="0.0")
        ttk.Entry(hier_row, textvariable=self._tr_w_family, width=5).pack(side="left", padx=2)
        ttk.Label(parent, text="  (hierarchical overrides label level; set family weight > 0 to include family head)",
                  foreground="grey").pack(anchor="w", padx=30)

        section_header(parent, "Logging")
        row4 = ttk.Frame(parent)
        row4.pack(fill="x", padx=8, pady=2)
        ttk.Label(row4, text="WandB project:", width=22, anchor="e").pack(side="left")
        self._tr_wandb_proj = tk.StringVar(value="")
        ttk.Entry(row4, textvariable=self._tr_wandb_proj, width=22).pack(side="left")
        ttk.Label(row4, text="  Run name:", width=10, anchor="e").pack(side="left")
        self._tr_wandb_name = tk.StringVar(value="herbarium_run")
        ttk.Entry(row4, textvariable=self._tr_wandb_name, width=20).pack(side="left")

        row5 = ttk.Frame(parent)
        row5.pack(fill="x", padx=8, pady=2)
        self._tr_resume = browse_entry(parent, "Resume checkpoint:", "", mode="file")

        ttk.Button(parent, text="▶  Run Training",
                   command=self._run_train).pack(pady=10)

    def _run_train(self):
        sources = self._tr_sources.get_sources()
        if not sources:
            messagebox.showerror("Missing", "Add at least one data source."); return
        out = self._tr_output.get().strip()
        if not out:
            messagebox.showerror("Missing", "Enter an output directory."); return

        cmd = [sys.executable, str(SCRIPTS["train"]),
               "--sources"] + sources + [
               "--output-dir",   out,
               "--model",        self._tr_model.get(),
               "--image-sz",     self._tr_imgsz.get(),
               "--batch-size",   self._tr_batch.get(),
               "--accum",        self._tr_accum.get(),
               "--num-gpus",     self._tr_gpus.get(),
               "--stage1-epochs",self._tr_s1ep.get(),
               "--stage1-lr",    self._tr_s1lr.get(),
               "--stage2-epochs",self._tr_s2ep.get(),
               "--stage2-lr",    self._tr_s2lr.get(),
        ]

        if self._tr_hierarchical.get():
            cmd += [
                "--hierarchical",
                "--species-weight", self._tr_w_species.get(),
                "--genus-weight",   self._tr_w_genus.get(),
                "--family-weight",  self._tr_w_family.get(),
            ]
        else:
            cmd += ["--label-level", self._tr_label_level.get()]

        proj = self._tr_wandb_proj.get().strip()
        if proj:
            cmd += ["--wandb-project", proj, "--wandb-run-name", self._tr_wandb_name.get()]
        else:
            cmd += ["--no-wandb"]

        resume = self._tr_resume.get().strip()
        if resume:
            cmd += ["--resume", resume]

        self._launch(cmd)

    # -------------------------------------------------------- Identify tab --

    def _build_identify(self, parent):
        section_header(parent, "Model")
        self._id_checkpoint = browse_entry(parent, "Checkpoint (.ckpt):", "", mode="file")
        self._id_nameslist  = browse_entry(parent, "nameslist.json:",     "", mode="file")
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="timm model (if needed):", width=22, anchor="e").pack(side="left")
        self._id_model = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self._id_model, width=40).pack(side="left")

        section_header(parent, "Data sources  (specsin CSV : images directory)")
        fr = ttk.Frame(parent)
        fr.pack(fill="x", padx=8, pady=2)
        self._id_sources = SourceList(fr)
        self._id_sources.pack(fill="x")

        section_header(parent, "Output")
        self._id_output = browse_entry(parent, "Review output dir:", "")

        section_header(parent, "Thresholds")
        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Mismatch threshold:", width=22, anchor="e").pack(side="left")
        self._id_thresh = tk.StringVar(value="0.7")
        ttk.Entry(row2, textvariable=self._id_thresh, width=7).pack(side="left")
        ttk.Label(row2, text="  Low-conf flag (0=off):", anchor="e").pack(side="left", padx=(8, 0))
        self._id_lowconf = tk.StringVar(value="0.3")
        ttk.Entry(row2, textvariable=self._id_lowconf, width=7).pack(side="left")

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="Image size (px):", width=22, anchor="e").pack(side="left")
        self._id_imgsz = tk.StringVar(value="640")
        ttk.Entry(row3, textvariable=self._id_imgsz, width=6).pack(side="left")
        ttk.Label(row3, text="  Batch size:", width=12, anchor="e").pack(side="left")
        self._id_batch = tk.StringVar(value="32")
        ttk.Entry(row3, textvariable=self._id_batch, width=6).pack(side="left")

        ttk.Button(parent, text="▶  Run Identify",
                   command=self._run_identify).pack(pady=10)

    def _run_identify(self):
        ckpt = self._id_checkpoint.get().strip()
        nl   = self._id_nameslist.get().strip()
        if not ckpt or not nl:
            messagebox.showerror("Missing", "Specify checkpoint and nameslist."); return
        sources = self._id_sources.get_sources()
        if not sources:
            messagebox.showerror("Missing", "Add at least one data source."); return
        out = self._id_output.get().strip()
        if not out:
            messagebox.showerror("Missing", "Enter an output directory."); return

        cmd = [sys.executable, str(SCRIPTS["identify"]),
               "--checkpoint", ckpt,
               "--nameslist",  nl,
               "--sources"]   + sources + [
               "--output-dir", out,
               "--threshold",       self._id_thresh.get(),
               "--low-conf-threshold", self._id_lowconf.get(),
               "--image-sz",   self._id_imgsz.get(),
               "--batch-size", self._id_batch.get(),
        ]
        model = self._id_model.get().strip()
        if model:
            cmd += ["--model", model]

        self._launch(cmd)

    # --------------------------------------------------------- Run All tab --

    def _build_run_all(self, parent):
        ttk.Label(parent, text="Runs steps 1–5 in sequence using the settings\n"
                               "configured in each individual tab.",
                  justify="center", font=("TkDefaultFont", 13)).pack(pady=20)

        ttk.Label(parent,
                  text="Configure each step in its own tab first, then click Run All.",
                  foreground="grey").pack()

        steps_frame = ttk.LabelFrame(parent, text="Steps to run")
        steps_frame.pack(padx=20, pady=10, fill="x")
        self._run_dl   = tk.BooleanVar(value=True)
        self._run_fc   = tk.BooleanVar(value=True)
        self._run_rs   = tk.BooleanVar(value=True)
        self._run_tr   = tk.BooleanVar(value=True)
        self._run_id   = tk.BooleanVar(value=True)
        for var, label in [(self._run_dl, "1  Download"),
                           (self._run_fc, "2  Filter & Crop"),
                           (self._run_rs, "3  Resize"),
                           (self._run_tr, "4  Train"),
                           (self._run_id, "5  Identify")]:
            ttk.Checkbutton(steps_frame, text=label, variable=var).pack(anchor="w", padx=10, pady=2)

        ttk.Button(parent, text="▶▶  Run Full Pipeline",
                   command=self._run_all).pack(pady=16)

    def _run_all(self):
        if self._runner and self._runner.is_running():
            messagebox.showwarning("Busy", "A process is already running."); return

        # Build step list in order
        steps = []
        if self._run_dl.get(): steps.append(("Download",       self._build_download_cmd))
        if self._run_fc.get(): steps.append(("Filter & Crop",  self._build_filter_crop_cmd))
        if self._run_rs.get(): steps.append(("Resize",         self._build_resize_cmd))
        if self._run_tr.get(): steps.append(("Train",          self._build_train_cmd))
        if self._run_id.get(): steps.append(("Identify",       self._build_identify_cmd))

        if not steps:
            messagebox.showinfo("Nothing to do", "No steps selected."); return

        self._pipeline_steps = steps
        self._run_next_step()

    def _run_next_step(self):
        if not self._pipeline_steps:
            self._log.insert(tk.END, "\n✓ Pipeline complete.\n", "done")
            self._status.set("Pipeline complete")
            return
        name, cmd_fn = self._pipeline_steps.pop(0)
        try:
            cmd = cmd_fn()
        except ValueError as e:
            messagebox.showerror("Config error", str(e))
            self._pipeline_steps = None
            return
        self._log.insert(tk.END, f"\n{'='*60}\n  Step: {name}\n{'='*60}\n", "cmd")
        self._launch(cmd, on_finish=self._step_done)

    def _step_done(self, rc: int):
        if rc != 0:
            self._log.insert(tk.END, "\n✗ Step failed — pipeline aborted.\n", "err")
            self._pipeline_steps = None
        else:
            self._run_next_step()

    def _build_resize_cmd(self) -> list[str]:
        inp = self._rs_input.get().strip()
        if not inp: raise ValueError("Resize: enter an input directory.")
        cmd  = [sys.executable, str(SCRIPTS["resize"]), "--input-dir", inp]
        out  = self._rs_output.get().strip()
        cmd += ["--output-dir", out] if out else ["--in-place"]
        cmd += ["--max-size", self._rs_maxsize.get()]
        if self._rs_noupscale.get(): cmd += ["--no-upscale"]
        if self._rs_nodali.get():    cmd += ["--no-dali"]
        cmd += ["--batch-size", self._rs_batch.get(), "--workers", self._rs_workers.get()]
        return cmd

    def _build_train_cmd(self) -> list[str]:
        sources = self._tr_sources.get_sources()
        out     = self._tr_output.get().strip()
        if not sources: raise ValueError("Train: add at least one data source.")
        if not out:     raise ValueError("Train: enter an output directory.")
        cmd = [sys.executable, str(SCRIPTS["train"]),
               "--sources"] + sources + [
               "--output-dir",   out,
               "--model",        self._tr_model.get(),
               "--image-sz",     self._tr_imgsz.get(),
               "--batch-size",   self._tr_batch.get(),
               "--accum",        self._tr_accum.get(),
               "--num-gpus",     self._tr_gpus.get(),
               "--stage1-epochs",self._tr_s1ep.get(),
               "--stage1-lr",    self._tr_s1lr.get(),
               "--stage2-epochs",self._tr_s2ep.get(),
               "--stage2-lr",    self._tr_s2lr.get(),
        ]
        proj = self._tr_wandb_proj.get().strip()
        cmd += (["--wandb-project", proj, "--wandb-run-name", self._tr_wandb_name.get()]
                if proj else ["--no-wandb"])
        resume = self._tr_resume.get().strip()
        if resume: cmd += ["--resume", resume]
        return cmd

    def _build_identify_cmd(self) -> list[str]:
        ckpt    = self._id_checkpoint.get().strip()
        nl      = self._id_nameslist.get().strip()
        sources = self._id_sources.get_sources()
        out     = self._id_output.get().strip()
        if not ckpt or not nl: raise ValueError("Identify: specify checkpoint and nameslist.")
        if not sources:        raise ValueError("Identify: add at least one data source.")
        if not out:            raise ValueError("Identify: enter an output directory.")
        cmd = [sys.executable, str(SCRIPTS["identify"]),
               "--checkpoint", ckpt, "--nameslist", nl,
               "--sources"]   + sources + [
               "--output-dir", out,
               "--threshold",           self._id_thresh.get(),
               "--low-conf-threshold",  self._id_lowconf.get(),
               "--image-sz",   self._id_imgsz.get(),
               "--batch-size", self._id_batch.get(),
        ]
        model = self._id_model.get().strip()
        if model: cmd += ["--model", model]
        return cmd

    # ----------------------------------------------------------- Launch --

    def _launch(self, cmd: list[str], on_finish=None):
        if self._runner and self._runner.is_running():
            messagebox.showwarning("Busy", "A process is already running."); return

        def finished(rc):
            self._stop_btn.config(state="disabled")
            if on_finish:
                on_finish(rc)

        self._runner = ProcessRunner(self._log, self._status, on_finish=finished)
        self._stop_btn.config(state="normal")
        self._runner.run(cmd)

    def _stop(self):
        if self._runner:
            self._runner.terminate()
            self._pipeline_steps = None
            self._status.set("Stopped by user")
            self._stop_btn.config(state="disabled")

    def _save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text", "*.txt"), ("All", "*")])
        if path:
            Path(path).write_text(self._log.get("1.0", tk.END))


# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
