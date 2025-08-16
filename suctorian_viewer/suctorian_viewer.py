# -*- coding: utf-8 -*-
"""

@author: coylelab @ uw-madison

"""


# two_pane_tentacle_viewer.py
import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLViewWidget, GLMeshItem, MeshData
import matplotlib.colors as mcolors
import random
import time
import matplotlib
import matplotlib.cm as cm
import os

from ._morphospace_coordinates import MORPHOSPACE_BOUNDARY_X, MORPHOSPACE_BOUNDARY_Y


def get_time_colors(times):
    norm = matplotlib.colors.Normalize(vmin=times.min(), vmax=times.max())
    turbo = cm.get_cmap("turbo")
    rgba = turbo(norm(times))
    return rgba[:, :3]  # strip alpha


class MorphPlayback(QtWidgets.QWidget):
    def __init__(self, summary_df, show_fn, parent=None):
        """
        summary_df: DataFrame with 'N' and 'L' columns.
        show_fn: function(cell_id) -> displays that morphology.
        """
        super().__init__(parent)

        self.summary = summary_df
        self.show_fn = show_fn

        # Sort order: N asc, L asc
        self.play_order = self.summary.sort_values(["N", "L"]).index.to_list()
        self.current_idx = 0

        # Button
        self.button = QtWidgets.QPushButton("▶")
        self.button.setFixedSize(24, 24)
        self.button.clicked.connect(self.toggle_play)

        # Timer for ~30fps
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.advance_frame)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.button)
        layout.setContentsMargins(0, 0, 0, 0)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.button.setText("▶")
        else:
            self.timer.start()
            self.button.setText("⏸")

    def advance_frame(self):
        if self.current_idx >= len(self.play_order):
            self.timer.stop()
            self.button.setText("▶")
            self.current_idx = 0
            return

        cell_id = self.play_order[self.current_idx]
        self.show_fn(cell_id)
        self.current_idx += 1


class TimePlayback(QtWidgets.QWidget):
    def __init__(self, time_slider, update_fn, parent=None):
        """
        time_slider: QSlider controlling time (0..T-1)
        update_fn: function(t) -> updates visualization to time t
        """
        super().__init__(parent)

        self.time_slider = time_slider
        self.update_fn = update_fn

        # Button
        self.button = QtWidgets.QPushButton("▶")
        self.button.setFixedSize(24, 24)
        self.button.clicked.connect(self.toggle_play)

        # Timer ~30fps
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  
        self.timer.timeout.connect(self.advance_frame)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.button)
        layout.setContentsMargins(0, 0, 0, 0)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.button.setText("▶")
        else:
            self.timer.start()
            self.button.setText("⏸")

    def advance_frame(self):
        current = self.time_slider.value()
        if current < self.time_slider.maximum():
            self.time_slider.setValue(current + 1)
        else:
            self.time_slider.setValue(0)  # loop back
        self.update_fn(self.time_slider.value())



class TentacleViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        #self.path_to=os.path.dirname(__file__)
        #self.splash()
        self.load_csv()


        # set a cell for drawing
        self.current_index = random.randint(0, len(self.cell_names) - 1)

        # timing
        self.t0 = np.min(self.ms["time"].values)
        self.current_time = self.t0

        self.plot_items = []  # tracks GL items in the 3D view for cleanup
        self.tentacle_spheres = []  # convenience list for tentacle sphere items

        # initial sphere radius (slider units -> px)
        self.sphere_radius_px = 15

        # boundary toggle
        self.show_boundary = True

        # initalize UI and set a cekk
        self.init_ui()
        self.initialize_boundary()
        self.initialize_scatterplot()
        self.load_cell_by_index(self.current_index)

    def splash(self):
        splash_image_path = os.path.join(self.path_to, "splash.png")
        pixmap = QtGui.QPixmap(splash_image_path)

        if pixmap.isNull():
            raise FileNotFoundError(f"Splash image not found: {splash_image_path}")

        splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
        splash.show()
        time.sleep(2)

    def _compute_summary(self):
        # tentacle length: if user provided 'tentacle_length' use it, otherwise compute euclidean distance
        df = self.ms.copy()
        if "tentacle_length" not in df.columns:
            df["tentacle_length"] = np.sqrt(
                df["tx_px"] ** 2 + df["ty_px"] ** 2 + df["tz_px"] ** 2
            )

        if "time" in self.ms.columns:
            summary = (
                df.groupby(["cell_name", "time"])
                .agg(N=("tentacle_id", "nunique"), L=("tentacle_length", "mean"))
                .reset_index()
            )
        else:
            summary = (
                df.groupby("cell_name")
                .agg(N=("tentacle_id", "nunique"), L=("tentacle_length", "mean"))
                .reset_index()
            )

        return summary

    def init_ui(self):
        self.setWindowTitle("Tentacle Viewer — 3D + Density")
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # ------------------ Left: 3D viewer + controls ------------------
        left_v = QtWidgets.QVBoxLayout()

        self.csv_name_widget = QtWidgets.QLabel(f"Viewing dataset: {self.csv_name}")
        left_v.addWidget(self.csv_name_widget)

        # 3D view
        self.view = GLViewWidget()
        self.view.opts["distance"] = 150
        self.view.setBackgroundColor((230, 230, 230, 255))  # light grey bg
        self.view.orbit(0, 90)
        left_v.addWidget(self.view, stretch=1)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_v)

        # controls: dropdown, prev/next, show spheres, radius slider

        ctrl_layout = QtWidgets.QHBoxLayout()

        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(self.cell_names)
        self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        ctrl_layout.addWidget(self.dropdown)

        self.prev_btn = QtWidgets.QPushButton("←")
        self.prev_btn.clicked.connect(self.prev_cell)
        ctrl_layout.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("→")
        self.next_btn.clicked.connect(self.next_cell)
        ctrl_layout.addWidget(self.next_btn)

        # summary_df: your per-cell summary DataFrame
        # show_fn: a function that draws the morphology for the given cell ID
        self.playback_widget = MorphPlayback(self.summary, self.load_cell_by_index)
        ctrl_layout.addWidget(self.playback_widget)

        self.show_spheres_cb = QtWidgets.QCheckBox("Tip spheres")
        self.show_spheres_cb.setChecked(True)
        self.show_spheres_cb.stateChanged.connect(self.update_tentacle_spheres)
        ctrl_layout.addWidget(self.show_spheres_cb)

        left_v.addLayout(ctrl_layout)

        self.radius_label = QtWidgets.QLabel(f"Radius: {self.sphere_radius_px}")
        left_v.addWidget(self.radius_label)
        self.radius_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(100)
        self.radius_slider.setValue(int(self.sphere_radius_px))
        self.radius_slider.valueChanged.connect(self.on_radius_changed)
        left_v.addWidget(self.radius_slider)

        # --- Time slider ---
        self.time_label = QtWidgets.QLabel(f"Time: {self.current_time}")
        left_v.addWidget(self.time_label)
        self.time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.time_slider.setTickInterval(1)
        self.time_slider.valueChanged.connect(self.on_time_changed)

        self.time_playback = TimePlayback(self.time_slider, self.on_time_changed)


        # Time slider + play button side by side
        time_layout = QtWidgets.QHBoxLayout()
        time_layout.addWidget(self.time_slider, 1)       # stretch
        time_layout.addWidget(self.time_playback)

        left_v.addLayout(time_layout)
        
        # ------------------ Right: density plot (N vs L) ------------------
        right_v = QtWidgets.QVBoxLayout()

        # add dataloader button
        self.load_btn = QtWidgets.QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.handle_load_csv_button)
        right_v.addWidget(self.load_btn)

        # add checkbox for morphospace
        self.show_boundary_cb = QtWidgets.QCheckBox("Morphospace boundary")
        self.show_boundary_cb.setChecked(True)
        self.show_boundary_cb.stateChanged.connect(self.toggle_boundary)
        right_v.addWidget(self.show_boundary_cb)

        # add main tentacle plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Tentacle Count (N)")
        self.plot_widget.setLabel("left", "Avg Tentacle Length (L)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground("w")

        right_v.addWidget(self.plot_widget, stretch=1)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_v)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.splitter)

    def initialize_boundary(self):
        # boundary polygon (background) ---
        self.boundary_item = pg.PlotDataItem(
            x=MORPHOSPACE_BOUNDARY_X,
            y=MORPHOSPACE_BOUNDARY_Y,
            pen=pg.mkPen("r", width=2),
            connect="all",
        )
        self.boundary_item.setZValue(-10)  # ensure it's behind points
        self.plot_widget.addItem(self.boundary_item)

    ### add clickable lightweight scatter (one point per cell) — transparent/fast
    def initialize_scatterplot(self):
        if "time" in self.ms.columns:
            t0 = self.ms["time"].values[0]

            first_time = self.summary[self.summary["time"] == t0]

            spots_pos = np.column_stack(
                [first_time["N"].values, first_time["L"].values]
            )
            data = first_time["cell_name"].tolist()
        else:
            spots_pos = np.column_stack(
                [self.summary["N"].values, self.summary["L"].values]
            )
            data = self.summary["cell_name"].tolist()

        brush_default = pg.mkBrush(25, 25, 25, 50)
        brushes = [brush_default for _ in range(len(spots_pos))]

        # index data for click-to-change
        self.scatter_all = pg.ScatterPlotItem(
            pos=spots_pos, brush=brushes, pen=None, size=10, data=data
        )

        self.scatter_all.sigClicked.connect(self.on_density_point_clicked)
        self.plot_widget.addItem(self.scatter_all)

        # add highlight point for current cell (red, visible)
        self.highlight_point = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.highlight_point)

        self.highlight_line = pg.PlotDataItem(
            pen=pg.mkPen(color="k", width=1), connect="all"
        )

        self.plot_widget.addItem(self.highlight_line)

    # ------------------------------- UI callbacks -------------------------------
    def on_radius_changed(self, v):
        self.sphere_radius_px = float(v)
        # update spheres live
        self.update_tentacle_spheres()
        self.radius_label.setText(f"Radius: {self.sphere_radius_px}")

    def prev_cell(self):
        self.load_cell_by_index((self.current_index - 1) % len(self.cell_names))

    def next_cell(self):
        self.load_cell_by_index((self.current_index + 1) % len(self.cell_names))

    def on_dropdown_changed(self, idx):
        cell_name = self.dropdown.currentText()
        self.load_cell_by_name(cell_name)

    def on_density_point_clicked(self, plot, points):
        if points is None or len(points) == 0:
            return
        point = points[0]  # take just the first clicked point
        cell_name = point.data()
        if cell_name is None:
            return
        self.load_cell_by_name(cell_name)
        self.current_time = self.t0
        #adjust playback widget to start at current index.
        main_idx=self.cell_names.index(cell_name)
        player_idx=self.playback_widget.play_order.index(main_idx)
        self.playback_widget.current_idx=player_idx

    def toggle_boundary(self, state):
        if state == QtCore.Qt.Checked:
            self.boundary_item.show()
            self.boundary_item.setZValue(-100)  # always behind points
        else:
            self.boundary_item.hide()

    def load_csv(self):
        # prompt for file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open morphology CSV", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return  # cancelled

        filename = file_path.split("/")
        self.csv_name = filename[-1]

        self.ms = pd.read_csv(file_path)
        if "tentacle_id" not in self.ms.columns:
            # synthetic id per cell (0..k-1)
            self.ms["tentacle_id"] = self.ms.groupby("cell_name").cumcount()

        # compute and set per-cell summary (N, L)
        self.cell_names = list(self.ms["cell_name"].unique())
        self.cell_names.sort()
        self.summary = self._compute_summary()

    def handle_load_csv_button(self):
        self.load_csv()

        self.csv_name_widget.setText(f"Viewing dataset: {self.csv_name}")
        self.plot_widget.clear()
        # indexing
        self.current_index = random.randint(0, len(self.cell_names) - 1)

        # timing
        self.t0 = np.min(self.ms["time"].values)
        self.current_time = self.t0

        # refresh
        self.initialize_boundary()
        self.initialize_scatterplot()
        self.load_cell_by_index(self.current_index)

    # ------------------------------- drawing helpers -------------------------------
    def clear_3d_items(self):
        # remove and clear any items we've previously added
        for it in getattr(self, "plot_items", []):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.plot_items = []
        # also clear tentacle spheres
        for s in getattr(self, "tentacle_spheres", []):
            try:
                self.view.removeItem(s)
            except Exception:
                pass
        self.tentacle_spheres = []

    def create_mesh_sphere(self, radius, rows=20, cols=20, color=(1, 0, 0, 0.1)):
        md = MeshData.sphere(rows=rows, cols=cols, radius=radius)
        mesh = GLMeshItem(
            meshdata=md, smooth=True, shader="shaded", color=color, drawEdges=False
        )
        mesh.setGLOptions("translucent")
        return mesh

    def load_cell_by_name(self, cell_name):
        if cell_name not in self.cell_names:
            print(f"Unknown cell_name: {cell_name}")
            return
        idx = self.cell_names.index(cell_name)
        self.current_index = idx

        self.dropdown.blockSignals(True)
        self.dropdown.setCurrentText(cell_name)
        self.dropdown.blockSignals(False)

        # update times and sliders as before
        if "time" in self.ms.columns:
            times = sorted(
                self.ms.loc[self.ms["cell_name"] == cell_name, "time"].unique()
            )
            self.cell_times = times
            self.time_slider.blockSignals(True)
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(len(times) - 1)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
            self.current_time = times[0]
        else:
            self.cell_times = [None]
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(0)
            self.current_time = None

        self.plot_cell_3d(cell_name, time=self.current_time)
        self.update_density_highlight()

    def load_cell_by_index(self, idx: int):
        if idx < 0 or idx >= len(self.cell_names):
            return
        self.current_index = idx
        self.dropdown.blockSignals(True)
        self.dropdown.setCurrentIndex(idx)
        self.dropdown.blockSignals(False)

        cell_name = self.cell_names[self.current_index]

        # update slider range for that cell
        if "time" in self.ms.columns:
            times = sorted(
                self.ms.loc[self.ms["cell_name"] == cell_name, "time"].unique()
            )
            self.cell_times = times
            self.time_slider.blockSignals(True)
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(len(times) - 1)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
            self.current_time = times[0]  # NEW
        else:
            self.cell_times = [None]
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(0)
            self.current_time = None

        # draw first timepoint for that cell
        self.plot_cell_3d(cell_name, time=self.cell_times[0])
        self.update_density_highlight()

    def on_time_changed(self, idx):
        if not hasattr(self, "cell_times") or not self.cell_times:
            return
        self.current_time = self.cell_times[idx]
        cell_name = self.cell_names[self.current_index]
        self.plot_cell_3d(cell_name, time=self.current_time)
        self.update_density_highlight()
        self.time_label.setText(f"Time: {self.current_time}")

    def plot_cell_3d(self, cell_name: str, time=None):
        self.clear_3d_items()
        if time is not None and "time" in self.ms.columns:
            cell_df = self.ms[
                (self.ms["cell_name"] == cell_name) & (self.ms["time"] == time)
            ]
        else:
            cell_df = self.ms[self.ms["cell_name"] == cell_name]

        if cell_df.empty:
            return

        tx = cell_df["tx_px"].values
        ty = cell_df["ty_px"].values
        tz = cell_df["tz_px"].values

        origin = np.array([0.0, 0.0, 0.0])

        for x, y, z in zip(tx, ty, tz):
            pts = np.array([origin, [x, y, z]])
            line = gl.GLLinePlotItem(
                pos=pts, color=(0.7, 0.7, 0.7, 1.0), width=12, antialias=True
            )
            self.view.addItem(line)
            self.plot_items.append(line)

        r = cell_df["cellbodyradius_um"].mean()

        body = self.create_mesh_sphere(
            radius=r, rows=24, cols=24, color=(0.7, 0.7, 0.7, 0.8)
        )
        self.view.addItem(body)
        self.plot_items.append(body)

        self.update_tentacle_spheres()

    def update_tentacle_spheres(self):
        # remove previous tentacle spheres
        for s in getattr(self, "tentacle_spheres", []):
            try:
                self.view.removeItem(s)
            except Exception:
                pass
        self.tentacle_spheres = []

        if not self.show_spheres_cb.isChecked():
            return

        cell_name = self.cell_names[self.current_index]
        if self.current_time is not None and "time" in self.ms.columns:
            cell_df = self.ms[
                (self.ms["cell_name"] == cell_name)
                & (self.ms["time"] == self.current_time)
            ]
        else:
            cell_df = self.ms[self.ms["cell_name"] == cell_name]

        if cell_df.empty:
            return

        # radius scale: slider maps directly to px (you can change scale mapping here)
        r = float(self.sphere_radius_px)

        for x, y, z in zip(cell_df["tx_px"], cell_df["ty_px"], cell_df["tz_px"]):
            sph = self.create_mesh_sphere(
                radius=r, rows=10, cols=10, color=(1.0, 0.0, 0.0, 0.12)
            )
            sph.translate(float(x), float(y), float(z))
            self.view.addItem(sph)
            self.tentacle_spheres.append(sph)
            # track for cleanup (optional)
            self.plot_items.append(sph)

    def update_density_highlight(self):
        # set the red highlight point to current cell's (N, L)

        df = self.ms.copy()
        if "tentacle_length" not in df.columns:
            df["tentacle_length"] = np.sqrt(
                df["tx_px"] ** 2 + df["ty_px"] ** 2 + df["tz_px"] ** 2
            )

        cell_name = self.cell_names[self.current_index]
        if self.current_time is not None and "time" in self.ms.columns:
            cell_df = df[(self.ms["cell_name"] == cell_name)].copy()
            cell_sum = (
                cell_df.groupby("time")
                .agg(N=("tentacle_id", "nunique"), L=("tentacle_length", "mean"))
                .reset_index()
            )

            N = cell_sum["N"].values
            L = cell_sum["L"].values
            times = cell_sum["time"].values

            turbo = cm.get_cmap("turbo_r")
            norm = mcolors.Normalize(vmin=times.min(), vmax=times.max())
            base_colors = (turbo(norm(times))[:, :3] * 255).astype(int)
            # base_colors = base_colors[:,:,:,0.1]

            brushes = [pg.mkBrush(*c, 75) for c in base_colors]

            self.highlight_point.setData(x=N, y=L, brush=brushes, size=15)

            self.highlight_line.setData(x=N, y=L)

            cur_N = cell_sum[cell_sum["time"] == self.current_time]["N"].values
            cur_L = cell_sum[cell_sum["time"] == self.current_time]["L"].values

            self.highlight_point.addPoints(
                x=cur_N, y=cur_L, brush=pg.mkBrush(255, 0, 0, 200), size=20
            )

        else:
            row = self.summary[self.summary["cell_name"] == cell_name]
            if row.empty:
                return
            N = float(row["N"].iloc[0])
            L = float(row["L"].iloc[0])
            self.highlight_point.setData([N], [L])


# main_application


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = TentacleViewer()
    viewer.resize(1200, 700)
    viewer.show()
    sys.exit(app.exec_())
