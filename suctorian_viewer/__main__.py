# suctorian_viewer/__main__.py
import sys
from PyQt5 import QtWidgets, QtGui
from .suctorian_viewer import TentacleViewer  # adjust if your class is in a different file

# importlib.resources for Python 3.7+
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources  # backport if needed

def main():
    # Create the Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Load the splash image from package data
    splash = None
    try:
        # use the context manager path() instead of files()
        with pkg_resources.path("suctorian_viewer", "splash.png") as splash_path:
            pixmap = QtGui.QPixmap(str(splash_path))
            splash = QtWidgets.QSplashScreen(pixmap)
            splash.show()
            app.processEvents()  # make splash appear immediately
    except FileNotFoundError:
        pass  # optional: proceed without splash if missing

    # Launch your main viewer
    viewer = TentacleViewer()
    viewer.resize(1200, 700)
    viewer.show()

    # Close splash once the main window is ready
    if splash:
        splash.finish(viewer)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
