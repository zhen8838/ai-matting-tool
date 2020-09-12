from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QDialog, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPainter, QPixmap, QPaintEvent, QMovie
from PyQt5.QtCore import Qt, QTimer, QThread, QEventLoop, pyqtSignal, QObject
import sys
import os
import time


class AnimeWaiting(QLabel):

  def __init__(self, parent) -> None:
    super().__init__(parent)

    self.setScaledContents(True)

    self.bgpix = QPixmap('等待.svg')
    self.setPixmap(self.bgpix)

    timer = QTimer(self)
    timer.timeout.connect(self.update_func)
    timer.start(10)

    self.rotate_angle = 0

  def update_func(self):
    self.rotate_angle += 1
    if self.rotate_angle == 360:
      self.rotate_angle = 0
    self.update()

  def paintEvent(self, event: QPaintEvent) -> None:
    painter = QPainter(self)
    sz = self.bgpix.size()
    w, h = sz.width(), sz.height()
    painter.translate(w / 2, h / 2)
    painter.rotate(self.rotate_angle)
    painter.translate(-w / 2, -h / 2)
    painter.drawPixmap(0, 0, w, h, self.bgpix)


class WaitingDialog(QDialog):
  def __init__(self, parent) -> None:
    super().__init__(parent)
    self.verticalLayout = QVBoxLayout()
    self.verticalLayout.setObjectName("verticalLayout")

    self.anime_lb = QLabel(self)

    self.verticalLayout.addWidget(self.anime_lb)
    self.text_lb = QLabel(self)
    self.text_lb.setScaledContents(True)
    self.text_lb.setText("Waiting...")
    self.text_lb.setAlignment(Qt.AlignCenter)
    self.verticalLayout.addWidget(self.text_lb)

    self.gridLayout = QGridLayout(self)
    self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

    self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    self.setWindowModality(Qt.ApplicationModal)
    # self.setWindowOpacity(1)

    self.anime_lb.setAlignment(Qt.AlignCenter)
    self.bg_mv = QMovie('asset/等待.gif')
    self.anime_lb.setMovie(self.bg_mv)
    self.bg_mv.start()


class MyThread(QThread):
  sig = pyqtSignal()

  def run(self) -> None:
    time.sleep(5)
    self.sig.emit()


if __name__ == "__main__":
  app = QApplication(sys.argv)
  ui = QMainWindow()
  imgview = WaitingDialog(ui)
  imgview.show()

  def pp():
    print('end')
    imgview.close()

  t = MyThread()
  t.sig.connect(pp)
  t.start()

  # imgview.close()

  sys.exit(app.exec_())
