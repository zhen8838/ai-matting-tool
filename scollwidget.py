from PyQt5.QtWidgets import QWidget, QLabel, QScrollArea, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QBrush, QColor, QPalette, QMouseEvent, QGuiApplication, QPaintEvent, QResizeEvent, QWheelEvent
from PyQt5.QtCore import QPoint, Qt
from enum import Enum
from typing import Union


class PaintPath(object):
  def __init__(self) -> None:
    self.l = []

  def lineTo(self, pos: QPoint):
    self.l.append(pos)

  def moveTo(self, pos: QPoint):
    self.l.clear()
    self.l.append(pos)

  def drawPath(self, painter: QPainter):
    for i in range(len(self.l) - 1):
      painter.drawLine(self.l[i], self.l[i + 1])

  def clear(self):
    self.l.clear()


class DrawLabel(QLabel):
  class State(Enum):
    disable = 1
    enable = 2

  class DrawMode(Enum):
    line = 1
    clear = 2

  def __init__(self, parent) -> None:
    super().__init__(parent)
    self.begin_point = QPoint()
    self.end_point = QPoint()
    self.drawmode = self.DrawMode.line
    self.state = self.State.disable
    self.currn_pix = None
    self.trans_pix = None
    self.pen = QPen()
    self.pen.setCapStyle(Qt.RoundCap)
    self.pen.setWidth(20)
    self.pen.setColor(QColor(0, 128, 0, 1))
    self.pen.setStyle(Qt.SolidLine)
    self.drawingPath = PaintPath()

  def setPixmap(self, pixmap: QPixmap, mask_pixmap: QPixmap = None) -> None:
    self.currn_pix = pixmap
    self.trans_pix = pixmap.copy()
    self.trans_pix.fill(Qt.transparent)
    if mask_pixmap != None:
      mask = mask_pixmap.mask()
      sz = mask.size()
      # using mask and SourceOut_mode cutout the target region. 
      self.trans_pix.fill(QColor(0, 255, 0, 128))
      painter = QPainter(self.trans_pix)
      painter.setCompositionMode(QPainter.CompositionMode_SourceOut)
      painter.drawPixmap(0, 0, sz.width(), sz.height(), mask)
      painter.end()
    self.drawingPath.clear()
    super().setPixmap(self.currn_pix)
    self.adjustSize()

  def setPenSize(self, size: int):
    self.pen.setWidth(size)

  def paintEvent(self, event: QPaintEvent) -> None:
    sz = self.size()
    if self.state == self.State.disable:
      painter = QPainter()
      painter.begin(self)
      painter.drawPixmap(0, 0, sz.width(), sz.height(), self.currn_pix)
      painter.drawPixmap(0, 0, sz.width(), sz.height(), self.trans_pix)
      painter.end()
    else:
      painter = QPainter()
      painter.begin(self.trans_pix)
      if self.drawmode == self.DrawMode.line:
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
      elif self.drawmode == self.DrawMode.clear:
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
      painter.setPen(self.pen)
      self.drawingPath.drawPath(painter)
      painter.end()
      painter2 = QPainter()
      painter2.begin(self)
      painter2.drawPixmap(0, 0, sz.width(), sz.height(), self.currn_pix)
      painter2.drawPixmap(0, 0, sz.width(), sz.height(), self.trans_pix)
      painter2.end()


class ScollWidget(QWidget):
  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    self.scaleFactor = 1.
    self.begin_point = QPoint()
    self.end_point = QPoint()
    self.imageLabel = DrawLabel(self)
    self.scrollArea = QScrollArea(self)

    self.imageLabel.setBackgroundRole(QPalette.Dark)
    self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    self.imageLabel.setScaledContents(True)

    self.scrollArea.setBackgroundRole(QPalette.Dark)
    self.scrollArea.setWidget(self.imageLabel)
    self.scrollArea.setVisible(False)

    self.shbar = self.scrollArea.horizontalScrollBar()
    self.svbar = self.scrollArea.verticalScrollBar()

  def setPixmap(self, pixmap, mask_pixmap):
    self.imageLabel.setPixmap(pixmap, mask_pixmap)
    self.scaleFactor = 1.
    self.scrollArea.setVisible(True)

  def setPenSize(self, size: int):
    self.imageLabel.pen.setWidth(size)

  def setDrawLabelState(self, state: DrawLabel.State):
    self.imageLabel.state = self.imageLabel.State[state]

  def setDrawLabelMode(self, mode: DrawLabel.DrawMode):
    self.imageLabel.drawmode = self.imageLabel.DrawMode[mode]

  def clearImage(self):
    self.imageLabel.clear()
    self.scrollArea.setVisible(False)

  def exportMaskImage(self):
    return self.imageLabel.trans_pix.toImage()

  def zoomIn(self, factor=0.9):
    self.scaleImage(self.imageLabel, factor)
    self.scaleScrollBar(factor)

  def zoomOut(self, factor=1.1):
    self.scaleImage(self.imageLabel, factor)
    self.scaleScrollBar(factor)

  def scaleImage(self, imageLabel: DrawLabel, factor: float):
    if (self.scaleFactor * factor) > 0.3 and (self.scaleFactor * factor) < 3:
      self.scaleFactor *= factor
      imageLabel.resize(self.scaleFactor * imageLabel.pixmap().size())
      scaleFactor = self.scaleFactor

  def scaleScrollBar(self, factor):
    self.adjustScrollBar(self.shbar, factor)
    self.adjustScrollBar(self.svbar, factor)

  def adjustScrollBar(self, scrollBar, factor):
    scrollBar.setValue(int(factor * scrollBar.value()
                           + ((factor - 1) * scrollBar.pageStep() / 2)))

  def posOffset(self, event: QMouseEvent):
    return event.pos() + QPoint(self.shbar.value(), self.svbar.value()) / self.scaleFactor

  def mouseMoveEvent(self, event: QMouseEvent) -> None:
    if event.buttons() == Qt.LeftButton:
      curpos = self.posOffset(event)
      self.imageLabel.drawingPath.lineTo(curpos / self.scaleFactor)
      self.imageLabel.update()

    elif event.buttons() == Qt.RightButton:
      self.end_point = event.pos()
      delta: QPoint = (self.end_point - self.begin_point)
      self.begin_point = self.end_point
      y = self.svbar.value()
      x = self.shbar.value()
      self.svbar.setValue(y - delta.y())
      self.shbar.setValue(x - delta.x())

  def mousePressEvent(self, event: QMouseEvent) -> None:
    if event.button() == Qt.LeftButton:
      curpos = self.posOffset(event)
      self.imageLabel.drawingPath.moveTo(curpos)

    elif event.button() == Qt.RightButton:
      self.begin_point = event.pos()

  def mouseReleaseEvent(self, event: QMouseEvent) -> None:
    if event.button() == Qt.LeftButton:
      curpos = self.posOffset(event)
      self.imageLabel.drawingPath.lineTo(curpos)
      self.imageLabel.update()
      pass

  def wheelEvent(self, event: QWheelEvent) -> None:
    if QApplication.keyboardModifiers() == Qt.ControlModifier:
      delta = event.angleDelta().y()
      if delta > 0:
        self.zoomIn()
      elif delta < 0:
        self.zoomOut()

  def resizeEvent(self, event: QResizeEvent) -> None:
    self.scrollArea.resize(self.size())
