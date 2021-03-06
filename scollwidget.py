from PyQt5.QtWidgets import QWidget, QLabel, QScrollArea, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import QPainter, QPainterPath, QPixmap, QImage, QPen, QBrush, QColor, QPalette, QMouseEvent, QGuiApplication, QPaintEvent, QResizeEvent, QWheelEvent
from PyQt5.QtCore import QPoint, QRect, Qt
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
    self.cur_point = QPoint()
    self.mousePressed = False
    self.drawmode = self.DrawMode.line
    self.state = self.State.disable
    self.currn_pix = None
    self.trans_pix = None
    self.pen = QPen()
    self.pen.setCapStyle(Qt.RoundCap)
    self.pen_size = 20
    self.scaleFactor = 1.
    self.pen.setWidth(self.pen_size / self.scaleFactor)
    self.pen.setColor(QColor(0, 128, 0, 1))
    self.pen.setStyle(Qt.SolidLine)
    self.drawingPath = PaintPath()
    """ 设置背景图 """
    self.m_tile = QPixmap(128, 128)
    self.m_tile.fill(Qt.white)
    pt = QPainter(self.m_tile)
    color = QColor(230, 230, 230)
    pt.fillRect(0, 0, 64, 64, color)
    pt.fillRect(64, 64, 64, 64, color)
    pt.end()

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
      # NOTE 不知为何SourceOut后存在暗色区域，再利用Clear消除
      painter.setCompositionMode(QPainter.CompositionMode_Clear)
      painter.drawPixmap(0, 0, sz.width(), sz.height(), mask)
      painter.end()
    self.drawingPath.clear()
    super().setPixmap(self.currn_pix)
    self.adjustSize()

  def setPenSize(self, size: int):
    self.pen_size = size

  def setScaleFactor(self, scaleFactor: float):
    self.scaleFactor = scaleFactor

  def paintEvent(self, event: QPaintEvent) -> None:
    sz = self.size()
    pen_width = int(self.pen_size / self.scaleFactor)
    self.pen.setWidth(pen_width)
    """ draw backgroud """
    static_image = QPixmap(sz)
    bpainter = QPainter()
    bpainter.begin(static_image)
    o = 10
    bg = self.palette().brush(QPalette.Window)
    bpainter.fillRect(0, 0, o, o, bg)
    bpainter.fillRect(self.width() - o, 0, o, o, bg)
    bpainter.fillRect(0, self.height() - o, o, o, bg)
    bpainter.fillRect(self.width() - o, self.height() - o, o, o, bg)
    bpainter.setClipRect(self.rect())
    bpainter.setRenderHint(QPainter.Antialiasing)
    bpainter.drawTiledPixmap(self.rect(), self.m_tile)
    bpainter.end()
    bpainter.begin(self)
    bpainter.drawPixmap(self.rect(), static_image)

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
      if self.mousePressed:
        painter2.setPen(QPen(Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(255, 255, 255, 0), Qt.SolidPattern))
        painter2.setRenderHint(QPainter.Antialiasing, True)
        painter2.drawEllipse(self.cur_point,
                             self.pen_size // 2,
                             self.pen_size // 2)
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
    self.imageLabel.setPenSize(size)

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

  def scaleScrollBar(self, factor):
    self.adjustScrollBar(self.shbar, factor)
    self.adjustScrollBar(self.svbar, factor)

  def adjustScrollBar(self, scrollBar, factor):
    scrollBar.setValue(int(factor * scrollBar.value()
                           + ((factor - 1) * scrollBar.pageStep() / 2)))

  def posOffset(self, event: QMouseEvent):
    pos = (event.pos() + QPoint(self.shbar.value(), self.svbar.value())) / self.scaleFactor
    return pos

  def mouseMoveEvent(self, event: QMouseEvent) -> None:
    if event.buttons() == Qt.LeftButton:
      curpos = self.posOffset(event)
      self.imageLabel.drawingPath.lineTo(curpos)
      self.imageLabel.cur_point = event.pos() + QPoint(self.shbar.value(), self.svbar.value())
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
      self.imageLabel.mousePressed = True

    elif event.button() == Qt.RightButton:
      self.begin_point = event.pos()

  def mouseReleaseEvent(self, event: QMouseEvent) -> None:
    if event.button() == Qt.LeftButton:
      curpos = self.posOffset(event)
      self.imageLabel.drawingPath.lineTo(curpos)
      self.imageLabel.mousePressed = False
      self.imageLabel.update()
      pass

  def wheelEvent(self, event: QWheelEvent) -> None:
    if QApplication.keyboardModifiers() == Qt.ControlModifier:
      delta = event.angleDelta().y()
      if delta > 0:
        self.zoomIn()
      elif delta < 0:
        self.zoomOut()
      self.imageLabel.setScaleFactor(self.scaleFactor)

  def resizeEvent(self, event: QResizeEvent) -> None:
    self.scrollArea.resize(self.size())
