from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal, QDir
from waiting import WaitingDialog
from inputcolor import InputColor
import sys
import os
sys.path.insert(0, os.getcwd())
import Ui_draw
from pathlib import Path
from typing import List
from operator import add, sub
import numpy as np
import cv2
import torch
from easydict import EasyDict
import yaml


class ImageFileList(object):
  def __init__(self, current_file: Path, file_list: List[Path]) -> None:
    self.list = file_list
    self.idx = file_list.index(current_file)
    self.size = len(self.list)

  def __len__(self):
    self.size

  def indexing(self, op, default_num) -> Path:
    self.idx = op(self.idx, 1)
    if self.idx < 0 or self.idx >= self.size:
      self.idx = default_num
    f = self.list[self.idx]
    return f

  def next(self) -> Path:
    return self.indexing(add, 0)

  def past(self) -> Path:
    return self.indexing(sub, self.size - 1)

  def curt(self) -> Path:
    return self.indexing(lambda a, b: a, self.idx)


def np2qimg(im_np: np.ndarray) -> QImage:
  h, w, channel = im_np.shape
  im = QImage(im_np.data, w, h, w * channel,
              QImage.Format_RGB888 if channel == 3 else QImage.Format_RGBA8888)
  return im


def qimg2np(im: QImage) -> np.ndarray:
  ptr = im.constBits()
  h, w = im.height(), im.width()
  ptr.setsize(h * w * 4)
  im_np = np.frombuffer(ptr, 'uint8').reshape((h, w, 4))
  return im_np


class AiMattingThread(QThread):
  finished = pyqtSignal(QPixmap)

  def setImage(self, im):
    self.im = im

  def setModel(self, model):
    self.model = model

  @staticmethod
  def ai_matting_mask(im: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """ use ai model get image mask

    Args:
        im (np.ndarray): image
        model (torch.nn.Module): network

    Returns:
        np.ndarray: mask (np.bool)
    """
    im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2BGR).astype('float32')
    im -= np.array((104.00699, 116.66877, 122.67892), 'float32')
    hw = np.array(im.shape[:2], dtype='uint32')
    # resize image to multiple of 32
    new_hw = (hw // 32) * 32
    im = cv2.resize(im, tuple(new_hw[::-1]))
    im = im.transpose((2, 0, 1))
    with torch.no_grad():
      ims = torch.autograd.Variable(torch.Tensor(im[None, ...]))
      preds = model(ims, mode=1)
      sigmod = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
      if (new_hw != hw).any():
        sigmod = cv2.resize(sigmod, tuple(hw[::-1]))
      mask = 255 * sigmod
    return mask > 1

  def run(self) -> None:
    mask = self.ai_matting_mask(self.im, self.model)
    h, w = self.im.shape[:2]
    mask_im = (mask[..., None] *
               np.tile(np.reshape(
                   np.array([0, 128, 0, 128], dtype='uint8'),
                   [1, 1, 4]), [h, w, 1]))
    mask_pixmap = QPixmap.fromImage(np2qimg(mask_im))
    self.finished.emit(mask_pixmap)


class MyMainWindow(QMainWindow):

  def setConfigDict(self, config: EasyDict):
    self.config = config

  def closeEvent(self, event) -> None:
    with open('./config.yml', 'w') as f:
      yaml.safe_dump(self.config.__dict__, f)

    super().closeEvent(event)


class Ui_DrawTask(Ui_draw.Ui_MainWindow):
  def setupCustom(self, MainWindow: MyMainWindow):
    """ global variable  """
    self.MainWindow = MainWindow
    self.input_list: ImageFileList = None
    self.output_dir: Path = None
    self.cur_path: Path = None
    self.export_path: Path = None
    self.cur_np_im: np.ndarray = None
    self.mask_pixmap: QPixmap = None
    with open('config.yml') as f:
      self.config = EasyDict(yaml.safe_load(f))
    self.MainWindow.setConfigDict(self.config)

    self.model = torch.load('./final-all.pth')
    self.model.to(torch.device('cpu'))
    self.model.eval()
    self.aimattingthread = AiMattingThread(None)
    self.aimattingthread.setModel(self.model)
    self.waitdialog = WaitingDialog(self.centralwidget)

    def _finshed(pixmap):
      # TODO 更改实现方式
      self.waitdialog.close()
      self.mask_pixmap = pixmap
      pix = QPixmap.fromImage(np2qimg(self.cur_np_im))
      self.draw_lb.setPixmap(pix, self.mask_pixmap)
      self.draw_lb.setDrawLabelState('enable')

    self.aimattingthread.finished.connect(_finshed)

  def setupSolt(self):
    self.set_state_bt()
    self.set_input_bt()
    self.set_output_bt()
    self.set_next_past_bt()
    self.set_pen_size_bt()
    self.set_export_bt()
    self.set_color_mb()

  @staticmethod
  def read_im(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    ch = im.shape[-1]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB if ch == 3 else cv2.COLOR_BGRA2RGBA)
    return im

  def set_color_mb(self) -> None:
    """连接menu bar与set color的操作
    """
    def _set_color():
      self.config.export_color = InputColor.getColors(self.centralwidget, self.config.export_color)
      self.MainWindow.setConfigDict(self.config)

    self.actionset_color.triggered.connect(_set_color)

  def set_draw_lb_background(self, img_path):
    self.cur_np_im = self.read_im(img_path)
    self.mask_pixmap = None
    if self.check_bx.isChecked():
      self.aimattingthread.setImage(self.cur_np_im)
      self.waitdialog.show()
      self.aimattingthread.start()
    else:
      pix = QPixmap.fromImage(np2qimg(self.cur_np_im))
      self.draw_lb.setPixmap(pix, self.mask_pixmap)
      self.draw_lb.setDrawLabelState('enable')

  def set_res_lb_background(self, img_path):
    np_im = self.read_im(img_path)
    self.resut_lb.setPixmap(QPixmap.fromImage(np2qimg(np_im)), None)

  def set_pen_size_bt(self):
    self.pen_size_sd.valueChanged['int'].connect(self.draw_lb.setPenSize)

  def set_state_bt(self):
    self.state_bt.setCheckable(True)

    def fc():
      if self.state_bt.isChecked():
        self.state_bt.setText("修补")
        self.draw_lb.setDrawLabelMode('clear')
      else:
        self.state_bt.setText("消除")
        self.draw_lb.setDrawLabelMode('line')

    self.state_bt.clicked.connect(fc)

  def change_matting_image(self):
    self.filename_lb.setText(self.cur_path.name)
    self.set_draw_lb_background(self.cur_path.as_posix())
    self.export_path = (self.output_dir / (self.cur_path.stem + '.png'))
    if self.export_path.exists():
      self.set_res_lb_background(self.export_path.as_posix())
    else:
      self.resut_lb.clearImage()

  def set_input_bt(self):
    def get_img_path():
      if self.output_dir == None:
        msgBox = QMessageBox()
        msgBox.setWindowTitle("提示")
        msgBox.setText("请先设置输出目录")
        msgBox.exec()
        return
      f_path, _ = QFileDialog.getOpenFileName(parent=None, caption="Select Image",
                                              filter="Images (*.jpg *.jpeg *.tif *.bmp *.png)",
                                              options=QFileDialog.ReadOnly)
      if f_path != '':
        f_path = Path(f_path)
        input_dir: Path = f_path.parent
        qdir = QDir(input_dir.as_posix())
        qdir.setNameFilters("*.jpg *.jpeg *.tif *.bmp *.png".split(' '))
        f_list = [input_dir / f.fileName() for f in qdir.entryInfoList()]
        self.input_list = ImageFileList(f_path, f_list)
        self.cur_path = self.input_list.curt()
        self.input_lb.setText(self.cur_path.parent.as_posix())
        # 设置进度条
        self.file_pb.setRange(0, self.input_list.size)
        self.file_pb.setValue(self.input_list.idx)
        self.change_matting_image()

      else:
        self.input_lb.setText("请选择输入文件")

    self.input_bt.clicked.connect(get_img_path)

  def set_output_bt(self):
    def get_out_path():
      d_path = QFileDialog.getExistingDirectory(parent=None, caption="Select Directory",
                                                options=QFileDialog.ReadOnly | QFileDialog.ShowDirsOnly)
      d_path = Path(d_path)
      self.output_dir = d_path
      self.output_lb.setText(self.output_dir.as_posix())

    self.output_bt.clicked.connect(get_out_path)

  def set_next_past_bt(self):
    def change_label(mode):
      if self.input_list != None and self.output_dir != None:
        self.cur_path = (self.input_list.next()
                         if mode == 'sub' else self.input_list.past())
        self.file_pb.setValue(self.input_list.idx)  # 进度条设置
        self.change_matting_image()

    self.next_bt.clicked.connect(lambda: change_label('add'))
    self.past_bt.clicked.connect(lambda: change_label('sub'))

  def set_export_bt(self):
    def export_im():
      if self.output_dir != None and self.cur_path != None:
        drawed_im = qimg2np(self.draw_lb.exportMaskImage())
        # the region need masked value is 1, other is 0
        mask = np.expand_dims(drawed_im[..., 1] > 0, -1)

        valid_part = self.cur_np_im.copy() * np.logical_not(mask)
        rgb = np.reshape(np.array(self.config.export_color[:3], 'uint8'), [1, 1, 3])
        alpha = np.reshape(np.array(self.config.export_color[3:4], 'uint8'), [1, 1, 1])
        invalid_part = np.tile(rgb,
                               list(self.cur_np_im.shape[:2]) + [1]) * mask
        valid_alpha = np.tile(np.array([[[255]]], 'uint8'),
                              list(self.cur_np_im.shape[:2]) + [1]) * np.logical_not(mask)
        invalid_alpha = np.tile(alpha,
                                list(self.cur_np_im.shape[:2]) + [1]) * mask

        self.cur_np_im_masked = valid_part + invalid_part
        alpha_masked = valid_alpha + invalid_alpha
        cv2.imwrite(self.export_path.as_posix(),
                    np.concatenate((cv2.cvtColor(self.cur_np_im_masked, cv2.COLOR_RGB2BGR),
                                    alpha_masked), -1))
        self.set_res_lb_background(self.export_path.as_posix())
        # print(f'export success:{export_path}')
      else:
        msgBox = QMessageBox()
        msgBox.setWindowTitle("提示")
        msgBox.setText("请先设置输入目录与输出目录")
        msgBox.exec()

    self.export_bt.clicked.connect(export_im)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  MainWindow = MyMainWindow()
  ui = Ui_DrawTask()
  ui.setupUi(MainWindow)
  ui.setupCustom(MainWindow)
  ui.setupSolt()
  MainWindow.show()
  sys.exit(app.exec_())
