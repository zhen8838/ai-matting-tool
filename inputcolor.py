from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QInputDialog, QLabel, QLineEdit, QPushButton, QWidget, QFormLayout
from PyQt5.QtGui import QIntValidator
from typing import List
import sys


class InputColor(QDialog):
  def __init__(self, parent, old_color: List[int]) -> None:
    super().__init__(parent)
    assert len(old_color) == 4
    self.old_color = old_color
    layout = QFormLayout(self)
    self.label = QLabel(self)
    self.label.setAlignment(Qt.AlignCenter)
    self.label.setText('请输入0~255间的数字')
    layout.addRow(self.label)

    self.fields: List[QLineEdit] = []
    for name, value in zip(['r', 'g', 'b', 'alpha'], self.old_color):
      valiator = QIntValidator(0, 255, self)
      editer = QLineEdit(parent)
      editer.setValidator(valiator)
      editer.setText(str(value))
      layout.addRow(name, editer)
      self.fields.append(editer)

    buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
    layout.addWidget(buttonBox)
    buttonBox.accepted.connect(self.accept)
    buttonBox.rejected.connect(self.reject)
    self.setLayout(layout)

  @staticmethod
  def getColors(parent, old_color: List[int]) -> List[int]:
    dialog = InputColor(parent, old_color)
    res = dialog.exec()
    dialog.deleteLater()
    color = dialog.new_color if res else dialog.old_color
    return color

  def accept(self) -> None:
    accept = True
    try:
      self.new_color = []
      for field in self.fields:
        self.new_color.append(int(field.text()))
    except ValueError as e:
      self.label.setText('输入了无效数字')
      accept = False
    if accept:
      super().accept()


class inputdialogdemo(QWidget):
  def __init__(self, parent=None):
    super(inputdialogdemo, self).__init__(parent)

    layout = QFormLayout()
    self.btn = QPushButton("Choose from list")
    self.btn.clicked.connect(self.getItem)
    self.le = QLineEdit()
    layout.addRow(self.btn, self.le)

    self.btn1 = QPushButton("get name")
    self.btn1.clicked.connect(self.gettext)
    self.le1 = QLineEdit()
    layout.addRow(self.btn1, self.le1)

    self.btn2 = QPushButton("Enter an integer")
    self.btn2.clicked.connect(self.getint)
    self.le2 = QLineEdit()
    layout.addRow(self.btn2, self.le2)

    self.btn3 = QPushButton("new")
    self.btn3.clicked.connect(self.getcolor)
    self.le3 = QLineEdit()
    layout.addRow(self.btn3, self.le3)

    self.setLayout(layout)
    self.setWindowTitle("Input Dialog demo")

  def getItem(self):
    items = ("C", "C++", "Java", "Python")

    item, ok = QInputDialog.getItem(self, "select input dialog",
                                    "list of languages", items, 0, False)

    if ok and item:
      self.le.setText(item)

  def gettext(self):
    text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your name:')

    if ok:
      self.le1.setText(str(text))

  def getint(self):
    num, ok = QInputDialog.getInt(self, "integer input dualog", "enter a number")

    if ok:
      self.le2.setText(str(num))

  def getcolor(self):
    rgbres = InputColor.getColors(self, [12, 34, 25, 128])
    print(rgbres)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = inputdialogdemo()
  ex.show()
  sys.exit(app.exec_())
