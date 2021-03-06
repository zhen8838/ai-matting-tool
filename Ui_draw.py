# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/zqh/workspace/qt-matting/draw.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.input_bt = QtWidgets.QPushButton(self.centralwidget)
        self.input_bt.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_bt.sizePolicy().hasHeightForWidth())
        self.input_bt.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.input_bt.setFont(font)
        self.input_bt.setObjectName("input_bt")
        self.horizontalLayout.addWidget(self.input_bt)
        self.output_bt = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.output_bt.setFont(font)
        self.output_bt.setObjectName("output_bt")
        self.horizontalLayout.addWidget(self.output_bt)
        self.state_bt = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.state_bt.setFont(font)
        self.state_bt.setObjectName("state_bt")
        self.horizontalLayout.addWidget(self.state_bt)
        self.export_bt = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.export_bt.setFont(font)
        self.export_bt.setObjectName("export_bt")
        self.horizontalLayout.addWidget(self.export_bt)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.pen_size_sd = QtWidgets.QSlider(self.centralwidget)
        self.pen_size_sd.setMinimumSize(QtCore.QSize(0, 22))
        self.pen_size_sd.setProperty("value", 20)
        self.pen_size_sd.setOrientation(QtCore.Qt.Horizontal)
        self.pen_size_sd.setObjectName("pen_size_sd")
        self.horizontalLayout.addWidget(self.pen_size_sd)
        self.pen_size_lb = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pen_size_lb.sizePolicy().hasHeightForWidth())
        self.pen_size_lb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.pen_size_lb.setFont(font)
        self.pen_size_lb.setObjectName("pen_size_lb")
        self.horizontalLayout.addWidget(self.pen_size_lb)
        self.file_pb = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_pb.sizePolicy().hasHeightForWidth())
        self.file_pb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.file_pb.setFont(font)
        self.file_pb.setProperty("value", 0)
        self.file_pb.setObjectName("file_pb")
        self.horizontalLayout.addWidget(self.file_pb)
        self.check_bx = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.check_bx.setFont(font)
        self.check_bx.setTabletTracking(False)
        self.check_bx.setChecked(True)
        self.check_bx.setObjectName("check_bx")
        self.horizontalLayout.addWidget(self.check_bx)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.next_bt = QtWidgets.QPushButton(self.centralwidget)
        self.next_bt.setMinimumSize(QtCore.QSize(0, 30))
        self.next_bt.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.next_bt.setFont(font)
        self.next_bt.setObjectName("next_bt")
        self.horizontalLayout_2.addWidget(self.next_bt)
        self.filename_lb = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename_lb.sizePolicy().hasHeightForWidth())
        self.filename_lb.setSizePolicy(sizePolicy)
        self.filename_lb.setText("")
        self.filename_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.filename_lb.setObjectName("filename_lb")
        self.horizontalLayout_2.addWidget(self.filename_lb)
        self.past_bt = QtWidgets.QPushButton(self.centralwidget)
        self.past_bt.setMinimumSize(QtCore.QSize(0, 30))
        self.past_bt.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.past_bt.setFont(font)
        self.past_bt.setObjectName("past_bt")
        self.horizontalLayout_2.addWidget(self.past_bt)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.draw_lb = ScollWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.draw_lb.sizePolicy().hasHeightForWidth())
        self.draw_lb.setSizePolicy(sizePolicy)
        self.draw_lb.setMinimumSize(QtCore.QSize(320, 320))
        self.draw_lb.setObjectName("draw_lb")
        self.horizontalLayout_4.addWidget(self.draw_lb)
        self.resut_lb = ScollWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resut_lb.sizePolicy().hasHeightForWidth())
        self.resut_lb.setSizePolicy(sizePolicy)
        self.resut_lb.setMinimumSize(QtCore.QSize(320, 320))
        self.resut_lb.setObjectName("resut_lb")
        self.horizontalLayout_4.addWidget(self.resut_lb)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.input_lb = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_lb.sizePolicy().hasHeightForWidth())
        self.input_lb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.input_lb.setFont(font)
        self.input_lb.setObjectName("input_lb")
        self.horizontalLayout_3.addWidget(self.input_lb)
        self.output_lb = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_lb.sizePolicy().hasHeightForWidth())
        self.output_lb.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(12)
        self.output_lb.setFont(font)
        self.output_lb.setObjectName("output_lb")
        self.horizontalLayout_3.addWidget(self.output_lb)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 22))
        self.menubar.setObjectName("menubar")
        self.menumenu = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.menumenu.setFont(font)
        self.menumenu.setObjectName("menumenu")
        MainWindow.setMenuBar(self.menubar)
        self.actionset_color = QtWidgets.QAction(MainWindow)
        self.actionset_color.setObjectName("actionset_color")
        self.menumenu.addAction(self.actionset_color)
        self.menubar.addAction(self.menumenu.menuAction())

        self.retranslateUi(MainWindow)
        self.pen_size_sd.valueChanged['int'].connect(self.pen_size_lb.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI Matting Tool"))
        self.input_bt.setText(_translate("MainWindow", "输入路径"))
        self.output_bt.setText(_translate("MainWindow", "输出路径"))
        self.state_bt.setText(_translate("MainWindow", "消除"))
        self.export_bt.setText(_translate("MainWindow", "导出"))
        self.label.setText(_translate("MainWindow", "画笔大小"))
        self.pen_size_lb.setText(_translate("MainWindow", "20"))
        self.check_bx.setText(_translate("MainWindow", "AUTO"))
        self.next_bt.setText(_translate("MainWindow", "上一张"))
        self.past_bt.setText(_translate("MainWindow", "下一张"))
        self.input_lb.setText(_translate("MainWindow", "请选择输入文件"))
        self.output_lb.setText(_translate("MainWindow", "请选择输出文件夹"))
        self.menumenu.setTitle(_translate("MainWindow", "菜单"))
        self.actionset_color.setText(_translate("MainWindow", "颜色设置"))

from scollwidget import ScollWidget
