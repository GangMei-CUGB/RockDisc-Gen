import os
import sys
import warnings
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QFileDialog, QTableWidget, \
    QTableWidgetItem
import win
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QSize
from base_method.boxplot.circular import plot_circular_structural_planes
from base_method.Copula import simulate_copula_data
from base_method.DDPM import run_diffusion_training
from base_method.GAN import train_gan
from base_method.Montecarlo import simulate_monte_carlo_data
warnings.filterwarnings('ignore')


class window(QMainWindow, win.Ui_mainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.setupUi(self)
        self.init_ui()
        self.work_path = None
        self.data_path = None
        self.image_paths = []
        self.pushButton_2.clicked.connect(self.select_data)
        self.pushButton_3.clicked.connect(self.select_wdir)
        self.pushButton.clicked.connect(self.main_fun)

    def main_fun(self):
        if self.work_path and self.data_path:
            if self.comboBox.currentIndex() == 0:
                simulate_copula_data(self.data_path, self.work_path)
                self.load_images(self.work_path)
            elif self.comboBox.currentIndex() == 1:
                run_diffusion_training(input_csv=self.data_path,output_dir=self.work_path)
                self.load_images(self.work_path)
            elif self.comboBox.currentIndex() == 2:
                train_gan(data_file=self.data_path,save_directory=self.work_path)
                self.load_images(self.work_path)
            elif self.comboBox.currentIndex() == 3:
                simulate_monte_carlo_data(self.data_path, self.work_path)
                self.load_images(self.work_path)
        else:
            QMessageBox.information(self, "Warning", "请先选择数据和工作目录")
            return

    def select_data(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, "选择输入数据", ".", "CSV Files (*.csv)")
        if csv_path:
            self.data_path = csv_path
            if self.work_path and self.data_path:
                plot_circular_structural_planes(self.data_path, self.work_path)
                self.load_images(self.work_path)

    def select_wdir(self):
        work_path = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if work_path:
            self.work_path = work_path
            if self.work_path and self.data_path:
                plot_circular_structural_planes(self.data_path, self.work_path)
                self.load_images(self.work_path)

    def init_ui(self):
        self.m_Position = None
        self.m_flag = False
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['文件名'])
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.cellClicked.connect(self.show_image)

    def load_images(self, directory):
        self.image_paths.clear()
        self.tableWidget.setRowCount(0)
        supported_types = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        all_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in supported_types
        ]
        all_files.sort(key=lambda x: os.path.getctime(x))
        for idx, file_path in enumerate(all_files):
            self.image_paths.append(file_path)
            file_name = os.path.basename(file_path)
            self.tableWidget.insertRow(idx)
            item = QTableWidgetItem(file_name)
            self.tableWidget.setItem(idx, 0, item)
        self.tableWidget.resizeColumnsToContents()

    def show_image(self, row, column):
        image_path = self.image_paths[row]

        pixmap = QPixmap(image_path)
        label_size = self.fatigue_image_label.size()
        pixmap_size = pixmap.size()
        scale_factor = min(label_size.width() / pixmap_size.width(),
                           label_size.height() / pixmap_size.height())
        new_width = int(pixmap_size.width() * scale_factor)
        new_height = int(pixmap_size.height() * scale_factor)
        scaled_pixmap = pixmap.scaled(new_width, new_height,
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.fatigue_image_label.setAlignment(Qt.AlignCenter)
        self.fatigue_image_label.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    def max_or_restore(self):
        if self.maxButton.isChecked():
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/img/icon/取消全屏_o.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.maxButton.setIcon(icon1)
            self.maxButton.setIconSize(QSize(20, 20))
            self.showMaximized()
        else:
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/img/icon/全屏 (1).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.maxButton.setIcon(icon1)
            self.maxButton.setIconSize(QSize(16, 16))
            self.showNormal()


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec_())
