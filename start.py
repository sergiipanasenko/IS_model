from PyQt5 import QtWidgets
from IS_Model import ISModelForm


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ISModelForm()
    MainWindow.show()
    sys.exit(app.exec_())