from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from ui.IS_Form import UI_IS_Model
import numpy as np
from direct import acf_3_full
from parameters import KhISR, SpectrumParams


class SignalGenerationThread(QObject):
    finished = pyqtSignal(tuple)
    heightCompleted = pyqtSignal(tuple)

    def __init__(self, count, params, fcorr, fsig):
        super().__init__()
        self.height_count = count
        self.plasma_params = params
        self.fcorr = fcorr
        self.fsig = fsig

    def run(self):
        LAG = KhISR.LAG
        TAU_C = KhISR.TAU_C
        df = SpectrumParams.DF
        freq_count = SpectrumParams.NUMBER_OF_HARM
        count_1min = KhISR.COUNT_1min
        ti, te, g1, g2 = self.plasma_params
        acf_th, spectrum_th = acf_3_full(1, 4, 16, g1, g2, ti, te, 1, 0, LAG, TAU_C)
        harm_std = np.sqrt(spectrum_th)
        rng = np.random.default_rng()
        ampl = []
        freq_ind = np.arange(freq_count)
        for height in range(self.height_count):
            acf = np.zeros(LAG)
            sig = np.zeros((LAG, count_1min))
            for i in range(count_1min):
                phase = rng.uniform(0, 2 * np.pi, freq_count)
                ampl = np.array(list(map(lambda x: rng.rayleigh(x), harm_std)))
                add_cos = np.cos(2 * np.pi * freq_ind * df * TAU_C)
                add_sin = np.sin(2 * np.pi * freq_ind * df * TAU_C)
                sig_harm = np.empty((freq_count, LAG))
                sig_harm_sin = np.empty((freq_count, LAG))
                sig_harm[:, 0] = np.cos(phase)
                sig_harm_sin[:, 0] = np.sin(phase)
                sig[0, i] = np.sum(ampl * sig_harm[:, 0])
                for k in range(1, LAG):
                    sig_harm[:, k] = sig_harm[:, k - 1] * add_cos - sig_harm_sin[:, k - 1] * add_sin
                    sig_harm_sin[:, k] = sig_harm_sin[:, k - 1] * add_cos + sig_harm[:, k - 1] * add_sin
                    sig[k, i] = np.sum(ampl * sig_harm[:, k])
                for k in range(LAG):
                    acf[k] += sig[0, i] * sig[k, i]
            acf_norm = acf / acf[0]
            np.savetxt(self.fcorr, acf_norm, fmt='%1.8f', newline=' ')
            self.fcorr.write('\n')
            self.fcorr.flush()
            for k in range(LAG):
                np.savetxt(self.fsig[k], sig[k], fmt='%1.8f', newline=' ')
                self.fsig[k].write('\n')
                self.fsig[k].flush()
            self.heightCompleted.emit((height, ampl, acf_norm))
        self.finished.emit((harm_std, acf_th))


class NoiseGenerationThread(QObject):
    finished = pyqtSignal(tuple)
    heightCompleted = pyqtSignal(tuple)

    def __init__(self, count, params, noise_params, fcorr, fsig):
        super().__init__()
        self.height_count = count
        self.plasma_params = params
        self.noise_params = noise_params
        self.fcorr = fcorr
        self.fsig = fsig

    def run(self):
        LAG = KhISR.LAG
        TAU_C = KhISR.TAU_C
        df = SpectrumParams.DF
        freq_count = SpectrumParams.NUMBER_OF_HARM
        count_1min = KhISR.COUNT_1min
        ti, te, g1, g2 = self.plasma_params
        height_count, disp_n = self.noise_params
        noise_lag = LAG + height_count - 1
        acf_th, spectrum_th = acf_3_full(1, 4, 16, g1, g2, ti, te, 1, 0, LAG, TAU_C)
        sig_noise_harm_std = np.sqrt(np.array(spectrum_th) * KhISR.HEIGHT_NUMBER + disp_n)
        rng = np.random.default_rng()
        ampl = np.array(list(map(lambda x: rng.rayleigh(x), sig_noise_harm_std)))
        freq_ind = np.arange(freq_count)
        acf_noise = np.zeros(LAG)
        noise = np.zeros((noise_lag, count_1min))
        for i in range(count_1min):
            phase_noise = rng.uniform(0, 2 * np.pi, freq_count)
            ampl_noise = rng.rayleigh(disp_n, freq_count)
            add_cos = np.cos(2 * np.pi * freq_ind * df * TAU_C)
            add_sin = np.sin(2 * np.pi * freq_ind * df * TAU_C)
            noise_harm = np.empty((freq_count, noise_lag))
            noise_harm_sin = np.empty((freq_count, noise_lag))
            noise_harm[:, 0] = np.cos(phase_noise)
            noise_harm_sin[:, 0] = np.sin(phase_noise)
            noise[0, i] = np.sum(ampl_noise * noise_harm[:, 0])
            for k in range(1, noise_lag):
                noise_harm[:, k] = noise_harm[:, k - 1] * add_cos - noise_harm_sin[:, k - 1] * add_sin
                noise_harm_sin[:, k] = noise_harm_sin[:, k - 1] * add_cos + noise_harm[:, k - 1] * add_sin
                noise[k, i] = np.sum(ampl_noise * noise_harm[:, k])
            for k in range(LAG):
                acf_noise[k] += noise[0, i] * noise[k, i]
        acf_noise_norm = acf_noise / acf_noise[0]
        #     np.savetxt(self.fcorr, acf_norm, fmt='%1.8f', newline=' ')
        #     self.fcorr.write('\n')
        #     self.fcorr.flush()
        #     for k in range(LAG):
        #         np.savetxt(self.fsig[k], sig[k], fmt='%1.8f', newline=' ')
        #         self.fsig[k].write('\n')
        #         self.fsig[k].flush()
        #     self.heightCompleted.emit((height, ampl, acf_norm))
        self.heightCompleted.emit((ampl, acf_noise_norm))
        self.finished.emit((sig_noise_harm_std,))


class ISModelForm(QMainWindow, UI_IS_Model):
    def __init__(self):
        # parent initialisation
        super().__init__()

        # ui loading
        self.setupUi(self)

        # settings
        self.height_count_step1 = 0
        self.height_count_step2 = 0
        self.sig_disp = 0
        self.step1_mpl1.canvas.ax.set_title('Signal spectrum')
        self.step1_mpl1.canvas.ax.tick_params(direction='in')
        self.step1_mpl1.canvas.ax.set_xlim(left=0, right=15000)
        self.step1_mpl2.canvas.ax.set_title('ACF')
        self.step1_mpl2.canvas.ax.tick_params(direction='in')
        self.step1_mpl2.canvas.ax.set_xlim(left=0, right=19)
        self.step2_mpl1.canvas.ax.set_title('Signal and noise spectrum')
        self.step2_mpl1.canvas.ax.tick_params(direction='in')
        self.step2_mpl1.canvas.ax.set_xlim(left=0, right=15000)
        self.step2_mpl2.canvas.ax.set_title('ACF')
        self.step2_mpl2.canvas.ax.tick_params(direction='in')
        self.step2_mpl2.canvas.ax.set_xlim(left=0, right=19)
        self.progressBar.reset()
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setTextVisible(True)

        # thread init
        self.thread = None
        self.worker = None
        self.fsig = []

        # count of saved heights
        self.fcorr = open('data/1/1corr.dat', 'r')
        if self.fcorr:
            self.height_count_step1 = len(self.fcorr.readlines())
            self.label_15.setText(f'{self.height_count_step1} saved heights')
        self.fcorr.close()

        self.fcorr = open('data/2/2corr.dat', 'r')
        if self.fcorr:
            self.height_count_step2 = len(self.fcorr.readlines())
            self.label_16.setText(f'{self.height_count_step2} saved heights')
        self.fcorr.close()

        # connections
        self.pushButton.clicked.connect(self.start)

    def get_plasma_params(self):
        ion_temp = float(self.lineEdit.text())
        electron_temp = float(self.lineEdit_2.text())
        o_plus_fraction = float(self.lineEdit_3.text())
        h_plus_fraction = float(self.lineEdit_4.text())
        he_plus_fraction = 1 - o_plus_fraction - h_plus_fraction
        return (ion_temp, electron_temp,
                h_plus_fraction, he_plus_fraction)

    def get_height_count(self):
        return int(self.lineEdit_7.text())

    def get_signal_to_noise_ratio(self):
        return float(self.lineEdit_6.text())

    def start(self):
        height_count = self.get_height_count()
        plasma_params = self.get_plasma_params()
        if self.tabWidget.currentIndex() == 1:
            self.progressBar.setRange(0, height_count)
            self.progressBar.setFormat(f'Processing... (%v of {height_count})')
            self.progressBar.setValue(0)

            # file init
            if self.checkBox_2.isChecked():
                self.fcorr = open('data/1/1corr.dat', 'a')
            else:
                self.fcorr = open('data/1/1corr.dat', 'w')
            for i in range(KhISR.LAG):
                if self.checkBox_2.isChecked():
                    file = open(f'data/1/1sig{i}.dat', 'a')
                else:
                    file = open(f'data/1/1sig{i}.dat', 'w')
                self.fsig.append(file)

            self.thread = QThread()
            self.worker = SignalGenerationThread(height_count, plasma_params, self.fcorr, self.fsig)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.gen_complete)
            self.worker.heightCompleted.connect(self.height_complete)
            self.thread.start()
        elif self.tabWidget.currentIndex() == 2:
            snr = float(self.lineEdit_6.text())
            if self.sig_disp == 0:
                ti, te, g1, g2 = plasma_params
                _, spectrum = acf_3_full(1, 4, 16, g1, g2, ti, te, 1, 0,
                                         KhISR.LAG, KhISR.TAU_C)
                self.sig_disp = np.sum(spectrum)
            disp_noise = (KhISR.HEIGHT_NUMBER * self.sig_disp /
                          ((SpectrumParams.NUMBER_OF_HARM - 1) * snr))

            # file init
            if self.checkBox_2.isChecked():
                self.fcorr = open('data/2/2corr.dat', 'a')
            else:
                self.fcorr = open('data/2/2corr.dat', 'w')
            self.fsig = []
            for i in range(KhISR.LAG):
                file = open(f'data/1/1sig{i}.dat', 'r')
                self.fsig.append(file)
            if self.height_count_step1 < KhISR.HEIGHT_LIMIT:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Height count error")
                msg.setInformativeText(f'Height count generated at step 1 '
                                       f'can not be less than {KhISR.HEIGHT_LIMIT}')
                msg.setWindowTitle("Error")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            self.height_count_step2 = self.height_count_step1 - KhISR.HEIGHT_LIMIT + 1
            noise_params = (self.height_count_step2, disp_noise)
            self.thread = QThread()
            self.worker = NoiseGenerationThread(height_count, plasma_params,
                                                noise_params, self.fcorr, self.fsig)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.noise_gen_complete)
            self.worker.heightCompleted.connect(self.noise_height_complete)
            self.thread.start()

    def height_complete(self, params):
        height, ampl, acf = params
        LAG = KhISR.LAG
        df = SpectrumParams.DF
        freq_count = SpectrumParams.NUMBER_OF_HARM
        freq = [x * df for x in range(freq_count)]
        if height == 0:
            self.step1_mpl1.canvas.ax.plot(freq, ampl, c='r', lw=0.5)
            self.step1_mpl1.canvas.draw()

        ind = [j for j in range(LAG)]
        self.step1_mpl2.canvas.ax.plot(ind, acf, color='r', lw=0.5)
        self.step1_mpl2.canvas.draw()
        self.progressBar.setValue(height + 1)

    def gen_complete(self, params):
        harm_std, acf_th = params
        self.sig_disp = np.sum(harm_std * harm_std)
        LAG = KhISR.LAG
        df = SpectrumParams.DF
        freq_count = SpectrumParams.NUMBER_OF_HARM
        freq = [x * df for x in range(freq_count)]
        ind = [j for j in range(LAG)]
        self.step1_mpl1.canvas.ax.plot(freq, harm_std, c='b', lw=0.8)
        self.step1_mpl1.canvas.draw()
        self.step1_mpl2.canvas.ax.plot(ind, acf_th, c='b', lw=0.8)
        self.step1_mpl2.canvas.draw()
        self.progressBar.setFormat('Completed')
        self.fcorr.close()

    def noise_height_complete(self, params):
        df = SpectrumParams.DF
        LAG = KhISR.LAG
        freq_count = SpectrumParams.NUMBER_OF_HARM
        freq = [x * df for x in range(freq_count)]
        ind = [j for j in range(LAG)]
        noise_ampl, acf_noise = params
        self.step2_mpl1.canvas.ax.plot(freq, noise_ampl, c='r', lw=0.5)
        self.step2_mpl1.canvas.draw()
        self.step2_mpl2.canvas.ax.plot(ind, acf_noise, c='r', lw=0.5)
        self.step2_mpl2.canvas.draw()

    def noise_gen_complete(self, noise_harm_std):
        df = SpectrumParams.DF
        freq_count = SpectrumParams.NUMBER_OF_HARM
        freq = [x * df for x in range(freq_count)]
        harm_std, = noise_harm_std
        self.step2_mpl1.canvas.ax.plot(freq, harm_std, c='b', lw=0.5)
        self.step2_mpl1.canvas.draw()
