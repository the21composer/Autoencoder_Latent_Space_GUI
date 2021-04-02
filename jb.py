import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.datasets import fashion_mnist
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont
import os
import sys
import time


class AutoEncoder:
    def __init__(self):
        self.coded_size = 2
        self.ae, self.encoder, self.decoder = None, None, None
        # loading data from dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 784)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 784)
        self.encoder_res = []
        # loading model and prepare data if model exists
        if "model.h5" in os.listdir():
            self.ae = load_model('model.h5')
            self.encoder = load_model('encoder.h5')
            self.decoder = load_model('decoder.h5')
            self.encoded_test = self.encoder.predict(self.x_test)
            self.decoded_test = self.decoder.predict(self.encoded_test)
            self.encoder_res = np.load("encoder_res.npy")

    def make_model(self):
        # initializing encoder
        input_img = Input(shape=(784,))
        encoded = Dense(512, activation='relu')(input_img)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(3, activation='linear')(encoded)
        self.encoder = Model(input_img, encoded)

        # initializing decoder
        encoded_input = Input(shape=(3,))
        decoded = Dense(128, activation='relu')(encoded_input)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)
        self.decoder = Model(encoded_input, decoded)

        # initializing full model
        self.encoder_res = []
        full = self.decoder(self.encoder(input_img))
        self.ae = Model(input_img, full)
        self.ae.compile(optimizer="adam", loss='mean_squared_error')
        for i in range(10):
            self.ae.fit(self.x_train, self.x_train,
                        epochs=1,
                        batch_size=256,
                        validation_data=(self.x_test, self.x_test))
            self.encoder_res.append(self.encoder.predict(self.x_test))
        self.ae.save('model.h5')
        self.encoder.save('encoder.h5')
        self.decoder.save('decoder.h5')
        self.encoder_res = np.array(self.encoder_res)
        np.save("encoder_res", self.encoder_res)

    def get_encoded(self):
        return self.encoded_test

    def get_decoded(self):
        return self.decoded_test

    def get_decoder(self):
        return self.decoder

    def get_model(self):
        return self.ae


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle('Задание для стажировки, Облизанов Александр')
        self.initGUI()
        self.auto_encoder = AutoEncoder()
        if self.auto_encoder.get_model() is None:
            self.auto_encoder.make_model()

    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        central_widget.setMinimumSize(500, 200)
        self.setCentralWidget(central_widget)
        gui_layout = QtWidgets.QVBoxLayout()
        main_label = QtWidgets.QLabel("Latent space visualizer")
        main_label.setFont(QFont('Arial', 12))
        main_label.setMaximumSize(450, 50)
        second_label = QtWidgets.QLabel("Model type: Autoencoder. Dataset: fashion_mnist. Have fun! v.1.0")
        second_label.setFont(QFont('Arial', 9))
        main_label.setMaximumSize(450, 50)
        status = QtWidgets.QLabel("")
        latent_space_button = QtWidgets.QPushButton("Latent space (Plotly)")
        latent_space_button.clicked.connect(lambda val: self.latent_space_epochs())
        latent_space_button.setMaximumSize(160, 80)
        latent_space_button.setFont(QFont('Arial', 9))
        latent_space_pyplot_button = QtWidgets.QPushButton("Latent space (Pyplot)")
        latent_space_pyplot_button.clicked.connect(lambda val: self.latent_space_pyplot())
        latent_space_pyplot_button.setMaximumSize(160, 80)
        latent_space_pyplot_button.setFont(QFont('Arial', 9))
        make_model_button = QtWidgets.QPushButton("Make model")
        make_model_button.clicked.connect(lambda val: self.make_model(status))
        make_model_button.setMaximumSize(160, 80)
        make_model_button.setFont(QFont('Arial', 9))
        btn1_layout = QtWidgets.QHBoxLayout()
        btn1_layout.addWidget(latent_space_button)
        btn1_layout.addWidget(latent_space_pyplot_button)
        btn2_layout = QtWidgets.QHBoxLayout()
        btn2_layout.addWidget(make_model_button)
        gui_layout.addWidget(main_label)
        gui_layout.addWidget(second_label)
        gui_layout.addLayout(btn1_layout)
        gui_layout.addLayout(btn2_layout)
        gui_layout.addWidget(status)
        central_widget.setLayout(gui_layout)

    def latent_space_pyplot(self):
        encoded_data = self.auto_encoder.get_encoded()
        fig = plt.figure()
        self.ax = [fig.add_subplot(121, projection='3d')]
        self.ax.append(fig.add_subplot(122))
        y_test = self.auto_encoder.y_test
        self.ax[0].scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=y_test, s=8, cmap='tab10')
        fig.canvas.mpl_connect('key_press_event', self.onclick)
        plt.title("Press any key to load image")
        plt.show()

    def latent_space_epochs(self):
        encoded_data = self.auto_encoder.encoder_res
        y_test = self.auto_encoder.y_test.reshape(len(self.auto_encoder.y_test), 1)
        output_data = np.array([])
        for i in range(encoded_data.shape[0]):
            tmp = np.append(encoded_data[i], y_test, axis=1)
            tmp = np.append(tmp, np.expand_dims(np.ones(tmp.shape[0]) * (i + 1), axis=1), axis=1)
            if output_data.size == 0:
                output_data = tmp
            else:
                output_data = np.concatenate((output_data, tmp), axis=0)
        fig = px.scatter_3d(
            output_data, x=0, y=1, z=2, color=3,
            labels={'color': 'species'}, animation_frame=4,
        )
        fig["layout"].pop("updatemenus")  # optional, drop animation buttons
        fig.update_traces(marker_size=3)
        fig.show()

    def make_model(self, status):
        status.setText("Model is creating...")
        self.repaint()
        time.sleep(1)
        self.auto_encoder.make_model()
        status.setText("Model created!")

    def onclick(self, event):
        s = self.ax[0].format_coord(event.xdata, event.ydata)
        out = [float(x.split('=')[1].strip().replace(u'\N{MINUS SIGN}', '-')) for x in s.split(',')]
        latent_vector = np.array([out])

        decoded_img = self.auto_encoder.decoder.predict(latent_vector)
        decoded_img = decoded_img.reshape(28, 28)
        self.ax[1].imshow(decoded_img, cmap='gray')
        plt.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
