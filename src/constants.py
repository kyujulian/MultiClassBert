import datetime
import tensorflow as tf

AUTO = tf.data.AUTOTUNE
# Diretório com os dados
DATA_DIR = "./data/"

LOG_DIR = "./logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


LEARNING_RATE = 1e-5
#Configurações do modelo
EPOCHS=10
BATCH_SIZE=1
MAX_LEN=128
