from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tokenizers import BertWordPieceTokenizer 

import transformers

from constants import *


transformer_layer = (
    transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased')
)
#Tokenizer
#importando da hugging faces
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#Salva o tokenizador localmente
tokenizer.save_pretrained("../model_content")
fast_tokenizer = BertWordPieceTokenizer("../model_content/vocab.txt", lowercase=False)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
eartly_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) 

CALLBACKS = [tensorboard_callback, eartly_stopping_callback]
def build_model(transformer, max_len=512):
    '''
    Função para construir o modelo de classificação de sentimentos,
    utilizando o modelo BERT pré-treinado.
    '''

    #Camadas relacionadas ao modelo pre-treinado
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    
    
    #Camada de classificação
    first_dense = Dense(12, activation="relu")(cls_token)
    out = Dense(4, activation="softmax")(first_dense)
    
    model = Model(input_word_ids, out)
    model.compile(Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

    
def train_model(train_dataset, dev_dataset, model, callbacks=CALLBACKS):

    n_steps = train_dataset.shape[0] // BATCH_SIZE
    # Função definida no inicio do notebook

    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=dev_dataset,
        epochs=EPOCHS,
        callbacks = callbacks
    )

    return train_history, model