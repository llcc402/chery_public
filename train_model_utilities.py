import numpy as np 
import tensorflow as tf 
from utilities import * 
from models import HMCModel, HMC_LSTM, CoherentLSTM
import matplotlib.pyplot as plt 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_train_step_fn(model, optimizer, loss_fn, model_name):
    def train_step_fn(x, y, validation_data=None):
        with tf.GradientTape() as tape:
            if model_name == 'coherent_hmc':
                logits = model(x)
                logits = model.postprocess(logits)
                loss = loss_fn(y, logits)
            elif model_name == 'lstm_hmc':
                logits = model(x)
                loss = loss_fn(y, logits)
            elif model_name == 'coherent_lstm':
                local_logits, global_logits = model(x)
                local_logits, global_logits = model.postprocess(local_logits, global_logits)
                local_loss = loss_fn(y, local_logits)
                global_loss = loss_fn(y, global_logits)
                loss = local_loss + global_loss
            else:
                raise("The input model is not a valid model!\nShould be one of 'coherent'_hmc', 'lstm_hmc' or 'coherent_lstm'")
            
            if validation_data is not None:
                valid_x = validation_data[0]
                valid_y = validation_data[1]
                if model_name == 'coherent_hmc':
                    valid_logits = model(valid_x)
                    valid_logits = model.postprocess(valid_logits)
                    valid_loss = loss_fn(valid_y, valid_logits)
                elif model_name == 'lstm_hmc':
                    valid_logits = model(valid_x)
                    valid_loss = loss_fn(valid_y, valid_logits)
                elif model_name == 'coherent_lstm':
                    valid_local_logits, valid_global_logits = model(valid_x)
                    valid_local_logits, valid_global_logits = model.postprocess(valid_local_logits, valid_global_logits)
                    valid_local_loss = loss_fn(valid_y, valid_local_logits)
                    valid_global_loss = loss_fn(valid_y, valid_global_logits)
                    valid_loss = valid_local_loss + valid_global_loss
            
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        return loss, valid_loss 
    return train_step_fn

def model_score(model, x, y, model_name):
    if model_name == 'coherent_hmc':
        logits = model(x)
        logits = model.postprocess(logits)
        prob = model.get_prob(logits)
    elif model_name == 'lstm_hmc':
        local_logits, global_logits = model(x)
        prob, _, _ = model.get_prob(local_logits, global_logits)
    elif model_name == 'coherent_lstm':
        local_logits, global_logits = model(x)
        local_logits, global_logits = model.postprocess(local_logits, global_logits)
        prob, _, _ = model.get_prob(local_logits, global_logits)
    else:
        raise("The input model is not a valid model!\nShould be one of 'coherent'_hmc', 'lstm_hmc' or 'coherent_lstm'")
    score = get_score(y, prob)
    return score 

