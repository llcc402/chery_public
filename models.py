from tensorflow.keras.layers import Dense, Dropout 
import tensorflow as tf 
from utilities import * 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class HMCModel(tf.keras.models.Model):
    def __init__(self, structure, num_classes, hid_dim_list, dropout):
        ''' 
        INPUT 
            structure           np.array, [M,M]
            num_classes         integer, M = num_classes
            hid_dim_list        a list of integers
        '''
        super(HMCModel, self).__init__()
        self.structure = structure
        self.num_hid_layers = len(hid_dim_list)
        
        self.W1 = list()
        self.dropout = list()
        for i in range(len(hid_dim_list)):
            self.W1.append(tf.keras.layers.Dense(hid_dim_list[i], activation='relu'))
            self.dropout.append(Dropout(dropout))
        self.W2 = tf.keras.layers.Dense(num_classes)
        
    def call(self, inputs):
        x = self.W1[0](inputs)
        for i in range(1, self.num_hid_layers):
            x = self.W1[i](x)
            x = self.dropout[i](x)
        outputs = self.W2(x)
        return outputs
    
    def get_prob(self, logits):
        return tf.nn.sigmoid(logits)
    
    def postprocess(self, inputs):
        max_with_structure = get_max_with_structure(self.structure)
        return max_with_structure(inputs)
    
class HMC_LSTM(tf.keras.Model):
    def __init__(self, units, beta, num_classes_list, num_layers):
        '''
        units               number of nurons in LSTM
        beta                integer, balance global and local predictions
        num_classes_list    a list of number of classes in each class layer
        num_layers          integer, len(num_classes_list) == num_layers
        '''
        super(HMC_LSTM, self).__init__()
        self.units = units 
        self.beta = beta 
        self.num_layers = num_layers
        
        self.input_gate = Dense(units,activation='sigmoid')
        self.output_gate = Dense(units, activation='sigmoid')
        self.forget_gate = Dense(units, activation='sigmoid')
        self.candidate_gate = Dense(units, activation='tanh')
        
        self.local_W = list()
        for i in range(self.num_layers):
            self.local_W.append(Dense(num_classes_list[i]))
        
        self.global_W = Dense(tf.reduce_sum(num_classes_list))
            
    def call(self, inputs):
        ''' 
        inputs     [N,k], where N is the number of examples, k is the number of features
        '''
        # init sequence
        x = tf.concat(
            [inputs, tf.zeros([inputs.shape[0],self.units])], 
            axis=-1
        )
        candidate_last = tf.zeros(self.units)
        
        # output probs
        local_probs = list()
        
        # main loop
        for i in range(self.num_layers):
            # gates
            forget_prob = self.forget_gate(x)
            input_prob = self.input_gate(x)
            output_prob = self.output_gate(x)
            candidate_hat = self.candidate_gate(x)
            
            # new candidate
            candidate = candidate_hat * input_prob + forget_prob * candidate_last
            
            # outputs
            outputs = output_prob * tf.math.tanh(candidate)
            
            # update
            candidate_last = candidate 
            x = tf.concat([inputs, outputs], axis=-1)

            # local probs
            local_probs.append(self.local_W[i](outputs))
            
        global_logits = self.global_W(
            tf.concat([inputs, outputs], axis=-1)
        )
        local_logits = tf.concat(local_probs, axis=-1)
        
        return local_logits, global_logits
    
    def get_prob(self, local_logits, global_logits):
        local_prob = tf.nn.sigmoid(local_logits)
        global_prob = tf.nn.sigmoid(global_logits)
        total_prob = self.beta * local_prob + (1-self.beta) * global_prob
        return total_prob, local_prob, global_prob
    
class CoherentLSTM(tf.keras.Model):
    def __init__(self, units, beta, num_classes_list, num_layers, structure):
        super(CoherentLSTM, self).__init__()
        self.units = units
        self.beta = beta 
        self.num_layers = num_layers
        self.structure = structure 
        
        self.input_gate = Dense(units, activation='sigmoid')
        self.forget_gate = Dense(units, activation='sigmoid')
        self.output_gate = Dense(units, activation='sigmoid')
        self.candidate_gate = Dense(units, activation='tanh')
        
        self.local_W = list()
        for i in range(num_layers):
            self.local_W.append(Dense(num_classes_list[i]))
        self.global_W = Dense(tf.reduce_sum(num_classes_list))
        
    def call(self, inputs):
        # init 
        x = tf.concat(
            [inputs, tf.zeros([inputs.shape[0], self.units])],
            axis = -1
        )
        candidate_last = tf.zeros(self.units)
        local_logits = list()
        
        # main loop
        for i in range(self.num_layers):
            # gates
            forget_prob = self.forget_gate(x)
            input_prob = self.input_gate(x)
            output_prob = self.output_gate(x)
            candidate_hat = self.candidate_gate(x)
            
            # outputs
            candidate = input_prob * candidate_hat + forget_prob * candidate_last
            outputs = output_prob * tf.math.tanh(candidate)
            
            # update
            candidate_last = candidate 
            x = tf.concat([inputs, outputs], axis=-1)
            
            local_logits.append(self.local_W[i](outputs))
            
        global_logits = self.global_W(
            tf.concat([inputs, outputs], axis=-1)
        )
        
        return tf.concat(local_logits, axis=-1), global_logits
    
    def postprocess(self, local_logits, global_logits):
        max_with_structure = get_max_with_structure(self.structure)
        local_logits = max_with_structure(local_logits)
        global_logits = max_with_structure(global_logits)
        return local_logits, global_logits
    
    def get_prob(self, local_logits, global_logits):
        local_prob = tf.math.sigmoid(local_logits)
        global_prob = tf.math.sigmoid(global_logits)
        total_prob = self.beta * local_prob + (1-self.beta) * global_prob
        return total_prob, local_prob, global_prob