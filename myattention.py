
import tensorflow as tf 
import numpy as np 

#%% generate data
inp = list()
targ = list()
for i in range(1000):
    for j in range(1000):
        r = str(i) + '+' + str(j)
        if len(r) < 7:
            r = r + ' ' * (7-len(r))
        inp.append(r)
        
        r = '_' + str(i+j)
        if len(r) < 5:
            r = r + ' ' * (5 - len(r))
        targ.append(r)
        
#%% prepare data for encoding: str to nums
num2str_dict = dict(list(enumerate([str(i) for i in range(10)] + ['_', '+', ' '])))
str2num_dict = {v:k for (k,v) in num2str_dict.items()}

inp_array = list()
for i in inp:
    inp_array.append([str2num_dict[j] for j in i])
    
targ_array = list()
for i in targ:
    targ_array.append([str2num_dict[j] for j in i])

#%% change data to np.array
permu_idx = np.random.permutation(len(inp_array))
inp_array = np.array(inp_array)[permu_idx]
targ_array = np.array(targ_array)[permu_idx]

# reverse input chars to have better results
inp_array = inp_array[:,::-1]
    
#%% split train and test
train_num = int(len(inp_array) * 0.6)
train_inp, train_targ = inp_array[:train_num], targ_array[:train_num]
test_inp, test_targ = inp_array[train_num:], targ_array[train_num:]
            
#%% reverse data to have better results (optional)
batch_size = 64
train_data = tf.data.Dataset.from_tensor_slices((train_inp, train_targ))\
               .batch(batch_size, drop_remainder=True)

#%% models
class Encoder(tf.keras.Model):
    def __init__(self, embed_dim, h_dim):
        super(Encoder, self).__init__()
        self.embed_layer = tf.keras.layers.Embedding(13, embed_dim)
        self.lstm_layer = tf.keras.layers.LSTM(h_dim,
                                               activation='tanh',
                                               stateful=False,
                                               return_sequences=True,
                                               return_state=True)
        
    def call(self, inputs):
        '''
        INPUT
            inputs     [batch_size, seq_length]
        '''
        x = self.embed_layer(inputs) #[batch_size, seq_length, embed_dim]
        keys, h, c = self.lstm_layer(x) # keys = [batch_size, seq_length, h_dim]
        return keys, h, c
        
class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, h_dim):
        super(Decoder, self).__init__()
        self.embed_layer = tf.keras.layers.Embedding(13, embed_dim)
        self.lstm_layer = tf.keras.layers.LSTM(h_dim,
                                               return_sequences=False,
                                               return_state=True,
                                               stateful=False,
                                               activation='tanh')
        self.dense_layer = tf.keras.layers.Dense(h_dim)
        self.softmax_layer = tf.keras.layers.Dense(13)
        
    def attention_layer(self, query, keys):
        '''
        INPUT
            query    [batch_size, h_dim]
            keys     [batch_size, seq_length, h_dim]
        '''
        attention_weights = keys * query[:,tf.newaxis,:] #[batch_size, seq_length, h_dim]
        attention_weights = tf.reduce_sum(attention_weights, axis=-1) #[batch_size, seq_length]
        attention_weights = tf.math.softmax(attention_weights, axis=-1) #[bath_size, seq_length]
        
        context_vec = keys * attention_weights[:,:,tf.newaxis] #[bath_size, seq_length, h_dim]
        context_vec = tf.reduce_sum(context_vec, axis=1) #[batch_size, h_dim]
        
        return attention_weights, context_vec
    
    def call(self, x, keys, h, c):
        '''
        INPUT
            x         [batch_size, 1]
            keys      [batch_size, seq_length, h_dim]
            h         [batch_size, h_dim]
            c         [batch_size, h_dim]
        '''
        attention_weights, context_vec = self.attention_layer(h, keys)
        x = self.embed_layer(x) #[batch_size, 1, embed_dim]
        x, h, c = self.lstm_layer(x, initial_state=(h,c)) # x = h = c = [batch_size, h_dim]
        x = tf.concat([context_vec, x], axis=-1) #[batch_size, h_dim*2]
        x = self.dense_layer(x)
        x = self.softmax_layer(x)
        return x, h, c, attention_weights
        
#%% model training settings
embed_dim = 32
h_dim = 32
encoder = Encoder(embed_dim, h_dim)
decoder = Decoder(embed_dim, h_dim)
learning_rate = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
loss_func = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_a_batch(x,y):
    '''
    INPUT
        x     [batch_size, seq_length]
        y     [batch_size, seq_length]
    '''
    with tf.GradientTape() as tape:
        keys, h, c = encoder(x)
        loss = 0
        
        # predict the next character
        pos = 0
        while pos < 4:
            inputs = y[:,pos][:,tf.newaxis]
            pred, h, c, _ = decoder(inputs, keys, h, c)
            loss += loss_func(y[:,pos+1], pred)
            pos += 1
        
        # optimizing
        variables = encoder.variables + decoder.variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables))
    
    return loss

def train_epoch(train_num, batch_size):
    steps = int(train_num / batch_size)
    loss = 0
    for x,y in train_data.take(steps):
        loss += train_a_batch(x,y)
    loss = loss / steps
    
    return loss

def predict(test_inp):
    keys, h, c = encoder(test_inp)
    
    # init pred to be the index of '_'
    pred = tf.constant([[10]]*test_inp.shape[0], dtype=tf.int64)
    total_prediction = pred
    
    # init the decoding
    pos = 0
    while pos < 4:
        pred_logit, h, c, attention_weights = decoder(pred, keys, h, c)
        pred = tf.argmax(pred_logit, axis=-1)[:,tf.newaxis]
        total_prediction = tf.concat([total_prediction, pred], axis=1)
        pos += 1

    return total_prediction
    
def pred_accuracy(test_targ, pred):
    loss = tf.pow(test_targ - pred, 2)
    loss = tf.reduce_sum(loss, axis=1)
    accuracy = tf.reduce_sum(tf.cast(loss == 0, dtype=tf.float32)) / len(loss)
    return accuracy
    
#%% train the model
nEpochs = 10
loss_list = []
acc_list = []
for epoch in range(nEpochs):
    loss = train_epoch(train_num, batch_size)
    pred = predict(test_inp)
    test_acc = pred_accuracy(test_targ, pred)
    print("epoch {}, loss {:.2f}, test accuracy {:.2f}".format(epoch, loss, test_acc))
    loss_list.append(loss)
    acc_list.append(test_acc)
