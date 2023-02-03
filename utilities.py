import tensorflow  as tf 
import numpy as np 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.metrics import average_precision_score

def custom_score(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    idx = np.sum(y_true, axis=0) != y_true.shape[0] # remove root nodes
    y_true = y_true[:, idx]
    y_pred = y_pred[:, idx]
    return average_precision_score(y_true, y_pred, average='micro')

def get_area(x1, y1, x2, y2):
    return (y1 + y2) * (x2 - x1) / 2

def get_score(y_true, y_pred):
    '''
    INPUT 
        y_true      binary, shape = [N,M],   
        y_pred      probabilities, shape = [N,M], N = number of examples, M = number of classes 
    OUTPUT
        AU(PRC) 
    '''
    precision_list = [0]
    recall_list = [1]
    for threshold in np.arange(0.01, 1, 0.01):
        pred_int = tf.cast(y_pred >= threshold, tf.float32)
        tp = pred_int * y_true
        fp = pred_int * (1-y_true)
        fn = (1-pred_int) * y_true
        
        tp = tf.reduce_sum(tp)
        fp = tf.reduce_sum(fp)
        fn = tf.reduce_sum(fn)
        
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        
        if not (np.isnan(precision) or np.isnan(recall)):
            precision_list.append(precision)
            recall_list.append(recall)
        
    idx = np.argsort(precision_list)
    precision_np = np.array(precision_list)
    recall_np = np.array(recall_list)
    
    precision_np = precision_np[idx]
    recall_np = recall_np[idx]
    
    area = 0 
    for i in range(len(precision_np)-1):
        area += get_area(precision_np[i], recall_np[i], precision_np[i+1], recall_np[i+1])
    
    return area

def get_max_with_structure(structure):
    ''' 
    structure   an np.array with shape [M,M], where M is the number of all classes
                structure[i,j] == 1 iff "i" is a subclass of "j". Remind that "i" is 
                subclass of itself, so the diagonal elements are all 1
                DIAGNOAL ELEMENTS ARE ZEROS
    '''
    def max_with_structure(logits):
        '''
        logits      shape = [N,M]
        '''
        logits = logits[..., tf.newaxis] # [N,M,1]
        
        structure1 = structure + np.eye(structure.shape[0])
        structure1 = np.where(structure1 == 0, -np.inf, structure1)
        structure1 = np.where(structure1 == 1, 0, structure1)
        structure1 = tf.convert_to_tensor(structure1[np.newaxis,...], tf.float32) # [1,M,M]
        
        logits = logits + structure1 # [N,M,M]
        logits = tf.reduce_max(logits, axis=1) #[N,M]
        
        logits = tf.where(tf.math.is_inf(logits), np.inf, logits)
        return logits 
    return max_with_structure

def softplus(x):
    ''' 
    return log(1+exp(-x))
    '''
    r = tf.maximum(-x, 0) + tf.math.log(1 + tf.exp(-tf.abs(x)))
    return r 

def get_cross_logits_y(y):
    @tf.custom_gradient
    def cross_logits_y(logits):
        outputs = tf.where(y == 0, -np.inf, logits)
        def grad(upstream):
            return upstream * tf.where(y == 1, 1.0, 0.0)
        return outputs, grad
    return cross_logits_y

def remove_root_from_loss(y_true, loss):
    '''
    INPUT
        y_true      [N,M], where M is the number of all classes
        loss        [N,M], loss for every single element
    OUTPUT  
        the "roots" column of loss are assigned 0
    '''
    idx = tf.where(tf.reduce_sum(y_true, axis=0) == y_true.shape[0]).numpy()
    mask = np.ones_like(loss)
    mask[:, idx] = 0
    return loss * tf.convert_to_tensor(mask, tf.float32)

def get_loss_fn_coherent(structure):
    def loss_fn_coherent(y_true, y_pred_logits):
        ''' 
        INPUT
            y_true      binary, [N,M]
            y_pred      logits before postprocessing, [N,M]
        '''
        cross_logits_y = get_cross_logits_y(y_true)
        max_with_structure = get_max_with_structure(structure)
        
        loss1 = y_true * softplus(max_with_structure(cross_logits_y(y_pred_logits)))
        loss2 = (1-y_true) * softplus(max_with_structure(y_pred_logits)) 
        loss3 = (1-y_true) * max_with_structure(y_pred_logits)
        loss = remove_root_from_loss(y_true, loss1+loss2+loss3)
        loss = tf.reduce_mean(loss)
        return loss 
    return loss_fn_coherent

def get_loss_fn_sigmoid():
    def loss_fn_sigmoid(y_true, y_pred):
        ''' 
        INPUTS  
            y_true      binary, shape = [N,M]
            y_pred      logits, shape = [N,M]
        '''
        loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_mean(loss)
    return loss_fn_sigmoid

def generate_rectangular_data(x, y, w, h, size=100):
    '''
    INPUT
        x       float, center coordinate of a rectangular
        y       float, center coordinate of a rectangular
        w       float(> 0), width of a rectangular
        h       float(> 0), height of a rectangular
        size    int, number of examples
    OUTPUT
        randomly sampled data in the given rectangular, shape = [size, 2]
    '''
    a = np.random.uniform(low=0, high=1, size=size)
    a *= w 
    a = a - w/2.0 + x 
    
    b = np.random.uniform(low=0, high=1, size=size)
    b *= h 
    b = b - h/2.0 + y
    
    return np.concatenate([a[:,np.newaxis], b[:,np.newaxis]], axis=1)

# def get_loss_fn_lstm(structure, alpha, beta):
#     ''' 
#     structure     array of shape [M,M], where M is the number of all classes 
#                   (a hierarchical level may contain multiple classes), 
#                   structure[i,j] == 1 iff "class i" is a subclass of "class j"
#     alpha         a hyper-parameter to balance hierarchical loss and prediction loss
#     beta          a hyper-parameter to balance local prob and global prob
#     '''
#     def loss_fn_lstm(y_true, y_pred_logits):
#         ''' 
#         y_true     [N,K], K = K1 + K2 + ... + KM, where M is the number of layers
#         y_pred     [total_prob, local_prob, global_prob], local_prob = [N,K], global_prob = [N,K]
#         '''
#         local_logits, global_logits = y_pred_logits 
        
#         local_loss = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(y_true, local_logits)
#         )
        
#         global_loss = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(y_true, global_logits)
#         )
        
#         total_prob = beta * tf.math.sigmoid(local_logits) + (1-beta) * tf.math.sigmoid(global_logits)
        
#         hierachical_loss = 0
#         for i in range(structure.shape[0]):
#             for j in range(structure.shape[1]):
#                 if structure[i,j] == 1:
#                     hierachical_loss += tf.reduce_sum(
#                         tf.where(
#                             total_prob[:,i] < total_prob[:,j], 
#                             0, 
#                             tf.pow(total_prob[:,i] - total_prob[:,j],2)
#                         )
#                     )
#         hierachical_loss /= y_true.shape[0]
        
#         return local_loss + global_loss + alpha * hierachical_loss
        
#     return loss_fn_lstm

def get_loss_fn_lstm(structure, alpha, beta):
    ''' 
    structure     array of shape [M,M], where M is the number of all classes 
                  (a hierarchical level may contain multiple classes), 
                  structure[i,j] == 1 iff "class i" is a subclass of "class j"
    alpha         a hyper-parameter to balance hierarchical loss and prediction loss
    beta          a hyper-parameter to balance local prob and global prob
    '''
    def loss_fn_lstm(y_true, y_pred_logits):
        ''' 
        y_true     [N,K], K = K1 + K2 + ... + KM, where M is the number of layers
        y_pred     [total_prob, local_prob, global_prob], local_prob = [N,K], global_prob = [N,K]
        '''
        local_logits, global_logits = y_pred_logits 
        
        local_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y_true, local_logits)
        )
        
        global_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y_true, global_logits)
        )
        
        total_prob = beta * tf.math.sigmoid(local_logits) + (1-beta) * tf.math.sigmoid(global_logits)
        
        hierachical_loss = 0
        total_prob_1 = total_prob[:,:,tf.newaxis]
        total_prob_2 = total_prob[:,tf.newaxis,:]
        structure_expand = structure[tf.newaxis,:,:]
        hierachical_loss += tf.reduce_sum(
            structure_expand * tf.cast(
                (-total_prob_1 + total_prob_2) < 0,
                tf.float32) * tf.pow(total_prob_1 - total_prob_2, 2)
        )
        
        hierachical_loss /= y_true.shape[0]
        
        return local_loss + global_loss + alpha * hierachical_loss
        
    return loss_fn_lstm

def split_train_valid_test(data, y):
    N = data.shape[0]
    idx = np.random.permutation(N)
    train_num = int(N * 6 / 10)
    valid_num = int(N * 2 / 10)

    train_idx = idx[:train_num]
    valid_idx = idx[train_num:(train_num + valid_num)]
    test_idx = idx[(train_num + valid_num):]

    x_train, x_valid, x_test = data[train_idx], data[valid_idx], data[test_idx]
    y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]

    y_train = tf.convert_to_tensor(y_train, tf.float32)
    y_valid = tf.convert_to_tensor(y_valid, tf.float32)
    y_test = tf.convert_to_tensor(y_test, tf.float32)
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def log_neg_prob(x):
    '''
    return log (1-p(x)), where p(x) = sigmoid(x)
    '''
    return -x - softplus(x)

# def get_loss_fn_log(structure, alpha):
#     def loss_fn_log(y_true, y_pred_logits):
#         ''' 
#         y_true     [N,K], K = K1 + K2 + ... + KM, where M is the number of layers
#         y_pred     [total_prob, local_prob, global_prob], local_prob = [N,K], global_prob = [N,K]
#         '''
#         local_logits, global_logits = y_pred_logits 
#         local_loss = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(y_true, local_logits)
#         )
        
#         global_loss = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(y_true, global_logits)
#         )
        
#         hierachical_loss = 0
#         for i in range(structure.shape[0]):
#             for j in range(structure.shape[1]):
#                 if structure[i,j] == 1:
#                     hierachical_loss += tf.reduce_sum(
#                         tf.where(
#                             local_logits[:,i] < local_logits[:,j], 
#                             0, 
#                             log_neg_prob(local_logits[:,j]) - log_neg_prob(local_logits[:,i])
#                         )
#                     )
#                     hierachical_loss += tf.reduce_sum(
#                         tf.where(
#                             global_logits[:,i] < global_logits[:,j], 
#                             0, 
#                             log_neg_prob(global_logits[:,j]) - log_neg_prob(global_logits[:,i])
#                         )
#                     )
#         hierachical_loss /= y_true.shape[0]
        
#         return local_loss + global_loss + alpha * hierachical_loss
        
#     return loss_fn_log

def get_loss_fn_log(structure, alpha):
    def loss_fn_log(y_true, y_pred_logits):
        ''' 
        y_true     [N,K], K = K1 + K2 + ... + KM, where M is the number of layers
        y_pred     [total_prob, local_prob, global_prob], local_prob = [N,K], global_prob = [N,K]
        '''
        local_logits, global_logits = y_pred_logits 
        local_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y_true, local_logits)
        )
        
        global_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y_true, global_logits)
        )
        
        structure_expand = structure[tf.newaxis,:,:]
        
        hierachical_loss = 0
        local_logits_1 = local_logits[:,:,tf.newaxis]
        local_logits_2 = local_logits[:,tf.newaxis,:]
        local_log_neg_prob = log_neg_prob(local_logits)
        local_log_neg_prob_1 = local_log_neg_prob[:,:,tf.newaxis]
        local_log_neg_prob_2 = local_log_neg_prob[:,tf.newaxis,:]
        
        hierachical_loss += tf.reduce_sum(
            structure_expand * tf.cast(
                (-local_logits_1 + local_logits_2) < 0,
                tf.float32) * (local_log_neg_prob_2 - local_log_neg_prob_1)
        )
        global_logits_1 = global_logits[:,:,tf.newaxis]
        global_logits_2 = global_logits[:,tf.newaxis,:]
        global_log_neg_prob = log_neg_prob(global_logits)
        global_log_neg_prob_1 = global_log_neg_prob[:,:,tf.newaxis]
        global_log_neg_prob_2 = global_log_neg_prob[:,tf.newaxis,:]
        
        hierachical_loss += tf.reduce_sum(
            structure_expand * tf.cast(
                (-global_logits_1 + global_logits_2) < 0,
                tf.float32) * (global_log_neg_prob_2 - global_log_neg_prob_1)
        )
        
        hierachical_loss /= y_true.shape[0]
        
        return local_loss + global_loss + alpha * hierachical_loss
        
    return loss_fn_log

def get_structure_from_adajancency(adajancency):
    structure = np.zeros(adajancency.shape)
    g = nx.DiGraph(adajancency) # train.A is the matrix where the direct connections are stored 
    for i in range(len(adajancency)):
        ancestors = list(nx.descendants(g, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        if ancestors:
            structure[i, ancestors] = 1
    return structure 