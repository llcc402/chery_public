import tensorflow  as tf 
import numpy as np 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
        pred_int = tf.cast(y_pred >= threshold, tf.int16)
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
        loss = tf.reduce_mean(loss1 + loss2 + loss3)

        return loss 
    return loss_fn_coherent

def get_loss_fn_sigmoid():
    def loss_fn_sigmoid(y_true, y_pred):
        ''' 
        INPUTS  
            y_true      binary, shape = [N,M]
            y_pred      logits, shape = [N,M]
        '''
        loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
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
    