from keras import backend as K

# ============
# Explaination
# ============

x1 = K.placeholder(shape=(64, 15))  # lstm_1 (LSTM)
print("x1:", x1.shape)
x2 = K.placeholder(shape=(64, 15, 15))  # dense (Dense)
print("x2:", x2.shape)
w0 = K.placeholder(shape=(15, 15))  # from lstm_1 (LSTM)
print("w0:", w0.shape)
w1 = K.placeholder(shape=(15, 15))  # dense (Dense)
print("w1:", w1.shape)
w2 = K.placeholder(shape=(15, 15))  # from lstm_1 (LSTM)
print("w2:", w2.shape)
x1 = K.permute_dimensions(x1, (0, 1))  # This line is extra in my opinion
print("x1 after permutation:", x1.shape)
x2 = K.permute_dimensions(x2[:, -1, :], (0, 1)) 
print("x2 after permutation:", x2.shape)
a = K.softmax(K.tanh(K.dot(x1, w0) + K.dot(x2, w1)))
print("a in first step:", x2.shape)
a = K.dot(a, w2)
print("a in second step:", x2.shape)
outputs = K.permute_dimensions(a * x1, (0, 1))
print("output after permutation:", outputs.shape)
# outputs = K.sum(outputs, axis=1)
outputs = K.l2_normalize(outputs, axis=1)
print("output after normalization:", outputs.shape)

"""
batch_size = None
x1: (None, 15) # LSTM o/p
x2: (None, 15, 15) # Dense o/p (after TimeDistributed convs)
w0: (15, 15) # from LSTM o/p
w1: (15, 15) # from Dense o/p
w2: (15, 15) # from LSTM o/p
x1 after permutation: (None, 15)
x2 after permutation: (None, 15)
a in first step: (None, 15)
a in second step: (None, 15)
output after permutation: (None, 15)
output after normalization: (None, 15)

output: (None, 15)
"""

"""
Attnention has two inputs with shapes: (None, 15) -> call it i_1,  (None, 15, 15) call it i_2
- W0: (15,15) from i_1
- W1: (15,15) from i_2
- W2: (15,15) from i_1
- No bias

- x1 = K.permute_dimensions(inputs[0], (0, 1)) -> (None, 15, 30) truns into (None, 15)
- x2 = K.permute_dimensions(inputs[1][:, -1, :], (0, 1)) -> (None, 15, 15) turns into (None, 15)

"""
"""
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 main_input (InputLayer)        [(None, 15, 7, 1)]   0           []                               
                                                                                                  
 time_distributed (TimeDistribu  (None, 15, 7, 15)   60          ['main_input[0][0]']             
 ted)                                                                                             
                                                                                                  
 time_distributed_1 (TimeDistri  (None, 15, 7, 15)   690         ['time_distributed[0][0]']       
 buted)                                                                                           
                                                                                                  
 time_distributed_2 (TimeDistri  (None, 15, 105)     0           ['time_distributed_1[0][0]']     
 buted)                                                                                           
                                                                                                  
 dense (Dense)                  (None, 15, 15)       1590        ['time_distributed_2[0][0]']     
                                                                                                  
 lstm (LSTM)                    (None, 15, 15)       1860        ['dense[0][0]']                  
                                                                                                  
 attention_with_context (Attent  (None, 15, 15)      255         ['lstm[0][0]']                   
 ionWithContext)                                                                                  
                                                                                                  
 auxiliary_input_w (InputLayer)  [(None, 15, 1)]     0           []                               
                                                                                                  
 auxiliary_input_d (InputLayer)  [(None, 15, 1)]     0           []                               
                                                                                                  
 lstm_1 (LSTM)                  (None, 15)           1860        ['attention_with_context[0][0]'] 
                                                                                                  
 bidirectional (Bidirectional)  (None, 15, 30)       2040        ['auxiliary_input_w[0][0]']      
                                                                                                  
 bidirectional_2 (Bidirectional  (None, 15, 30)      2040        ['auxiliary_input_d[0][0]']      
 )                                                                                                
                                                                                                  
 attention_layer (AttentionLaye  (None, 15)          675         ['lstm_1[0][0]',                 
 r)                                                               'dense[0][0]']                  
                                                                                                  
 bidirectional_1 (Bidirectional  (None, 30)          5520        ['bidirectional[0][0]']          
 )                                                                                                
                                                                                                  
 bidirectional_3 (Bidirectional  (None, 30)          5520        ['bidirectional_2[0][0]']        
 )                                                                                                
                                                                                                  
 concatenate (Concatenate)      (None, 75)           0           ['attention_layer[0][0]',        
                                                                  'bidirectional_1[0][0]',        
                                                                  'bidirectional_3[0][0]']        
                                                                                                  
 dense_1 (Dense)                (None, 20)           1520        ['concatenate[0][0]']            
                                                                                                  
 dense_2 (Dense)                (None, 10)           210         ['dense_1[0][0]']                
                                                                                                  
 main_output (Dense)            (None, 1)            11          ['dense_2[0][0]']                
                                                                                                  
==================================================================================================
Total params: 23,851
Trainable params: 23,851
Non-trainable params: 0
__________________________________________________________________________________________________
"""
