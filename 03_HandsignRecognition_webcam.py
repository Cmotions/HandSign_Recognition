import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# define the minimum certainty of the prediction
threshold = .4

# define the webcam channel
webcamchannel = 0

# top k results that are shown at the screen
k = 5
top_k_print = ''

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    
    # create an empty array
    top_k_print = ''
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        
        if node_id < k:
            top_k_print = top_k_print + human_string.upper() + ' with a score of: ' + str(score) + '\n'
        
        if score > max_score:
            max_score = score
            res = human_string
    
    return res, max_score, top_k_print

    
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/output_labels.txt")]


# Unpersists graph from file
with tf.gfile.FastGFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(webcamchannel)

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]
            #img_cropped = img

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            a = cv2.waitKey(33)
            
            # check if backspace is pressed
            if  a == 8:
                sequence = sequence[:-1]
            
            if i == 4:
                res_tmp, score, top_k_print = predict(image_data)
                res = res_tmp
                i = 0
                
                #print(top_k_print)
                
                if mem == res and score > threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            i += 1
            
            # add a new line when the current line contains (a multiple of) 60 signs
            if len(sequence) > 0 and len(sequence) % 60 == 0:
                # add an enter to resume on the next line
                print('add enter')
                sequence += '\n'
            
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            
            mem = res
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)
            
            img_sequence = np.zeros((200,1200,3), np.uint8)
            
            # cut the sequence in different lines when relevant
            y0, dy, n = 30, 4, 1
            for seq in sequence.split('\n'):
                y = y0 + n * dy
                n += 10
                cv2.putText(img_sequence, '%s' % (seq.upper()), (30,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.imshow('sequence', img_sequence)
            
            
            img_sequence2 = np.zeros((250,600,3), np.uint8)
            # cut the sequence in different lines when relevant
            y0, dy, n = 30, 4, 1
            for seq in top_k_print.split('\n'):
                y = y0 + n * dy
                n += 10
                cv2.putText(img_sequence2, '%s' % (seq.upper()), (30,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.imshow('Top ' + str(k), img_sequence2)
            
        else:
            break

# Sluit alle geopende schermen
cv2.destroyAllWindows() 
cv2.VideoCapture(webcamchannel).release()
