import tensorflow as tf
import numpy as np

class Logger(object):
    
    def __init__(self, log_dir):
        
        self.writer = tf.summary.FileWriter(log_dir)
    
    def scalar_summary(self, tag, value, step):
        
        summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writter.add_summary(summary, step)

    def image_summary(self, name, x, step):

        x = x.numpy()[0,:,:,:]
        x = np.moveaxis(x,0,-1)
        x = np.expand_dims(x,0)

        tensor = tf.convert_to_tensor(
                x,
                dtype=tf.floadt32,
                name=None,
                preferred_dtype=None
        )

        print(tensor.value)

        summary = tf.summary.image(name=name, tensor=tensor)

        self.writer.add_summary(summary, step).eval()
                
