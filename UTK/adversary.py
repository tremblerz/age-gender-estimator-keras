import tensorflow as tf
import keras.backend as K

def fgsm(model, race_labels, eps=0.3, clip_min=0.0, clip_max=1.0):
    x = model.get_input_at(0)
    age_output, race_output, gender_output  = model(x)
    #race_loss = K.categorical_crossentropy(race_labels, race_output)
    
    dense_out = model.get_layer('dense_4').output
    grads = K.gradients(dense_out, x)
    delta = K.sign(grads[0])
    x_adv = x + eps*delta
    x_adv = K.clip(x_adv, clip_min, clip_max)

    return x_adv


    
