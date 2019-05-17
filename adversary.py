import keras.backend as K
from keras.models import Model

def fgsm(model, gender_labels, eps=0.3, clip_min=0.0, clip_max=1.0):
    x = model.get_input_at(0)
    gender_output, _  = model(x)
    gender_loss = K.categorical_crossentropy(gender_labels, gender_output)
    
    grads = K.gradients(gender_loss, x)
    delta = K.sign(grads[0])
    x_adv = x + eps*delta
    x_adv = K.clip(x_adv, clip_min, clip_max)

    return x_adv




    
