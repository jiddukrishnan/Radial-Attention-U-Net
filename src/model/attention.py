from tensorflow.keras.layers import Conv2D, Add, Activation, Multiply, BatchNormalization, LeakyReLU

def attention_gate(g, x, inter_channels):
    theta_x = Conv2D(inter_channels, (1,1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1,1), padding='same')(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    f = Conv2D(inter_channels//2, (3,3), padding='same')(f)
    f = BatchNormalization()(f)
    f = LeakyReLU(0.1)(f)
    psi = Conv2D(1, (1,1), padding='same')(f)
    coef = Activation('sigmoid')(psi)
    return Multiply()([x, coef])
