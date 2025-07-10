from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Concatenate

def radial_convolution(x, filters, radial_levels=3):
    base = Conv2D(filters, (3,3), padding='same')(x)
    base = BatchNormalization()(base)
    base = LeakyReLU(0.1)(base)
    radial_features = [base]
    for i in range(1, radial_levels+1):
        sc = Conv2D(filters, (3,3), padding='same', dilation_rate=(i,i))(base)
        sc = BatchNormalization()(sc)
        sc = LeakyReLU(0.1)(sc)
        radial_features.append(sc)
    fused = Concatenate()(radial_features)
    fused = Conv2D(filters, (1,1), padding='same')(fused)
    fused = BatchNormalization()(fused)
    x_proj = Conv2D(filters, (1,1), padding='same')(x)
    x_proj = BatchNormalization()(x_proj)
    out = Add()([fused, x_proj])
    out = LeakyReLU(0.1)(out)
    return out

 def conv_block(x, filters):
    x = radial_convolution(x, filters)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    return x
