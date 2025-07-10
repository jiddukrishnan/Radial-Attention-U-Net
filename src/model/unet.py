from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from .blocks import conv_block
from .attention import attention_gate

def build_radial_attention_unet(input_shape=(128,128,3), base_filters=32, spiral_levels=3):
    inputs = Input(shape=input_shape)
    filters = [base_filters * (2**i) for i in range(4)]
    skips = []
    x = inputs
    for f in filters:
        x = conv_block(x, f)
        skips.append(x)
        x = MaxPooling2D()(x)
    x = spiral_convolution(x, filters[-1]*2, spiral_levels)
    for i in reversed(range(len(filters))):
        f = filters[i]
        x = UpSampling2D()(x)
        attn = attention_gate(x, skips[i], f//2)
        x = Concatenate()([x, attn])
        x = conv_block(x, f)
    outputs = Conv2D(1, (1,1), activation='sigmoid', name='output')(x)
    return Model(inputs=inputs, outputs=outputs)
