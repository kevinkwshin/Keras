def conv3d(layer_input, filters, axis=-1, pooling=True):
    d = layers.Conv3D(filters, (3, 3, 3), padding='same')(layer_input)
    d = layers.BatchNormalization(axis=axis)(d)
    d = layers.Activation('relu')(d)
    d = layers.Conv3D(filters, (3, 3, 3), padding='same')(d)
    d = layers.BatchNormalization(axis=axis)(d)
    d = layers.Activation('relu')(d)
    if pooling == True:
        d_out = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
        return d, d_out
    else:
        return d

def deconv3d(layer_input, skip_input, filters, axis=-1):
    u = layers.Conv3DTranspose(filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(layer_input)
    u = layers.BatchNormalization(axis = axis)(u)
    u = layers.Activation('relu')(u)
    u = layers.concatenate([u, skip_input], axis=axis)
    u = layers.Conv3D(filters, (3, 3, 3), padding='same')(u)
    u = layers.BatchNormalization(axis = axis)(u)
    u = layers.Activation('relu')(u)
    return u

def load_3dunet(input_shape, num_labels, init_filter=32, transition_block=5, noise=0.1):
    """
    load_3dunet
    """
    d0 = layers.Input(shape=input_shape)
    d1 = layers.GaussianNoise(noise)(d0)
    d1_skip, d1_out = conv3d(d1, init_filter)
    d2_skip, d2_out = conv3d(d1_out, init_filter*2)
    d3_skip, d3_out = conv3d(d2_out, init_filter*4)
    d4_skip, d4_out = conv3d(d3_out, init_filter*8)

    if transition_block == 5:
        d5_skip, d5_out = conv3d(d4_out, init_filter*16)
        d6_out = conv3d(d5_out, init_filter*32, pooling=False)
        u1_out = deconv3d(d6_out, d5_skip, init_filter*16)
    elif transition_block == 4:
        u1_out = conv3d(d4_out, init_filter*16, pooling=False)
    else:
        raise Exception("'transition_block' must be 4 or 5. you put ", transition_block)

    u2_out = deconv3d(u1_out, d4_skip, init_filter*8)
    u3_out = deconv3d(u2_out, d3_skip, init_filter*4)
    u4_out = deconv3d(u3_out, d2_skip, init_filter*2)
    u5_out = deconv3d(u4_out, d1_skip, init_filter)

    output_img = layers.Conv3D(num_labels, (1, 1, 1), activation='sigmoid')(u5_out)
    model = Model(inputs=d0, outputs=output_img)
    return model
