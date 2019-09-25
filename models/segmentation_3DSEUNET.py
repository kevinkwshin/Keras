
def load_model(input_shape, num_labels, axis=-1, base_filter=32, depth_size=4, se_res_block=True, se_ratio=16, noise=0.2, last_relu=False, atten_gate=False):
    
    def conv3d(layer_input, filters, axis=-1, se_res_block=True, se_ratio=16, down_sizing=True):
        if down_sizing == True:
            layer_input = MaxPooling3D(pool_size=(2, 2, 2))(layer_input)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
        d = InstanceNormalization(axis=axis)(d)
        d = LeakyReLU(alpha=0.3)(d)
        d = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(d)
        d = InstanceNormalization(axis=axis)(d)
        if se_res_block == True:
            se = GlobalAveragePooling3D()(d)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            d = Multiply()([d, se])
            shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(layer_input)
            shortcut = InstanceNormalization(axis=axis)(shortcut)
            d = add([d, shortcut])
        d = LeakyReLU(alpha=0.3)(d)
        return d

    def deconv3d(layer_input, skip_input, filters, axis=-1, se_res_block=True, se_ratio=16, atten_gate=False):
        if atten_gate == True:
            gating = Conv3D(filters, (1, 1, 1), use_bias=False, padding='same')(layer_input)
            gating = InstanceNormalization(axis=axis)(gating)
            attention = Conv3D(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='valid')(skip_input)
            attention = InstanceNormalization(axis=axis)(attention)
            attention = add([gating, attention])
            attention = Conv3D(1, (1, 1, 1), use_bias=False,
                               padding='same', activation='sigmoid')(attention)
            # attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[2],'ax':3})(attention) # error when None dimension is feeded.
            #attention = Lambda(resize_by_axis, arguments={'dim_1':skip_input.get_shape().as_list()[1],'dim_2':skip_input.get_shape().as_list()[3],'ax':2})(attention)
            attention = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(attention)
            attention = UpSampling3D((2, 2, 2))(attention)
            attention = CropToConcat3D(mode='crop')([attention, skip_input])
            attention = Lambda(lambda x: K.tile(
                x, [1, 1, 1, 1, filters]))(attention)
            skip_input = multiply([skip_input, attention])

        u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(layer_input)
        u1 = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
        u1 = InstanceNormalization(axis=axis)(u1)
        u1 = LeakyReLU(alpha=0.3)(u1)
        u1 = CropToConcat3D()([u1, skip_input])
        u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
        u2 = InstanceNormalization(axis=axis)(u2)
        u2 = LeakyReLU(alpha=0.3)(u2)
        u2 = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u2)
        u2 = InstanceNormalization(axis=axis)(u2)
        if se_res_block == True:
            se = GlobalAveragePooling3D()(u2)
            se = Dense(filters // se_ratio, activation='relu')(se)
            se = Dense(filters, activation='sigmoid')(se)
            se = Reshape([1, 1, 1, filters])(se)
            u2 = Multiply()([u2, se])
            shortcut = Conv3D(filters, (3, 3, 3), use_bias=False, padding='same')(u1)
            shortcut = InstanceNormalization(axis=axis)(shortcut)
            u2 = add([u2, shortcut])
        u2 = LeakyReLU(alpha=0.3)(u2)
        return u2

    def CropToConcat3D(mode='concat'):
        def crop_to_concat_3D(concat_layers, axis=-1):
            bigger_input, smaller_input = concat_layers
            bigger_shape, smaller_shape = tf.shape(bigger_input), \
                tf.shape(smaller_input)
            sh, sw, sd = smaller_shape[1], smaller_shape[2], smaller_shape[3]
            bh, bw, bd = bigger_shape[1], bigger_shape[2], bigger_shape[3]
            dh, dw, dd = bh-sh, bw-sw, bd-sd
            cropped_to_smaller_input = bigger_input[:, :-dh,
                                                    :-dw,
                                                    :-dd, :]
            if mode == 'concat':
                return K.concatenate([smaller_input, cropped_to_smaller_input], axis=axis)
            elif mode == 'add':
                return smaller_input+cropped_to_smaller_input
            elif mode == 'crop':
                return cropped_to_smaller_input
        return Lambda(crop_to_concat_3D)

    # it is available only for 1 channel 3D
    def resize_by_axis(image, dim_1, dim_2, ax):
        resized_list = []
        unstack_img_depth_list = tf.unstack(image, axis=ax)
        for i in unstack_img_depth_list:
            # defaults to ResizeMethod.BILINEAR
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2]))
        stack_img = tf.stack(resized_list, axis=ax+1)
        return stack_img

    input_img = Input(shape=input_shape)
    d0 = GaussianNoise(noise)(input_img)
    d1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(d0)
    d1 = InstanceNormalization(axis=axis)(d1)
    d1 = LeakyReLU(alpha=0.3)(d1)
    d2 = conv3d(d1, base_filter*2, se_res_block=se_res_block)
    d3 = conv3d(d2, base_filter*4, se_res_block=se_res_block)
    d4 = conv3d(d3, base_filter*8, se_res_block=se_res_block)

    if depth_size == 4:
        d5 = conv3d(d4, base_filter*16, se_res_block=se_res_block)
        u4 = deconv3d(d5, d4, base_filter*8,
                      se_res_block=se_res_block, atten_gate=atten_gate)
        u3 = deconv3d(u4, d3, base_filter*4,
                      se_res_block=se_res_block, atten_gate=atten_gate)
    elif depth_size == 3:
        u3 = deconv3d(d4, d3, base_filter*4,
                      se_res_block=se_res_block, atten_gate=atten_gate)
    else:
        raise Exception('depth size must be 3 or 4. you put ', depth_size)

    u2 = deconv3d(u3, d2, base_filter*2,
                  se_res_block=se_res_block, atten_gate=atten_gate)
    u1 = ZeroPadding3D(((0, 1), (0, 1), (0, 1)))(u2)
    u1 = Conv3DTranspose(base_filter, (2, 2, 2), strides=(2, 2, 2), use_bias=False, padding='same')(u1)
    u1 = InstanceNormalization(axis=axis)(u1)
    u1 = LeakyReLU(alpha=0.3)(u1)
    u1 = CropToConcat3D()([u1, d1])
    u1 = Conv3D(base_filter, (3, 3, 3), use_bias=False, padding='same')(u1)
    u1 = InstanceNormalization(axis=axis)(u1)
    u1 = LeakyReLU(alpha=0.3)(u1)
    output_img = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', activation='sigmoid')(u1)
    if last_relu == True:
        output_img = ThresholdedReLU(theta=0.5)(output_img)
    model = Model(inputs=input_img, outputs=output_img)
    return model



class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
