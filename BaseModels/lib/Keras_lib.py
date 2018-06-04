from tensorflow import keras

def crt_MLP(input_size, layer_list, act_func='relu', use_bnorm=False, inp_drop=0):
    """Create a simple MLP"""

    net_input = keras.Input(shape=[input_size])
    x = keras.layers.GaussianDropout(inp_drop)(net_input)

    for nb_nodes in layer_list:
        x = keras.layers.Dense(units=nb_nodes, activation=act_func)(x)

        if use_bnorm:
            x = keras.layers.BatchNormalization()(x)

    out = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.models.Model(inputs=[net_input], outputs=[out])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model

def crt_CNN(input_size, layer_list, act_func='relu', use_bias=False):
    """Create a simple CNN"""

    net_input = keras.Input(shape=input_size)
    x = keras.layers.Conv2D(10, (3, 3), data_format='channels_last', activation=act_func, use_bias=use_bias)(net_input)
    x = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    out = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(inputs=[net_input], outputs=[out])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model
