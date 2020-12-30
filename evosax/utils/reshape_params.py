

def flat_to_mlp(flat_params, sizes):
    """ Reshape flat param vector into feedforward/MLP network param dict. """
    pop_size = flat_params.shape[0]
    W1_stop = sizes[0]*sizes[1]
    b1_stop = W1_stop + sizes[1]
    W2_stop = b1_stop + (sizes[1]*sizes[2])
    b2_stop = W2_stop + sizes[2]
    # Reshape params into weight/bias shapes
    params = {"W1": flat_params[:, :W1_stop].reshape(pop_size,
                                                     sizes[1], sizes[0]),
              "b1": flat_params[:, W1_stop:b1_stop],
              "W2": flat_params[:, b1_stop:W2_stop].reshape(pop_size,
                                                            sizes[2], sizes[1]),
              "b2": flat_params[:, W2_stop:b2_stop]}
    return params
