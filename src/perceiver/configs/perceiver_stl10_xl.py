from ml_collections import ConfigDict

def C(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)

def get_config():
    return C(
        cuda                = True,
        dataset             = 'stl10',
        image_size          = 96,
        num_classes         = 10,
        result_file         = 'result_perceiver_stl10_xl.yml',

        train=C(
            batch_size          = 32,
            num_epochs          = 200,
            random_aug          = True,
        ),

        optimizer_type      = 'adamw',
        optimizer_args=C(
            lr                  = 3e-4,
        ),

        model_type          = 'perceiver',
        model_args=C(
            input_channels      = 3,        # number of channels for each token of the input
            input_axis          = 2,        # number of axis for input data (2 for images, 3 for video)
            num_freq_bands      = 6,        # number of freq bands, with original value (2 * K + 1)
            max_freq            = 10.,      # maximum frequency, hyperparameter depending on how fine the data is
            depth               = 6,        # depth of net. The shape of the final attention mechanism will be:
                                            #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents         = 48,       # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim          = 96,       # latent dimension
            cross_heads         = 1,        # number of heads for cross attention. paper said 1
            latent_heads        = 8,        # number of heads for latent self attention, 8
            cross_dim_head      = 32,       # number of dimensions per cross attention head
            latent_dim_head     = 64,       # number of dimensions per latent self attention head
            num_classes         = 10,       # output number of classes
            attn_dropout        = 0.3,
            ff_dropout          = 0.3,
            weight_tie_layers   = False,    # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,     # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2         # number of self attention blocks per cross attention
        ),
    )