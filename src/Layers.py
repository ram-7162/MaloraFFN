from src.MaloraLayer import MALoRADownProjLayer


def updating_layers(model, r1, r2, alpha, n_experts) :
    d_model = model.config.hidden_size        
    d_ffn   = model.config.intermediate_size

    for idx in range(8, 24):
        original_mlp = model.model.layers[idx].mlp

        model.model.layers[idx].mlp = MALoRADownProjLayer(original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha)

    return model

