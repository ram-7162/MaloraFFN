from src.MaloraLayer import MALoRADownProjLayer


def updating_layers(model, r1, r2, alpha, n_experts,layer_range) :
    d_model = model.config.hidden_size        
    d_ffn   = model.config.intermediate_size

    start, end = layer_range

    for idx in range(start, end):
        original_mlp = model.model.layers[idx].mlp

        new_mlp = MALoRADownProjLayer(
    original_mlp,
    n_experts,
    r1,
    r2,
    d_model,
    d_ffn,
    alpha,
)

        new_mlp.to(original_mlp.down_proj.weight.device)

        model.model.layers[idx].mlp = new_mlp

    return model

