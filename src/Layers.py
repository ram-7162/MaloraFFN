from src.MaloraLayer import MALoRADownProjLayer,DenseLoRADownProjLayer,SymmetricMoEDownProjLayer


def updating_layers(model, r1, r2, alpha, n_experts,layer_range,mode="malora") :
    d_model = model.config.hidden_size        
    d_ffn   = model.config.intermediate_size

    start, end = layer_range

    for idx in range(start, end):
        original_mlp = model.model.layers[idx].mlp
        if mode == "symmetric_moe":
            new_mlp = SymmetricMoEDownProjLayer(original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha)
        elif mode == "lora":
            new_mlp = DenseLoRADownProjLayer(original_mlp, r2, d_model, d_ffn, alpha)
        elif mode == "malora":
            new_mlp = MALoRADownProjLayer(original_mlp, n_experts, r1, r2, d_model, d_ffn, alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        new_mlp.to(original_mlp.down_proj.weight.device)

        model.model.layers[idx].mlp = new_mlp

    return model

