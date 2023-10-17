def define_arch(opt):
    arch_name = opt["type"]
    if arch_name == "AugmentedTransformer":
        from models.archs.augmentedtransformer import AugmentedTransformer as M
    elif arch_name == "Transformer":
        from models.archs.transformer import Transformer as M  
    else:
        raise NotImplementedError(f"{arch_name} is not implemented")
    
    return M(**opt["opt"])