def define_arch(opt):
    arch_name = opt["type"]
    if arch_name == "AugmentedTransformer":
        from archs.augmentedtransformer import AugmentedTransformer as M
    else:
        raise NotImplementedError(f"{arch_name} is not implemented")
    
    return M(**opt["opt"])