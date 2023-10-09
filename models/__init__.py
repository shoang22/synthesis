def define_model(opt):
    model_name = opt["name"]
    if model_name == "retro_seq2seq":
        from models.retro_seq2seq_model import RetroSeq2SeqModel as M
    else:
        raise NotImplementedError(f"{model_name} is not supported")
    
    return M(opt)

