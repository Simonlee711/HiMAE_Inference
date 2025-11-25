


def get_model(cfg: dict):


    if 'himae'.lower() in cfg['model']:
        from utils.model_arch.himae import HiMAE
        return HiMAE(cfg)

    else:
        print('No such type of model...')
        exit()







        
        