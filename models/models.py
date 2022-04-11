
def create_model(opt, weights=None):
    model = None
    from .HG_model import HGModel
    model = HGModel(opt, weights=weights)
    print("model [%s] was created" % (model.name()))
    return model
