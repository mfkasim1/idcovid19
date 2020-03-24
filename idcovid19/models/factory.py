from idcovid19.models.model1 import Model1

def get_model(modelname, data):
    return {
        "model1": Model1,
    }[modelname.lower()](data)
