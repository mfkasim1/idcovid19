from idcovid19.models.model1 import Model1
from idcovid19.models.model2 import Model2

def get_model(modelname, data):
    return {
        "model1": Model1,
        "model2": Model2,
    }[modelname.lower()](data)
