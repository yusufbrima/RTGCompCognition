import numpy as np 
import pandas as pd
from model import build_model
import config


def train_model(MODELS,dl,input_shape,output_nums):
    df = pd.DataFrame()
    hist = {'model':[], 'scores': []}
    for m in MODELS:
        model =  build_model(m, input_shape, output_nums)
        performance = {f'{model.name}_loss':[],f'{model.name}_accuracy':[],f'{model.name}_val_loss':[],f'{model.name}_val_accuracy':[]}
        print(f"We are training {model.name}")
        print("====================================================================")
        history =  model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.valid_dataset) )
        test_scores = model.evaluate(dl.test_dataset)
        print(test_scores)
        performance[f'{model.name}_loss'].append(history.history['loss'])
        performance[f'{model.name}_accuracy'].append(history.history['accuracy'])
        performance[f'{model.name}_val_loss'].append(history.history['val_loss'])
        performance[f'{model.name}_val_accuracy'].append(history.history['val_accuracy'])
        # performance[f'{model.name}_test_loss'].append(test_scores[0])
        # performance[f'{model.name}_test_accuracy'].append(test_scores[1])
        hist['model'].append(model.name)
        hist['scores'].append(test_scores)
        data = pd.DataFrame(performance)
        df = df.append(data)
    return df,pd.DataFrame(hist)


if __name__ == "__main__":
    pass