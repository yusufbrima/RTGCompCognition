import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
from model import build_model
import config
from utils import Visualize
from dataloader import DataLoader
import time
from tqdm import tqdm

def train_model(MODELS,dl,input_shape,output_nums):
    hist = {'model':[], 'loss': [], 'accuracy': [],'f1_score':[], 'time':[]}
    for j in  tqdm(range(len(MODELS))):
        model =  build_model(MODELS[j], input_shape, output_nums)
        print(f"We are training {model.name}")
        print("====================================================================")
        start = time.time()
        _ =  model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.valid_dataset) )
        stop = time.time()
        test_scores =  model.evaluate(dl.test_dataset)
        y_pred = np.argmax(model.predict(dl.X_test), axis = 1)
        print(test_scores)
        # performance[f'{model.name}_test_loss'].append(test_scores[0])
        # performance[f'{model.name}_test_accuracy'].append(test_scores[1])
        hist['model'].append(model.name)
        hist['loss'].append(test_scores[0])
        hist['accuracy'].append(test_scores[1])
        hist['f1_score'].append(f1_score(dl.y_test, y_pred, average='micro'))
        hist['time'].append(stop - start)
        viz =  Visualize(config.figures['figpath'])
        viz.plot_confusion_matrix(dl.y_test, y_pred,dl.CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
    return pd.DataFrame(hist)


def train_model_variable_representation(MODELS):
    hist = {'model':[], 'loss': [], 'accuracy': [],'f1_score':[], 'time':[], 'sf': []}
    sf = ['PS', 'Melspectogram','MFCC']
    for j in tqdm(range(len(MODELS))):
        for i,s in enumerate(sf):
            if(j == 0):
                dl =  DataLoader(datapath = config.data['file_path'],keepdims=True,pt = .1,make=True)
            else:
                dl =  DataLoader(datapath = config.data['file_path'],keepdims=True,pt = .1,make=False)
            dl =  DataLoader(datapath = config.data['file_path'],keepdims=True,pt = .1,make=True)
            if(i == 0):
                dl.create_tensor_set()
            elif(i == 1) :
                dl.create_tensor_set(n = 1)
            elif(i == 2):
                dl.create_tensor_set(n = 2) 
            model =  build_model(MODELS[j], dl.input_shape, dl.output_nums)
            print(f"We are training {model.name} using {s}")
            print("====================================================================")
            start = time.time()
            _ =  model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.valid_dataset))
            stop = time.time()
            test_scores = model.evaluate(dl.test_dataset)
            y_pred = np.argmax(model.predict(dl.X_test), axis = 1)
            print(test_scores)
            hist['model'].append(model.name)
            hist['loss'].append(test_scores[0])
            hist['accuracy'].append(test_scores[1])
            hist['f1_score'].append(f1_score(dl.y_test, y_pred, average='micro'))
            hist['time'].append(stop - start)
            hist['sf'].append(sf.index(s))
        # viz =  Visualize(config.figures['figpath'])
        # viz.plot_confusion_matrix(dl.y_test, y_pred,dl.CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
    return pd.DataFrame(hist)

def train_model_variable_batchsize(MODELS,dl,input_shape,output_nums):
    hist = {'model':[], 'loss': [], 'accuracy': [],'f1_score':[], 'time':[], 'bs': []}
    for j in tqdm(range(len(MODELS))):
        batchsize = [2**x for x in range(3, 8)]
        for bs in batchsize:
            model =  build_model(MODELS[j], input_shape, output_nums)
            print(f"We are training {model.name} with bs={bs}")
            print("====================================================================")
            start = time.time()
            _ =  model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.valid_dataset), batch_size=bs)
            stop = time.time()
            test_scores = model.evaluate(dl.test_dataset,batch_size=bs)
            y_pred = np.argmax(model.predict(dl.X_test, batch_size=bs), axis = 1)
            print(test_scores)
            hist['model'].append(model.name)
            hist['loss'].append(test_scores[0])
            hist['accuracy'].append(test_scores[1])
            hist['f1_score'].append(f1_score(dl.y_test, y_pred, average='micro'))
            hist['time'].append(stop - start)
            hist['bs'].append(bs)
        # viz =  Visualize(config.figures['figpath'])
        # viz.plot_confusion_matrix(dl.y_test, y_pred,dl.CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
    return pd.DataFrame(hist)

def train_model_variable_epochs(MODELS,dl,input_shape,output_nums):
    hist = {'model':[], 'loss': [], 'accuracy': [],'f1_score':[], 'time':[], 'epoch': []}
    for j in  tqdm(range(len(MODELS))):
        epochs = np.arange(20, 101, 20)
        for e in epochs:
            model =  build_model(MODELS[j], input_shape, output_nums)
            print(f"We are training {model.name}")
            print("====================================================================")
            start = time.time()
            _ =  model.fit(dl.train_dataset, epochs=e, validation_data=(dl.valid_dataset) )
            stop = time.time()
            test_scores = model.evaluate(dl.test_dataset)
            y_pred = np.argmax(model.predict(dl.X_test), axis = 1)
            print(test_scores)
            # performance[f'{model.name}_test_loss'].append(test_scores[0])
            # performance[f'{model.name}_test_accuracy'].append(test_scores[1])
            hist['model'].append(model.name)
            hist['loss'].append(test_scores[0])
            hist['accuracy'].append(test_scores[1])
            hist['f1_score'].append(f1_score(dl.y_test, y_pred, average='micro'))
            hist['time'].append(stop - start)
            hist['epoch'].append(e)
        # viz =  Visualize(config.figures['figpath'])
        # viz.plot_confusion_matrix(dl.y_test, y_pred,dl.CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
    return pd.DataFrame(hist)
def train_model_variable_sequence(MODELS):
    hist = {'model':[], 'loss': [], 'accuracy': [],'f1_score':[], 'time':[], 'second': []}
    for j in  tqdm(range(len(MODELS))):
        seconds =  np.arange(0.5,3.5,0.5)
        for s in seconds:
            dl =  DataLoader(datapath = config.data['file_path'],dur=s,keepdims=True,make=True, crop_dims= (128,128) )
            # dl.load()
            dl.create_tensor_set()

            model =  build_model(MODELS[j],dl.input_shape,len(dl.CLASSES))
            print(f"We are training {model.name}")
            print("====================================================================")
            start = time.time()
            _ =  model.fit(dl.train_dataset, epochs=config.hyperparams['EPOCHS'], validation_data=(dl.valid_dataset) )
            stop = time.time()
            test_scores = model.evaluate(dl.test_dataset)
            y_pred = np.argmax(model.predict(dl.X_test), axis = 1)
            print(test_scores)
            hist['model'].append(model.name)
            hist['loss'].append(test_scores[0])
            hist['accuracy'].append(test_scores[1])
            hist['f1_score'].append(f1_score(dl.y_test, y_pred, average='micro'))
            hist['time'].append(stop - start)
            hist['second'].append(s)
        # viz =  Visualize(config.figures['figpath'])
        # viz.plot_confusion_matrix(dl.y_test, y_pred,dl.CLASSES, save=True,filename=f"{model.name}_plot_confusion_matrix.png" )
    return pd.DataFrame(hist)

if __name__ == "__main__":
    pass