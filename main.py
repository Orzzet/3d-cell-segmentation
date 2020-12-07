from utils import draw_images
import torch
import yaml
import sys

from dataset import PATHS, get_train_valid_dataloaders, get_test_dataset
from train import train
from models import initialize_model
from utils import metrics, compare_output, predict

def train_model(dim_size_reduction, target_mode, train_path, valid_path, model_name, gpu, arch, n_epochs, loss_function, AMP):
    train_loader, valid_loader = get_train_valid_dataloaders(dim_size_reduction = dim_size_reduction, target_mode = target_mode, train_path = train_path, valid_path = valid_path)
    model = initialize_model(model_name=model_name, gpu=gpu, arch = arch)
    train(model, train_loader, valid_loader, model_name=model_name, n_epochs=n_epochs, loss_function=loss_function, AMP=AMP, gpu=gpu)

def test_model(model_name, gpu, arch, target_mode, dim_size_reduction, test_path):
    model = initialize_model(model_name=model_name, gpu=gpu, arch=arch)
    data = get_test_dataset(target_mode = target_mode, dim_size_reduction = dim_size_reduction, test_path = test_path)

    metrics(model, data)
    print('-------------------')
    compare_output(model, 1, data, [20, 25, 45, 50])

def predict_model(model_name, gpu, arch, target_mode, dim_size_reduction, input_path, output_path):
    model = initialize_model(model_name=model_name, gpu=gpu, arch=arch)
    data = get_test_dataset(target_mode = target_mode, dim_size_reduction = dim_size_reduction, test_path = input_path)
    predict(model, input_images=[d[0] for d in data], output_path=output_path, gpu=gpu)

if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.full_load(f)
        
        job = config['job']
        dim_size_reduction = config['dim_size_reduction']
        target_mode = config['target_mode']
        gpu = config['gpu']
        if gpu and not torch.cuda.is_available():
            print('CUDA no disponible, se va a utilizar cpu')
            gpu = False
        model_name = config['model_name']
        arch = config['arch']
        AMP = config['AMP']

        if job == 'train':
            train_path = config['train_path']
            valid_path = config['valid_path']
            loss_function = config['loss_function']
            n_epochs = config['n_epochs']
            train_model(dim_size_reduction, target_mode, train_path, valid_path, model_name, gpu, arch, n_epochs, loss_function, AMP)
        elif job == 'test':
            test_path = config['test_path']
            test_model(model_name=model_name, gpu=gpu, arch=arch, target_mode=target_mode, dim_size_reduction=dim_size_reduction, test_path=test_path)
        elif job == 'predict':
            input_path = config['input_path']
            output_path = config['output_path']
            predict_model(model_name=model_name, gpu=gpu, arch=arch, target_mode=target_mode, dim_size_reduction=dim_size_reduction, input_path=input_path, output_path=output_path)