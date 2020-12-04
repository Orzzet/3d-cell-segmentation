import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from apex import amp
from losses import simple_dice_loss3D, WeightedCrossEntropyLoss
from utils import compare_output, metrics, draw_images, plot_epochs

# https://github.com/mcarilli/mixed_precision_references/blob/master/Pytorch_Devcon_2019/devcon_2019_mcarilli_final.pdf

def target_to_one_hot(target):
    temp = torch.reshape(target, (-1,)).long()
    target = torch.zeros([torch.numel(temp), 2])
    target[torch.arange(torch.numel(temp)),temp] = 1
    return target

def train(model, train_loader, valid_loader, model_name, n_epochs = 100, loss_function = 'dice', AMP=True, gpu=True, test_data = None):
    optimizer = optim.Adam(model.parameters(), weight_decay=0.00001)
    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if loss_function == 'wce':
        criterion = WeightedCrossEntropyLoss()
    elif loss_function == 'ce':
        criterion = CrossEntropyLoss()
    else: # DICE
        criterion = nn.Softmax(dim=1)

    exists_best_model = False
    filename = model_name + '.pth'

    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        current_epoch = checkpoint['epochs']
        best_model_state_dict = checkpoint['best_model_state_dict']
        best_optimizer_state_dict = checkpoint['optimizer_state_dict']
        valid_loss_min = checkpoint['valid_loss_min']
        if AMP:
            amp.load_state_dict(checkpoint['amp_state_dict'])
        exists_best_model = True
        del checkpoint
        torch.cuda.empty_cache()
        n_epochs = max(n_epochs - current_epoch, 0)
        print("---")
        print("Modelo entrenado {} epochs, se entrenará {} epochs más para llegar a {}".format(current_epoch, n_epochs, n_epochs + current_epoch))
    else:
        train_losses = []
        valid_losses = []
        current_epoch = 0
        valid_loss_min = np.Inf
        best_model_state_dict = model.state_dict()
        best_optimizer_state_dict = optimizer.state_dict()

    start_training = time.time()
    for epoch in range(current_epoch+1, current_epoch + n_epochs + 1):   
        start_epoch = time.time()
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data, target, correct_cell_count, resized_cell_count in train_loader:
            target = target.squeeze(0)
            # move tensors to GPU if CUDA is available
            if gpu:
                data = Variable(data).cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            datasize = data.size(0)
            del data
            if loss_function in {'wce', 'ce'}:
                if gpu:
                    target = Variable(target).cuda().long()
                loss = criterion(output, target)
            else: # 'dice'
                target = target_to_one_hot(target).float()
                if gpu:
                    target = Variable(target).cuda()
                # calculate the batch loss
                output = output.permute(0,2,3,4,1).contiguous().view(-1,2).float()
                loss = simple_dice_loss3D(criterion(output), target)
            # backward pass: compute gradient of the loss with respect to model parameters
            if AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * datasize
            del target
            del output

        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target, correct_cell_count, resized_cell_count in valid_loader:
            target = target.squeeze(0)
            # move tensors to GPU if CUDA is available
            if gpu:
                data = Variable(data).cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            datasize = data.size(0)
            del data
            if loss_function in {'wce', 'ce'}:
                if gpu:
                    target = Variable(target).cuda().long()
                loss = criterion(output, target)
            else:
                target = target_to_one_hot(target).float()
                if gpu:
                    target = Variable(target).cuda()
                # calculate the batch loss
                output = output.permute(0,2,3,4,1).contiguous().view(-1,2).float()
                loss = simple_dice_loss3D(criterion(output), target)
            del target
            del output
            # update average validation loss
            valid_loss += loss.item() * datasize
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # print training/validation statistics
        print('Epoch: {} Tiempo:{:.0f}s \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, time.time()-start_epoch, train_loss, valid_loss))
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            best_model_state_dict = model.state_dict()
            best_optimizer_state_dict = optimizer.state_dict()
            print('Validation loss decreased. Train loss: {:.6f} Validation Loss: ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss,
            valid_loss_min,
            valid_loss,
            ))
            valid_loss_min = valid_loss
            exists_best_model = True
        if exists_best_model:
            file_obj = {
                'epochs': epoch,
                'best_model_state_dict': best_model_state_dict,
                'best_optimizer_state_dict' : best_optimizer_state_dict,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_loss_min': valid_loss_min,
            }
            if AMP:
                file_obj['amp_state_dict'] = amp.state_dict()
            torch.save(file_obj, filename)

    print("-----")

    plot_epochs(train_losses, valid_losses, model_name)
    if test_data:
        metrics(model, test_data, save=True, model_name=model_name)

    print("Entrenamiento terminado en {:.2f}m".format((time.time() - start_training)/60))