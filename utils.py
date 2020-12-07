import torch
import torch.nn as nn
import kornia.utils as Ku
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from skimage import measure

def full_image_output(test_data, index, model, patch_og_target=False):
    x_reduction, y_reduction, z_reduction = test_data.dim_size_reduction
    first_slice = index*(x_reduction * y_reduction * z_reduction)
    last_slice = (index+1)*(x_reduction * y_reduction * z_reduction)
    filename, _ = test_data.files[first_slice]
    image, target, _, _ = test_data[0]
    full_output = torch.zeros((image.shape[1]*x_reduction, image.shape[2]*y_reduction, image.shape[3]*z_reduction))
    if patch_og_target:
        og_target = torch.zeros((target.shape[1]*x_reduction, target.shape[2]*y_reduction, target.shape[3]*z_reduction))
    for slice_number, index_file in enumerate(range(first_slice, last_slice)):
        _, ijk = test_data.files[index_file]
        i, j, k = ijk
        image, target, _, _ = test_data[index_file]
        c, z, x, y = image.shape
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        image = image.cuda()
        output = model(image)
        del image
        _, preds_tensor = torch.max(output, 1)
        full_output[k*z:(k+1)*z, i*x:(i+1)*x, j*y:(j+1)*y] = preds_tensor[0]
        if patch_og_target:
            og_target[k*z:(k+1)*z, i*x:(i+1)*x, j*y:(j+1)*y] = target[0]
        del target
    if patch_og_target:
        return full_output, filename, og_target
    return full_output, filename

def compare_output(model, index, dataset, Zs):
    image, target, correct_cell_count, resized_cell_count = dataset[index]
    output = model(image.unsqueeze(0).cuda()).cpu()
    pred = torch.max(output, 1)[1]
    pred_colored, pred_cell_count = measure.label(pred.squeeze(0), return_num=True)
    conf_matrix = Ku.confusion_matrix(pred.reshape((1, -1)), target.reshape((1, -1)), 2, normalized=True)
    miou = Ku.mean_iou(pred.reshape((1, -1)), target.reshape((1, -1)), 2)
    print("Células etiquetado original: {}\t Células etiquetado reescalado: {} \t Células predicción: {}".format(
    correct_cell_count, resized_cell_count, pred_cell_count
    ))
    print("Matriz de confusión por clases de esta imagen: \n{}".format(conf_matrix))
    print("IoU de esta imagen: \n{}".format(miou))
    print(pred_colored.shape)
    draw_images([image[0], measure.label(target[0]), pred_colored], Zs)
    torch.cuda.empty_cache()
    return pred, output


def metrics(model, dataset, save=False, model_name=None):
    if model_name:
        filename = model_name + '.pth'
    model.cuda()
    preds, targets, correct_cell_count, resized_cell_count = zip(*
                         [(torch.max(model(image.unsqueeze(0).cuda()).cpu(), 1)[1], target, correct_cc[0], resized_cc[0]) 
                          for image, target, correct_cc, resized_cc in dataset]
                               )
    preds_cell_count = [measure.label(pred.squeeze(0), return_num=True)[1] for pred in preds]
    preds = torch.cat(preds, dim=1)
    targets = torch.cat(targets, dim=1)
    preds = preds.reshape((1, -1))
    targets = targets.reshape((1, -1))
    conf_matrix = Ku.confusion_matrix(preds, targets, 2, normalized=True)[0]
    miou = Ku.mean_iou(preds, targets, 2)[0]
    print("Matriz de confusión por clases:\n {}".format(conf_matrix))
    print("mean IoU:\n {}".format(miou))
    print("Células etiquetado original:\n {} \n Células etiquetado reescalado: \n {}\n Células predicción:\n {}".format(
    correct_cell_count, resized_cell_count, preds_cell_count
    ))
    if save:
        checkpoint = torch.load(model_name + '.pth')
        checkpoint['test_conf_matrix'] = conf_matrix
        checkpoint['test_miou'] = miou
        AMP = 'amp_state_dict' in checkpoint
        if AMP:
            amp_state_dict = checkpoint['amp_state_dict']
        torch.save(checkpoint, filename)
    torch.cuda.empty_cache()
    return conf_matrix, miou

def metrics_prediction(prediction, target):
    prediction, target = torch.tensor(prediction, dtype=torch.int16), torch.tensor(target)
    miou = Ku.mean_iou(prediction.reshape((1,-1)), target.reshape((1,-1)), 2)[0]
    pred_colored, pred_cell_count = measure.label(prediction, return_num=True)
    target_colored, target_cell_count = measure.label(target, return_num=True)
    print("mean IoU:\n {}".format(miou))
    print("Células etiquetado reescalado:\n {} \n Células predicción:\n {}".format(
    target_cell_count, pred_cell_count
    ))
    draw_images([target_colored, pred_colored], [20, 25, 40, 45])
        
def draw_images(images, Zs):
    imshow([image[Z,:,:] for image in images for Z in Zs])

def imshow(imgs):
    if len(imgs) == 1:
        plt.matshow(imgs[0])
    else:
        nrows = 1+((len(imgs) - 1) // 4)
        ncols = min([len(imgs), 4])
        f, axarr = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        f.set_figwidth(4*ncols)
        f.set_figheight(4*nrows)
        for i, img in enumerate(imgs):
            axarr[i//4][i % 4].matshow(img)

def plot_epochs(train_losses, valid_losses, model_name):
    valid_min = np.nanmin(valid_losses)
    min_index = np.where(valid_losses == valid_min)[0][0]
    train_min = train_losses[min_index]
    print("Mejor valor de validación: {}\t Valor de entrenamiento: {}\t iteracion: {}\n Último valor de validación: {}\t Valor de entrenamiento: {}\t iteracion: {}".format(
           valid_min, train_min, min_index,
           valid_losses[-1], train_losses[-1], len(train_losses)
    ))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.title(model_name)
    plt.ylabel('Loss')
    plt.ylim([0,1])
    plt.xlabel('Epochs')
    plt.legend(frameon=False)
