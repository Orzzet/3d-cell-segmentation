from fastai.vision import *
from fastai.callbacks.hooks import *


def camvid():
    # https://towardsdatascience.com/image-segmentation-with-fastai-9f8883cc5b53
    path = untar_data(URLs.CAMVID)
    path_lbl = path / 'labels'
    path_img = path / 'images'
    codes = np.loadtxt(path/'codes.txt', dtype=str)
    get_y_fn = lambda x: path_lbl / '{}_P{}'.format(x.stem, x.suffix)
    size = np.array((720, 960)) // 8
    batch_size = 2
    data = (SegmentationItemList.from_folder(path_img)
            # Where are the images?
            .filter_by_rand(0.1)
            .split_by_fname_file('../valid.txt')
            # How to split in train/valid?
            .label_from_func(get_y_fn, classes=codes)
            # How to find the labels? -> use get_y_func on the file name of the data
            .transform(get_transforms(), size=size, tfm_y=True)
            # Data augmentation? -> Standard transforms; also transform the label images
            .databunch(bs=batch_size, num_workers=0)
            #  we convert to a DataBunch, use a batch size of batch_size,
            .normalize(imagenet_stats))
    # The accuracy in an image segmentation problem is the same as that in any classification problem.
    # Accuracy = no. of correctly classified pixels / total no. of pixels
    # some pixels are labelled as Void and shouldn’t be considered when calculating the accuracy.
    name2id = {v: k for k, v in enumerate(codes)}
    void_code = name2id['Void']

    def acc_camvid(input_, target):
        target = target.squeeze(1)
        mask = target != void_code
        return (input_.argmax(dim=1)[mask] == target[mask]).float().mean()

    metrics = acc_camvid
    wd = 1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
    lr_find(learn)
    # print("recorder plot")
    #learn.recorder.plot()
    learn.fit_one_cycle(10, slice(1e-06,1e-03), pct_start=0.9)
    # learn.fit(10)


if __name__ == '__main__':
    camvid()