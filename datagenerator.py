class Dataset_Segmentation_CSpine3D_7ch:
    """
    Args:
        list_image (str): lists of path to images
        list_mask (str) : lists of path to masks
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    """
    
    def __init__(
            self, 
            x_list,
            y_list,
            augmentation=False,
    ):
        self.x_list = x_list
        self.y_list = y_list
        self.augmentation = augmentation

    def __getitem__(self, index):
        """self.augmentation = augmentation"""
        x_idx = self.x_list[index]
        y_idx = self.y_list[index]

        image = data_load_nii(x_idx,return_array=True)
        image = image_resize_3D(image,img_dep,img_rows,img_cols)
        image = image_windowing(image,1800,400)
        
        image = image_preprocess_float(image)
        image = np.reshape(image,(img_dep,img_rows,img_cols,1))

        mask = data_load_nii(y_idx,return_array=True)
        mask = label_resize_3D(mask,img_dep,img_rows,img_cols)
        mask = to_categorical(mask)
        mask = mask[...,1:].astype('uint8')
        
        if self.augmentation==True:
            image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2],image.shape[3]))
            mask = np.reshape(mask,(1,mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3]))
            image, mask = data_aug(image, mask)
            image = np.reshape(image,(image.shape[1],image.shape[2],image.shape[3],image.shape[4]))
            mask = np.reshape(mask,(mask.shape[1],mask.shape[2],mask.shape[3],mask.shape[4]))
            
        return image, mask
        
    def __len__(self):
        return int(len(self.x_list))

from tensorflow.python.keras.utils.data_utils import Sequence
class Dataloader(Sequence):
    """Load data from dataset and form batches (!!DO NOT MODIFY!!)
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        """ collect batch data"""
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
            
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
