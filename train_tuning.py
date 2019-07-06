def train(x_train_):
    
    train_loss = []
    train_metric = []
    for idx_train in range(len(x_train_)):
        image, label = load_data(x_train_[idx_train],y_train_[idx_train])
        image, label = augmentation_imgaug(image,label)

        idx_batch = 0 # batch index
        batch_sample = int(np.ceil(len(image)/batch_size)) # 1 epoch ≈ batch_size * batch_sample
        
        for idx_batch in range(batch_sample):
            image_batch = image[idx_batch*batch_size:(idx_batch+1)*batch_size].copy()
            label_batch = label[idx_batch*batch_size:(idx_batch+1)*batch_size].copy()
            
            train_loss_tmp,train_metric_tmp = model.train_on_batch(image_batch, label_batch)
            train_loss.append(train_loss_tmp)
            train_metric.append(train_metric_tmp)
    
    train_loss = np.mean(np.array(train_loss))
    train_metric = np.mean(np.array(train_metric))
    return train_loss, train_metric

def tuning(x_tuning_):
    
    tuning_loss = []
    tuning_metric = []
    for idx_tuning in range(len(x_tuning_)):
        image, label = load_data(x_tuning_[idx_tuning],y_tuning_[idx_tuning])
        
        idx_batch = 0 # batch index
        batch_sample = int(np.ceil(len(image)/batch_size)) # 1 epoch ≈ batch_size * batch_sample
        
        for idx_batch in range(batch_sample):
            image_batch = image[idx_batch*batch_size:(idx_batch+1)*batch_size].copy()
            label_batch = label[idx_batch*batch_size:(idx_batch+1)*batch_size].copy()

            tuning_loss_tmp,tuning_metric_tmp = model.evaluate(image_batch, label_batch, verbose=0)
            tuning_loss.append(tuning_loss_tmp)
            tuning_metric.append(tuning_metric_tmp)
            
    tuning_loss = np.mean(np.array(tuning_loss))
    tuning_metric = np.mean(np.array(tuning_metric))
    return tuning_loss, tuning_metric  
