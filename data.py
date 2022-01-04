import torch
import numpy as np

def mnist():
    
    # Dictionaries
    test = {'images': [],
            'labels': []
           }
    train = {'images': [],
            'labels': []
           }
    
    # Load images and labels
    for i in range(0,5):
        file = 'train_' + str(i) + '.npz'
        with np.load(file) as f:
            train["images"].append(f['images'])
            train["labels"].append(f['labels'])

            
    with np.load('test.npz') as f:
        test["images"].append(f['images'])
        test["labels"].append(f['labels'])  
             
    # convert to appropiate dimensions
    ims = np.array(train['images'])
    m, n = ims.shape[2:4]
    train['images'] = ims.reshape(-1, m, n)
    train['labels'] = np.array(train['labels']).flatten()
    
    ims = np.array(test['images'])
    m, n = ims.shape[2:4]
    test['images'] = ims.reshape(-1, m, n)
    test['labels'] = np.array(test['labels']).flatten()
    
    # Convert to data loader by using tensor datasets
    train = torch.utils.data.TensorDataset(torch.Tensor(train['images']), torch.LongTensor(train['labels']))
    train = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    
    test = torch.utils.data.TensorDataset(torch.Tensor(test['images']), torch.LongTensor(test['labels']))
    test = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    return train, test

# plot image frede........
#image, label = next(iter(trainloader))
#import matplotlib.pyplot as plt
#plt.imshow(image[0,:]);
#plt.show()