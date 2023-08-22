import torch
from skimage.io import imread
from skimage.color import rgb2gray
from torch.utils.data import DataLoader, Dataset
from utils import quantize_down, dct2


class Image_Dataset(Dataset):
    def __init__(self,root='test.png'):
        self.root = root

        img = rgb2gray(imread(root))
        self.imgs = []
        
        for i in range(0,img.shape[0],8):
            for j in range(0,img.shape[1],8):
                self.imgs.append(quantize_down(dct2(img[i:(i+8),j:(j+8)])))
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,index):
        rr =  torch.tensor(self.imgs[index], dtype=torch.float)
        return rr,rr



class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(ae,dataset,epochs=5):

    loader = DataLoader(dataset = dataset,
                                     batch_size = 16,
                                     shuffle = True)

    optimizer = torch.optim.Adam(ae.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
    loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        for (img, _) in loader:
            img = img.reshape(-1,64)
            out = ae(img)
            loss_amount = loss(out, img)
            #actual training steps
            optimizer.zero_grad()
            loss_amount.backward()
            optimizer.step()

def encode(chunk,ae):
    assert chunk.shape == (8,8)
    flat = []
    for i in range(8):
        for j in range(8):
            flat.append(chunk[i][j])
    rr = ae.encoder(torch.FloatTensor(flat)).detach().numpy()
    ret = []
    for i in range(len(rr)):
        ret.append(rr[i])
    return ret

def decode(bts,ae):
    flat = []
    for i in range(8):
        flat.append(float(bts[i]))
    rr = ae.decoder(torch.FloatTensor(flat)).detach().numpy()
    ret = []
    for i in range(len(rr)):
        ret.append(rr[i])
    return ret