from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset
import pickle 

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128


# Size of z latent vector (i.e. size of generator input)
nz = 1

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = 0.0005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.2

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0




#class NoisyDataset(Dataset):
#    def __init__(self):

#       with open('dataset/all_bitstrings_4qubits.pkl', 'rb') as outp:
#          data = pickle.load(outp)

#       new_data = []
#       for d in data:
          #d is tuple (original_bitstring, counts_dictionary)
#          listkey =[]
#          for k in d[1].keys():
#              for i in range(d[1][k]):
                 
#                 listkey.append(([int(ch) for ch in d[0]] ,[int(ch) for ch in k]))
#                 new_data.append(listkey)
       #temp_array = np.asarray(new_data)


       
#       self.samples = torch.from_numpy(temp_array)

#    def __len__(self):
#        return len(self.samples)

#    def __getitem__(self, idx):
#        return self.samples[idx]



import jax.numpy as jnp
import jax

### Data Setup
def convert(qstr):
    return np.array(list(map(int, qstr)), dtype=float)
    

def load_data(data):
    N = len(data[0][0])
    tot_counts = jax.tree_util.tree_reduce(lambda x,y: x+y, [d[1] for d in data], initializer=0)
    sigma = np.zeros((tot_counts,N), dtype=float)
    eta = np.zeros((tot_counts,N), dtype=float)
    
    j = 0
    for (k,v) in data:
        _sigma = convert(k)
        for kk,vv in v.items():
            sigma[j:j+vv,:] = _sigma
            eta[j:j+vv,:] = convert(kk)
            j = j + vv

    return jnp.array(sigma), jnp.array(eta)

class NoisyDataset(Dataset):
   def __init__(self):
       with open('dataset/all_bitstrings_4qubits.pkl', 'rb') as outp:
          data = pickle.load(outp)

       self.samples=load_data(data)
   def __len__(self):
       return len(self.samples[0])

   def __getitem__(self, idx):
       x,y = self.samples
       return np.array(x[idx]), np.array(y[idx])


dataset = NoisyDataset()




# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images

real_batch = next(iter(dataloader))
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def G_discretize(G_output):
    #G_output.shape = [batch_size, n_qubits]
     X=torch.ones(G_output.shape)
     Y=torch.zeros(G_output.shape)
     return torch.where(G_output >0.5, X, Y) 

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz+4, 32),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(32, 256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(16,4),
            nn.Softmax()
        )

    def forward(self, input):
        return self.main(input)
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(4,32),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(32, 256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(16,4),
            nn.ReLU(True),
            nn.Linear(4,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device) #read in noiseless state 
 
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        latent=np.concatenate((noise, data[1].to(device)), axis=1)
        
# Generate fake image batch with G
        fake = G_discretize(netG(torch.tensor(latent)))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        writer.add_scalar("G_Loss/train", errG.item(), epoch)
        writer.add_scalar("D_Loss/train", errD.item(), epoch)
        writer.flush()

        # Check how the generator is doing by saving G's output on fixed_noise
#        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#            with torch.no_grad():
#                fake = netG(fixed_noise).detach().cpu()
#            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
writer.close()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('losses.png')
