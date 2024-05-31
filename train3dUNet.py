"""
Copyright (C) 2023 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import time
import copy

import numpy as np
import torch

from patch_seg import handle_patch_update_in_epoch_step
from model_utils import debug_memory
from loss import get_batch_loss
from viz import save_patches_image
from unet3d import UNet3D #RootPainter3D UNet
import nibabel as nib
from matplotlib import pyplot as plt
from patchify import patchify
from torch.utils.data import Dataset, DataLoader
import schedulefree
from torchvision.transforms import v2

class FiberData(Dataset):
    def __init__(self, inputs, labels, mode="train"):
        # print(f"input sizes {inputs.shape=}, {labels.shape=}")
        # inputs = patchify(inputs, (116, 116, 116), step=82)
        # print(f"patchified {inputs.shape=}")
        # labels = patchify(labels, (116, 116, 116), step=82)
        # print(f"patchified {labels.shape=}")
        # numd1, numd2, numd3, _,_,_ = inputs.shape
        # inputs = inputs.reshape(numd1*numd2*numd3, 116, 116, 116)
        # labels = labels.reshape(numd1*numd2*numd3, 116, 116, 116)
        inputs = (inputs - np.min(inputs))/(np.max(inputs)- np.min(inputs))
        # labels = (inputs - np.min(labels))/(np.max(labels)- np.min(labels))
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
        # print(f"{len(inputs)}")
        # print(f"{inputs.shape}")
    def __len__(self):
        # return len(self.labels)
        if self.mode == "train":
            return 100
        elif self.mode == "val":
            return 50
    def __getitem__(self, idx):
        x1, x2, x3 = np.shape(self.inputs)
        x1_rand = np.random.randint(x1-116)
        x2_rand = np.random.randint(x2-116)
        x3_rand = np.random.randint(x3-116)
        input = self.inputs[x1_rand:x1_rand+116, x2_rand:x2_rand+116, x3_rand:x3_rand+116]
        label = self.labels[x1_rand:x1_rand+116, x2_rand:x2_rand+116, x3_rand:x3_rand+116]
        # print(f"{np.max(self.inputs)=}, {np.max(self.labels)=}")
        # print(f"{self.inputs.shape=}, {self.labels.shape=}")
        # print(f"{input.shape=}, {label.shape=}")
        input = np.expand_dims(input, 0)
        input = torch.from_numpy(np.array(input)).cuda()
        label = label[17:-17,17:-17,17:-17]
        label = np.expand_dims(label, 0)
        label = torch.from_numpy(np.array(label)).cuda()
        return input, label
    
class FiberDataValidation(Dataset):
    def __init__(self, inputs, labels):
        inputs = (inputs - np.min(inputs))/(np.max(inputs)- np.min(inputs))
        # labels = (inputs - np.min(labels))/(np.max(labels)- np.min(labels))
        print(f"input sizes {inputs.shape=}, {labels.shape=}")
        inputs = patchify(inputs, (116, 116, 116), step=82)
        print(f"patchified {inputs.shape=}")
        labels = patchify(labels, (116, 116, 116), step=82)
        print(f"patchified {labels.shape=}")
        numd1, numd2, numd3, _,_,_ = inputs.shape
        inputs = inputs.reshape(numd1*numd2*numd3, 116, 116, 116)
        labels = labels.reshape(numd1*numd2*numd3, 116, 116, 116)
        self.inputs = inputs
        self.labels = labels
        # print(f"{len(inputs)}")
        # print(f"{inputs.shape}")
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        # print(f"{input.shape=}, {label.shape=}")
        input = np.expand_dims(input, 0)
        input = torch.from_numpy(np.array(input)).cuda()
        label = label[17:-17,17:-17,17:-17]
        label = np.expand_dims(label, 0)
        label = torch.from_numpy(np.array(label)).cuda()
        return input, label


def get_nii_array(path):
    img = nib.load(path)
    img = np.array(img.dataobj).astype(np.float32)
    return img

model = UNet3D(1, im_channels=1)
# model.load_state_dict(torch.load('real_sofmodel_2'))
model.cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.0005)


loss_fn = torch.nn.MSELoss()


input = get_nii_array("im02.nii.gz")
print(f"image shape {input.shape=}")
label = get_nii_array("59_edt_labels.nii.gz")
_, a2, _ = input.shape

plt.figure()
plt.imshow(label[:,:,100])
plt.show()

# 45 % test, 30 % val and 25 % test
val_idx = int(np.floor(a2*0.50))
test_idx = int(val_idx+np.floor(a2*0.25))
train_input = input[:, :val_idx, :]
train_label = label[:, :val_idx, :]
val_input = input[:, val_idx:test_idx, :]
val_label = label[:, val_idx:test_idx, :]


    
train_dataset = FiberDataValidation(train_input, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = FiberDataValidation(val_input, val_label)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)


def train_epoch():
    loss_sum = 0
    losses = []
    model.train()
    num_batches = 0
    torch.set_grad_enabled(True)
    #for step, (input, label) in enumerate(zip(inputs, labels)):
    for step, (input, label) in enumerate(train_dataloader): 

        optimizer.zero_grad()
        # print(f"{input.shape=}")
        outputs = model(input)

        loss = loss_fn(outputs, label)
        loss_sum += loss.item()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        num_batches += 1 #input.shape[0]
        
    # print(f"epoch train loss = {loss_sum/len(losses)}")
    
    print(f"{num_batches=}")
    return loss_sum/(num_batches)


best_v_loss = 1000000

input, label = next(iter(train_dataloader))
input = input.cpu().detach().numpy()
label = label.cpu().detach().numpy()
plt.figure()
plt.title("input and label train data sample")
plt.subplot(211)
plt.imshow(input[0, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[0, 0, :, :, 50])
plt.show()

num_epochs = 100
train_losses = []
val_losses = []
for i in range(num_epochs):
    epoch_loss = train_epoch()
    train_losses.append(epoch_loss)
    # print(f"{i} : {epoch_loss=}")
    model.eval()
    v_loss_sum = 0
    num_batches = 0
    with torch.no_grad():
        for j, (v_input, v_label) in enumerate(val_dataloader):
            #?? why use enumerate?
            v_output = model(v_input)
            v_loss = loss_fn(v_output, v_label)
            v_loss_sum += v_loss.item()
            num_batches += 1#v_input.shape[0]
    # print(f"validation loss = {v_loss_sum/num_batches}")
    val_loss = v_loss_sum/num_batches
    print(f"{num_batches=}")
    print(f"epoch {i}: train = {epoch_loss}  val = {val_loss}")
    val_losses.append(val_loss)
    if v_loss_sum/num_batches < best_v_loss:
        model_path = '3d_edt_30_not_random'
        torch.save(model.state_dict(), model_path)
        best_v_loss = v_loss_sum/num_batches
        print("Best validation score!")
    print()
torch.save(model.state_dict(), 'overfit_model_test')
input, label = next(iter(train_dataloader))
output = model(input).cpu().detach().numpy()
label = label.cpu().detach().numpy()

plt.figure()
plt.plot(train_losses, label="training loss")
plt.plot(val_losses, label="validation loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.figure()
plt.title("train data sample")
plt.subplot(211)
plt.imshow(output[0, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[0, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("train data sample")
plt.subplot(211)
plt.imshow(output[1, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[1, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("train data sample")
plt.subplot(211)
plt.imshow(output[2, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[2, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("train data sample")
plt.subplot(211)
plt.imshow(output[3, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[3, 0, :, :, 50])
plt.show()


input, label = next(iter(val_dataloader))
output = model(input).cpu().detach().numpy()
label = label.cpu().detach().numpy()

plt.figure()
plt.title("val data sample")
plt.subplot(211)
plt.imshow(output[0, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[0, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("val data sample")
plt.subplot(211)
plt.imshow(output[1, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[1, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("val data sample")
plt.subplot(211)
plt.imshow(output[2, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[2, 0, :, :, 50])
plt.show()

plt.figure()
plt.title("val data sample")
plt.subplot(211)
plt.imshow(output[3, 0, :, :, 50])
plt.subplot(212)
plt.imshow(label[3, 0, :, :, 50])
plt.show()