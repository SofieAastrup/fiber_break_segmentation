import ants
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from image_lib import save_nii


def register_pair_transform(fixed, moving):
    f_d1, f_d2, f_d3 = fixed.shape
    m_d1, m_d2, m_d3 = moving.shape

    fixed_rescaled = ants.resample_image(fixed,(f_d1//3, f_d2//3, f_d3//3),True,0)
    moving_rescaled = ants.resample_image(moving,(m_d1//3, m_d2//3, m_d3//3),True,0)

    aff = ants.registration(fixed_rescaled, moving_rescaled, "Affine", aff_random_sampling_rate=0.3)
    # print(aff)
    transformation = aff['fwdtransforms'] #aff['fwdtransforms']
    return transformation[0]


def register_pair(fixed, moving):
    f_d1, f_d2, f_d3 = fixed.shape
    m_d1, m_d2, m_d3 = moving.shape

    fixed_rescaled = ants.resample_image(fixed,(f_d1//3, f_d2//3, f_d3//3),True,0)
    moving_rescaled = ants.resample_image(moving,(m_d1//3, m_d2//3, m_d3//3),True,0)

    aff = ants.registration(fixed_rescaled, moving_rescaled, "Affine", aff_random_sampling_rate=0.3)
    # print(aff)
    transformation = aff['invtransforms'] #aff['fwdtransforms']
    registered = ants.apply_transforms( fixed=fixed, moving=moving, transformlist=transformation )

    return registered, transformation



def plot_reg_result(fixed, moving, output, index=100, filename="output", savefig=False, figsize=(20,10)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.set_title("Fixed image", fontsize=25)
    a1 = ax1.imshow(fixed[:,:,index],cmap='gray')
    ax1.grid(c='b')
    ax2.set_title("Moving image", fontsize=25)
    a2 = ax2.imshow(moving[:,:,index], cmap='gray')
    ax2.grid(c='b')
    ax3.set_title("Registration result", fontsize=25)
    a3 = ax3.imshow(output[:,:,index], cmap='gray')
    ax3.grid(c='b')
    fig.tight_layout()
    if savefig:
        plt.savefig(f"{EXAMPLES}/{filename}.jpg", pad_inches='tight')
    else:
        plt.show()



def register_multiple():
    accepted_transforms = []
    first = 0
    last = len(fnames)-1
    current = last//2

    while first < current or current < last:
        first_path =  os.path.join(PATH, fnames[first])
        last_path = os.path.join(PATH, fnames[current])

        fixed = ants.image_read(first_path) 
        moving =  ants.image_read(last_path) 
        print(f"Registering {fnames[first]} to {fnames[current]}")
        registered_image, transformation = register_pair(fixed, moving)
        print(f"finished registration. Inspect the registration and close the image")
        plot_reg_result(fixed, moving, registered_image)
        print(f"Do you accept the registration? [y/n]")
        accepted = input()
        if accepted == "y":
            print("saving transformation. This may take a while...")
            name = f"{first}-{current}"
            accepted_transforms.append(name)
            # ants.image_write((registered_image), f"{name}.nii.gz", ri=False)
            tx = ants.read_transform(transformation[0])
            ants.write_transform(tx, f"{TRANSFORMS}/{name}.mat")
            first = current
            current = last
        else:
            print(f"Would you like to redo the transformation or try closer images?")
            print(f"r: redo")
            print(f"c: try closer images")
            next_action = input()
            if next_action == "c":
                current = first + (current-first)//2

    return accepted_transforms


parser = argparse.ArgumentParser(description="Registers images in series of gradually deforming images.")
parser.add_argument("folder_path", type=str, help="Path to a folder containing images to be registerred. Assumes that they are ordered alphabetically")
parser.add_argument("--output_folder", type=str, help="Name of the folder where the registerred images and transformations are saved")

args = parser.parse_args()
PATH = args.folder_path
OUT_PATH = ""
if args.output_folder:
    OUT_PATH = args.output_folder
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
fnames = os.listdir(PATH)
fnames.sort()
print(f"registering the images: {fnames}")
EXAMPLES = os.path.join(OUT_PATH, "reg_examples")
TRANSFORMS = os.path.join(OUT_PATH, "transformations")
REGIMS = os.path.join(OUT_PATH, "registered_images")
if not os.path.exists(OUT_PATH):
    os.mkdir(EXAMPLES)
if not os.path.exists(OUT_PATH):
    os.mkdir(TRANSFORMS)
if not os.path.exists(OUT_PATH):
    os.mkdir(REGIMS)

    
transforms = register_multiple()


print("registers remaining images automatically. This may take a while...\n")
im02_path =  os.path.join(PATH, fnames[0])
for f in range(0, len(fnames)):
    fixed_path = im02_path
    if f == 0:
        fixed = ants.image_read(fixed_path) 
        plt.figure(figsize=(20,20))
        plt.imsave(f"{EXAMPLES}/example-im{f}.png", fixed[:,:,100])
        ants.image_write(fixed, f"{REGIMS}/{f}.nii.gz", ri=False)
        print()
        continue
    f_transforms = []
    moving_path = os.path.join(PATH, fnames[f])
    moving = ants.image_read(moving_path)
    print(f"Registering {moving_path}\n")
    for t in transforms:
        prev_t = (t.split('-'))
        if int(prev_t[1]) <= f:
            f_transforms.append(f"{TRANSFORMS}/{t}.mat")
            fixed_path = os.path.join(PATH, fnames[int(prev_t[1])]) #f"{t}.nii.gz"
        else:
            break
    fixed = ants.image_read(fixed_path) 
    if int(prev_t[0]) < f:
        transformation = register_pair_transform(fixed, moving)
        f_transforms.append(transformation)
    if fixed_path != im02_path:
        fixed = ants.image_read(im02_path) 
    if f_transforms:
        registered_image = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=f_transforms)
    plt.figure(figsize=(20,20))
    plt.imsave(f"{EXAMPLES}/example-im{f}.png", registered_image[:,:,100])
    ants.image_write((registered_image), f"{REGIMS}/{f}.nii.gz", ri=False)

    