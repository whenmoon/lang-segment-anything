#import numpy as np
#from lang_sam.utils import draw_image
#from PIL import Image
#from lang_sam import LangSAM
#from heic2png import HEIC2PNG
#import os

#if __name__ == '__main__':
#    heic_img = HEIC2PNG('./IMG_4313.heic', quality=70)  # Specify the quality of the converted image
#    heic_img.save()  # The converted image will be saved as `test.png`

#model = LangSAM()
#image_pil = Image.open("./IMG_4313.png").convert("RGB")
#text_prompt = "beanbag"
#masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

#masks.shape

#labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
#image_array = np.asarray(image_pil)
#image = draw_image(image_array, masks, boxes, labels)
#chosen_mask = Image.fromarray(np.uint8(image)).convert("RGB")
#chosen_mask.show()

#chosen_mask = np.array(chosen_mask).astype("uint8")
#chosen_mask[chosen_mask != 0] = 255
#chosen_mask[chosen_mask == 0] = 1
#chosen_mask[chosen_mask == 255] = 0
#chosen_mask[chosen_mask == 1] = 255

## create a base blank mask
#width = 3024    
#height = 4032
#mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # create an opaque image mask

## Convert mask back to pixels to add our mask replacing the third dimension
#pix = np.array(mask)
#pix[:, :, 3] = chosen_mask[..., 1]

## Convert pixels back to an RGBA image and display
#new_mask = Image.fromarray(pix, "RGBA")
##new_mask.show()
#cwd = os.getcwd()
##new_mask.save()
#new_mask.save(os.path.join(cwd, "final_mask.png"))

import numpy as np
from lang_sam.utils import draw_image
from PIL import Image
from lang_sam import LangSAM
from heic2png import HEIC2PNG
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import os
import sys

os.environ['CURL_CA_BUNDLE'] = ''

def draw_image(image, masks, boxes, labels, alpha=1):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['white'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

serverImageUploadPath = sys.argv[1]
target = sys.argv[2]
MASK_FILE_NAME_POSTFIX = sys.argv[3]

#heic_img = HEIC2PNG(serverImageUploadPath + '.heic', quality=50)  # Specify the quality of the converted image
#heic_img.save(serverImageUploadPath + '.png')

# Open the paletted image
imageRGB = Image.open(serverImageUploadPath + '.png')

# Convert to RGB
rgb_image = imageRGB.convert('RGB')

# Save the new image
rgb_image.save(serverImageUploadPath + '.png')

model = LangSAM()
image_pil = Image.open(serverImageUploadPath + '.png').convert("RGB")
text_prompt = target
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

mask_im = Image.new('RGB', size=(image_pil.size))
image_array = np.asarray(mask_im)
image = draw_image(image_array, masks, [], [])
chosen_mask = Image.fromarray(np.uint8(image)).convert("RGB")

tmp = list()
for item in chosen_mask.getdata():
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        # this 0 should mean the pixel is transparent
        # print(item)
        tmp.append((255, 255, 255, 0))
    else:
        tmp.append(item)

img = Image.new(mode="RGBA", size=chosen_mask.size)
img.putdata(tmp)
mask_path = serverImageUploadPath + MASK_FILE_NAME_POSTFIX
img.save(mask_path)
