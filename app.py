import numpy as np
from lang_sam.utils import draw_image
from PIL import Image
from lang_sam import LangSAM
from heic2png import HEIC2PNG
import os

if __name__ == '__main__':
    heic_img = HEIC2PNG('./IMG_4313.heic', quality=70)  # Specify the quality of the converted image
    heic_img.save()  # The converted image will be saved as `test.png`

model = LangSAM()
image_pil = Image.open("./IMG_4313.png").convert("RGB")
text_prompt = "beanbag"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

masks.shape

labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
image_array = np.asarray(image_pil)
image = draw_image(image_array, masks, boxes, labels)
chosen_mask = Image.fromarray(np.uint8(image)).convert("RGB")
chosen_mask.show()

chosen_mask = np.array(chosen_mask).astype("uint8")
chosen_mask[chosen_mask != 0] = 255
chosen_mask[chosen_mask == 0] = 1
chosen_mask[chosen_mask == 255] = 0
chosen_mask[chosen_mask == 1] = 255

# create a base blank mask
width = 3024    
height = 4032
mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # create an opaque image mask

# Convert mask back to pixels to add our mask replacing the third dimension
pix = np.array(mask)
pix[:, :, 3] = chosen_mask[..., 1]

# Convert pixels back to an RGBA image and display
new_mask = Image.fromarray(pix, "RGBA")
#new_mask.show()
cwd = os.getcwd()
#new_mask.save()
new_mask.save(os.path.join(cwd, "final_mask.png"))
