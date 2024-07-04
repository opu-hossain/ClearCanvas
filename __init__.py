import torch
import numpy as np
import cv2
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image

# Get The Current Directory
currentDir = os.path.dirname(__file__)


# Functions:
# Save Results
    

def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(d_dir + output_name)


# Remove Background From Image (Generate Mask, and Final Results)


def removeBg(imagePath):
    """
    Remove the background from an image and save the result.

    Args:
        imagePath (str): The path to the image file.

    Returns:
        tuple: A tuple containing a message and the filename of the saved image.
            - message (str): A success message if the background removal is successful,
              otherwise an error message.
            - filename (str): The filename of the saved image.
    """
    # Set the directories for inputs and results
    inputs_dir = os.path.join(currentDir, 'static/inputs/')
    results_dir = os.path.join(currentDir, 'static/results/')

    # Ensure directories exist
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Read the image file into a numpy array
    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    # Check if the image is empty
    if len(nparr) == 0:
        return '---Empty image---', None

    # Decode the image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # Return an error message if the image cannot be decoded
        return "---Empty image---", None

    # Correct the filename for saving to inputs directory
    filename = os.path.basename(imagePath)
    input_filename = os.path.splitext(filename)[0] + '.jpg'  # Ensure the extension is .jpg
    cv2.imwrite(os.path.join(inputs_dir, input_filename), img)

    # Process the image
    image = transform.resize(img, (320, 320), mode='constant')

    # Normalize the image
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    # Pass the image through the network and get the predicted mask
    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred - mi) / (ma - mi)
    pred = dn

    # Correct the filename for saving to results directory
    result_filename = os.path.splitext(filename)[0] + '.png'  # Ensure the extension is .png
    save_output(imagePath, result_filename, pred, results_dir, 'image')

    return "---Success---", result_filename

# ------- Load Trained Model --------
print("---Loading Model---")
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
# ------- Load Trained Model --------


print("---Removing Background...")
# ------- Call The removeBg Function --------
