import sys
sys.path.append(r"/data/Desktop/BioMiner") 
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
import tqdm
import cv2
import skimage.measure as measure
from Vision_grading_model.model.vision_grading import vision_grading_model
from Vision_grading_model.cnn_visualization.misc_fucntion import save_class_activation_images, preprocess_image
from pathlib import Path

def generate_roi(cam, img, seg, threshold=0.7, target_size=(192, 192)):
    """
    Generate ROI images for AuxNet according to the paper
    :param cam: Grad-CAM heatmap
    :param img: original CCM image
    :param seg: segmented nerve fibers
    :param threshold: threshold for binarization (t=0.7 as in paper)
    :param target_size: target size for AuxNet input (192x192 as in paper)
    :return: ROI image for AuxNet
    """
    # Resize CAM to match image size
    if isinstance(img, Image.Image):
        img_size = img.size
    else:
        img_size = (img.shape[1], img.shape[0])
    
    heatmap = cv2.resize(cam, img_size)
    
    # Binarize the heatmap with threshold t=0.7
    binary_heatmap = binary_img(heatmap, threshold)
    
    # Convert images to numpy arrays for processing
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
        
    if isinstance(seg, Image.Image):
        seg_array = np.array(seg)
    else:
        seg_array = seg
    
    # Ensure seg_array is 2D
    if len(seg_array.shape) == 3:
        seg_array = seg_array[:, :, 0] if seg_array.shape[2] >= 1 else seg_array
    
    # Create N-ROI by multiplying binary mask with segmented nerve fibers
    # This follows the paper: "nerve fiber segments within the regions of interest, namely N-ROI"
    n_roi = cv2.bitwise_and(seg_array, seg_array, mask=binary_heatmap)
    
    # Resize to target size for AuxNet (192x192 as specified in paper)
    n_roi_resized = cv2.resize(n_roi, target_size, interpolation=cv2.INTER_NEAREST)
    
    return n_roi_resized

def binary_img(heatmap, threshold):
    """
    Perform thresholding on the cam heatmap to get the ROI region
    """
    if threshold < 1:
        threshold_value = int(threshold * 255)
    else:
        threshold_value = threshold
    
    _, binary_heatmap = cv2.threshold(heatmap.astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)
    return binary_heatmap

def save_cam_heatmap(cam, original_img, output_path, alpha=0.6):
    """
    Save CAM as blended image with original image (0.3 image + 0.7 gradcam)
    :param cam: CAM heatmap (0-255)
    :param original_img: original PIL image for blending
    :param output_path: path to save the blended image
    :param alpha: weight for gradcam (0.7), image weight will be 1-alpha (0.3)
    """
    # Ensure cam is the same size as original image
    if isinstance(original_img, Image.Image):
        target_size = original_img.size
        original_array = np.array(original_img)
    else:
        target_size = (original_img.shape[1], original_img.shape[0])
        original_array = original_img
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, target_size)
    
    # Apply jet colormap to CAM
    cam_jet = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Convert original image to color if it's grayscale
    if len(original_array.shape) == 2:
        original_color = cv2.cvtColor(original_array, cv2.COLOR_GRAY2BGR)
    else:
        original_color = original_array
    
    # Ensure both images have the same data type and range
    original_color = original_color.astype(np.float32)
    cam_jet = cam_jet.astype(np.float32)
    
    # Normalize to 0-1 range
    original_color = original_color / 255.0
    cam_jet = cam_jet / 255.0
    
    # Blend images: 0.3 * original + 0.7 * gradcam
    beta = 1.0 - alpha  # 0.3 for original image
    blended = cv2.addWeighted(original_color, beta, cam_jet, alpha, 0)
    
    # Convert back to 0-255 range
    blended = (blended * 255).astype(np.uint8)
    
    # Save the blended image
    cv2.imwrite(output_path, blended)

class ResNetCAMExtractor():
    """
    Extract features and gradients from ResNet50 part only for CAM generation
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward_pass_on_resnet(self, x):
        """
        Perform a forward pass only on ResNet50 part of the model
        """
        conv_output = None
        
        # Forward through ResNet50 layers only
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        
        # Hook the target layer
        if self.target_layer == "layer4":
            x = self.model.layer4(x)
            x.register_hook(self.save_gradients)
            conv_output = x
        
        return conv_output, x

class GradCam():
    """
    Generate class activation map using only ResNet50 part
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = ResNetCAMExtractor(model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM heatmap using only ResNet50 features
        """
        # Ensure input requires gradient
        if not input_image.requires_grad:
            input_image.requires_grad = True
            
        # Get features from ResNet50
        conv_output, _ = self.extractor.forward_pass_on_resnet(input_image)
        
        # Continue forward pass through the rest of ResNet to get final output
        x = self.model.avgpool(conv_output)
        x = torch.flatten(x, 1)
        model_output = self.model.fc1(x)  # Only ResNet branch output
        
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        
        # Target for backprop
        one_hot = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        one_hot[0][target_class] = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass with specific target
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get hooked gradients
        if self.extractor.gradients is None:
            raise RuntimeError("Gradients are None. Check if the hook was properly set.")
            
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        target = conv_output.data.cpu().numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        cam = np.maximum(cam, 0)
        # Normalize to 0~1
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        cam = np.uint8(cam * 255)
        
        # Resize to input image size
        input_size = (input_image.shape[2], input_image.shape[3])
        cam = cv2.resize(cam, input_size)
        
        return cam

def load_model(model_path):
    """
    Load the pre-trained vision_grading_model
    """
   
    if os.path.isdir(model_path):
        # Find the latest model file in the directory
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth') or f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")
        
        # Sort by modification time and get the latest
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
        latest_model = os.path.join(model_path, model_files[0])
        print(f"Loading model: {latest_model}")
    else:
        # Assume it's a direct path to a model file
        latest_model = model_path
    
    # Load the model
    if torch.cuda.is_available():
        checkpoint = torch.load(latest_model)
    else:
        checkpoint = torch.load(latest_model, map_location='cpu')
    
    # Initialize model with num_class parameter (4 classes as in the paper)
    model = vision_grading_model(num_class=4)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def process_dataset_split(data_root, split_name, model):
    """
    Process a specific dataset split (train, val, or test)
    :param data_root: Root directory of the dataset
    :param split_name: Name of the split (train, val, test)
    :param model: Pre-trained model
    """
    root_path = os.path.join(data_root, split_name)
    img_dir = os.path.join(root_path, "image")
    seg_dir = os.path.join(root_path, "Predict_label")
    roi_output_dir = os.path.join(root_path, "ROI_image")
    cam_output_dir = os.path.join(root_path, "CAM_image")  # New directory for CAM heatmaps
    
    # Check if input directories exist
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(seg_dir):
        print(f"Segmentation directory not found: {seg_dir}")
        return
    
    # Create output directories if they don't exist
    os.makedirs(roi_output_dir, exist_ok=True)
    os.makedirs(cam_output_dir, exist_ok=True)  # Create CAM_image directory
    
    print(f"Processing {split_name} dataset...")
    print(f"Input images: {img_dir}")
    print(f"Output ROI images: {roi_output_dir}")
    print(f"Output CAM images: {cam_output_dir}")
    
    # Get all image files
    img_files = glob.glob(os.path.join(img_dir, "*.png"))
    if not img_files:
        print(f"No PNG images found in {img_dir}")
        return
    
    print(f"Found {len(img_files)} images to process")
    
    for file in tqdm.tqdm(img_files):
        base_name = os.path.basename(file)[:-4]  # Remove .png extension
        seg_file = os.path.join(seg_dir, base_name + ".png")
        
        # Check if segmentation file exists
        if not os.path.exists(seg_file):
            print(f"Segmentation file not found: {seg_file}")
            continue
            
        try:
            # Load and preprocess images
            img = Image.open(file).convert("L")
            seg = Image.open(seg_file).convert("L")
            
            # Preprocess images for model input
            input_image = preprocess_image(img, resize=384)
            input_seg = preprocess_image(seg, resize=384)
            
            # Create input for model (concatenate CCM, Seg Map, and CCM again as in paper)
            # This creates 3 channels: [gray1, seg, gray2] as described
            input_tensor = torch.cat((input_image, input_seg, input_image), dim=1)
            
            # Ensure input tensor requires gradient for Grad-CAM
            input_tensor.requires_grad_(True)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Generate Grad-CAM using only ResNet50 part
            grad_cam = GradCam(model, target_layer="layer4")
            cam = grad_cam.generate_cam(input_tensor)
            
            # Save CAM heatmap (before binarization) as jet color map
            cam_filename = os.path.join(cam_output_dir, base_name + ".png")
            save_cam_heatmap(cam, img, cam_filename)
            
            # Generate ROI for AuxNet (N-ROI with threshold t=0.7)
            roi_image = generate_roi(cam, img, seg, threshold=0.7, target_size=(192, 192))
            
            # Save ROI image for AuxNet
            roi_filename = os.path.join(roi_output_dir, base_name + ".png")
            cv2.imwrite(roi_filename, roi_image)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    """
    Main function to process all dataset splits
    """
    print("Starting ROI image generation for AuxNet...")
    
    # data_root = "/data/Desktop/BioMiner/Dataset/Grading_task/Activation"  
    data_root = "/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset"  
    model_root = "/data/Desktop/BioMiner/Vision_grading_model/best_weight"

    DATA_NAME = "CORN-LCs/1" 
    MODEL_NAME = "CORN-LCs/1" 

    DATA_PATH = os.path.join(data_root, DATA_NAME)
    MODEL_PATH = os.path.join(model_root, MODEL_NAME)
    # Load the pre-trained model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process all dataset splits
    splits = ['test'] # 'train', 'val', 
    
    for split in splits:
        print(f"\n{'='*50}")
        process_dataset_split(DATA_PATH, split, model)
        print(f"Completed processing {split} dataset")
    
    print(f"\n{'='*50}")
    print("All ROI images and CAM heatmaps generated successfully!")
    print("ROI images saved in:")
    for split in splits:
        roi_path = os.path.join(DATA_PATH, split, "ROI_image")
        print(f"  {split}: {roi_path}")
    print("CAM heatmaps saved in:")
    for split in splits:
        cam_path = os.path.join(DATA_PATH, split, "CAM_image")
        print(f"  {split}: {cam_path}")

if __name__ == '__main__':
    main()