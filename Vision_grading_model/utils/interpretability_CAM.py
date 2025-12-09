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
import pandas as pd
from Vision_grading_model.model.vision_grading import vision_grading_model
from Vision_grading_model.cnn_visualization.misc_fucntion import preprocess_image
from pathlib import Path

class ProbabilityMapGenerator:
    """
    Generate probability maps for different tortuosity levels as shown in Figure 9 of the paper
    """
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        
    def save_gradients(self, grad):
        self.gradients = grad
        
    def forward_pass_on_resnet(self, x):
        """
        Forward pass through ResNet50 part only
        """
        conv_output = None
        
        # Forward through ResNet50 layers
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
        
    def generate_probability_maps(self, input_image, seg_map, original_img):
        """
        Generate probability maps for all 4 tortuosity levels
        Returns:
        - probability_maps: list of probability maps for each level
        - class_probabilities: probabilities for each class
        """
        # Ensure input requires gradient
        if not input_image.requires_grad:
            input_image.requires_grad = True
            
        # Get features from ResNet50
        conv_output, _ = self.forward_pass_on_resnet(input_image)
        
        # Continue forward pass to get final output
        x = self.model.avgpool(conv_output)
        x = torch.flatten(x, 1)
        model_output = self.model.fc1(x)
        
        # Get probabilities using softmax
        probabilities = F.softmax(model_output, dim=1)
        class_probabilities = probabilities.data.cpu().numpy()[0]
        
        # Generate probability maps for each class
        probability_maps = []
        
        for target_class in range(4):  # 4 tortuosity levels
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
            if self.gradients is None:
                raise RuntimeError("Gradients are None. Check if the hook was properly set.")
                
            guided_gradients = self.gradients.data.cpu().numpy()[0]
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
            
            # Resize to input image size
            input_size = (input_image.shape[2], input_image.shape[3])
            cam_resized = cv2.resize(cam, input_size)
            
            probability_maps.append(cam_resized)
        
        return probability_maps, class_probabilities
    
    def create_visualizations(self, original_img, seg_map, probability_maps, class_probabilities, output_dir, base_name):
        """
        Create visualizations as shown in Figure 9 of the paper
        """
        # Convert images to numpy arrays
        if isinstance(original_img, Image.Image):
            original_array = np.array(original_img)
        else:
            original_array = original_img
            
        if isinstance(seg_map, Image.Image):
            seg_array = np.array(seg_map)
        else:
            seg_array = seg_map
            
        # Ensure seg_array is 2D
        if len(seg_array.shape) == 3:
            seg_array = seg_array[:, :, 0] if seg_array.shape[2] >= 1 else seg_array
        
        # Create output directories
        level_dirs = []
        for level in range(4):
            level_dir = os.path.join(output_dir, f"level_{level+1}")
            os.makedirs(level_dir, exist_ok=True)
            level_dirs.append(level_dir)
        
        overlay_dir = os.path.join(output_dir, "overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        
        # 1. Create simple overlay of original image and segmentation map using cv2.add
        if len(original_array.shape) == 2:  # Grayscale
            original_rgb = cv2.cvtColor(original_array, cv2.COLOR_GRAY2BGR)
        else:
            original_rgb = original_array
            
        # Create segmentation RGB (white color)
        if len(seg_array.shape) == 2:
            seg_rgb = cv2.cvtColor(seg_array, cv2.COLOR_GRAY2BGR)
        else:
            seg_rgb = seg_array
            
        # Simple cv2.add overlay - direct addition without weights
        overlay = cv2.add(original_rgb, seg_rgb)
        overlay_path = os.path.join(overlay_dir, f"{base_name}.png")
        cv2.imwrite(overlay_path, overlay)
        
        # 2. Create probability maps for each level
        for level in range(4):
            prob_map = probability_maps[level]
            
            # Resize probability map to match original image size
            if isinstance(original_img, Image.Image):
                target_size = original_img.size
            else:
                target_size = (original_img.shape[1], original_img.shape[0])
            
            prob_map_resized = cv2.resize(prob_map, target_size)
            
            # Apply colormap to probability map
            prob_map_colored = cv2.applyColorMap((prob_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Create binary mask from segmentation
            seg_binary = (seg_array > 0).astype(np.uint8)
            
            # Apply segmentation mask to probability map (only show probabilities on nerve fibers)
            prob_map_masked = cv2.bitwise_and(prob_map_colored, prob_map_colored, mask=seg_binary)
            
            # Convert original image to BGR if needed
            if len(original_array.shape) == 2:  # Grayscale
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_GRAY2BGR)
            else:
                original_bgr = original_array
            
            # Create background (original image with weight 0.3)
            background = cv2.addWeighted(original_bgr, 0.3, np.zeros_like(original_bgr), 0.7, 0)
            
            # Create foreground (probability map with weight 0.7)
            foreground = cv2.addWeighted(prob_map_masked, 0.7, np.zeros_like(prob_map_masked), 0.3, 0)
            
            # Blend: use background everywhere, but replace segmented regions with blended result
            blended = background.copy()
            
            # Create mask for blending - only blend in segmented regions
            seg_mask_3d = np.stack([seg_binary, seg_binary, seg_binary], axis=2)
            
            # In segmented regions: blend background and foreground
            # In non-segmented regions: keep background only
            blended = np.where(seg_mask_3d > 0, 
                             cv2.addWeighted(background, 0.3, foreground, 0.7, 0),
                             background)
            
            # Convert to uint8
            blended = blended.astype(np.uint8)
            
            # Save the visualization (no text added)
            level_path = os.path.join(level_dirs[level], f"{base_name}.png")
            cv2.imwrite(level_path, blended)
        
        return class_probabilities

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
    Process a specific dataset split (train, val, or test) to generate probability maps
    """
    root_path = os.path.join(data_root, split_name)
    img_dir = os.path.join(root_path, "image")
    seg_dir = os.path.join(root_path, "Predict_label")
    output_dir = os.path.join(root_path, "Probability_Maps")
    
    # Check if input directories exist
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return None
    
    if not os.path.exists(seg_dir):
        print(f"Segmentation directory not found: {seg_dir}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {split_name} dataset...")
    print(f"Input images: {img_dir}")
    print(f"Output probability maps: {output_dir}")
    
    # Get all image files
    img_files = glob.glob(os.path.join(img_dir, "*.png"))
    if not img_files:
        print(f"No PNG images found in {img_dir}")
        return None
    
    print(f"Found {len(img_files)} images to process")
    
    # Initialize probability map generator
    prob_generator = ProbabilityMapGenerator(model, target_layer="layer4")
    
    # Create CSV to store probabilities
    csv_data = []
    
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
            
            # Create input for model (concatenate CCM, Seg Map, and CCM again)
            input_tensor = torch.cat((input_image, input_seg, input_image), dim=1)
            
            # Ensure input tensor requires gradient
            input_tensor.requires_grad_(True)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Generate probability maps for all 4 levels
            probability_maps, class_probabilities = prob_generator.generate_probability_maps(
                input_tensor, seg, img
            )
            
            # Create visualizations
            prob_generator.create_visualizations(
                img, seg, probability_maps, class_probabilities, output_dir, base_name
            )
            
            # Add to CSV data
            csv_data.append({
                "image_name": base_name + ".png",
                "level_1_prob": class_probabilities[0],
                "level_2_prob": class_probabilities[1], 
                "level_3_prob": class_probabilities[2],
                "level_4_prob": class_probabilities[3],
                "predicted_level": np.argmax(class_probabilities) + 1
            })
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save CSV file
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{split_name}_probabilities.csv")
        df.to_csv(csv_path, index=False)
        print(f"Probability CSV saved to: {csv_path}")
        
        return df
    else:
        return None

def main():
    """
    Main function to generate probability maps for all dataset splits
    """
    print("Starting probability map generation...")
    # data_root = "/data/Desktop/BioMiner/Dataset/Grading_task/Activation"  
    data_root = "/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset"  
    model_root = "/data/Desktop/BioMiner/Vision_grading_model/best_weight"

    DATA_NAME = "CORN-LCs/1"

    DATA_PATH = os.path.join(data_root, DATA_NAME)
    MODEL_PATH = os.path.join(model_root, DATA_NAME)
    # Load the pre-trained model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process all dataset splits
    splits = ['train', 'val', 'test']
    all_probabilities = {}
    
    for split in splits:
        print(f"\n{'='*50}")
        df = process_dataset_split(DATA_PATH, split, model)
        if df is not None:
            all_probabilities[split] = df
        print(f"Completed processing {split} dataset")
    
    print(f"\n{'='*50}")
    print("All probability maps generated successfully!")
    print("Probability maps saved in:")
    for split in splits:
        prob_path = os.path.join(DATA_PATH, split, "Probability_Maps")
        print(f"  {split}: {prob_path}")
        print(f"    - level_1, level_2, level_3, level_4 directories")
        print(f"    - overlay directory")
        print(f"    - {split}_probabilities.csv")

if __name__ == '__main__':
    main()