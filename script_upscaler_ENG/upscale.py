import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from spandrel import ImageModelDescriptor, ModelLoader


# OPTIONS
# Folder with source images
images_path = "/home/user/frames_in"

# Folder for saving results
output_path = "/home/user/frames_upscaled"
os.makedirs(output_path, exist_ok=True) # Create folder if missing

# Output image format
OUTPUT_FORMAT = "JPG"  # PNG, JPG

# List of source images in the directory
all_files = sorted(
    f for f in os.listdir(images_path)
    if os.path.isfile(os.path.join(images_path, f))
)

# Model path
model_path = "/home/user/models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth"

BATCH_SIZE = 2  # Batch size
batch_images = []  # Images in current batch

# Use torch.cuda.empty_cache() (True/False); sometimes helps fit images into a batch
CLEAN_CACHE = False

# Model loading
model = ModelLoader().load_from_file(model_path)
assert isinstance(model, ImageModelDescriptor)
model.cuda().eval()


def save_image(image_name, output_tensor):
    ''' Tensor to image conversion and saving to file according to the specified format '''
    
    output_image = transforms.ToPILImage()(output_tensor.cpu().clamp(0, 1))
    output_image_path = os.path.join(output_path, f"{image_name}.{OUTPUT_FORMAT.lower()}")
    
    fmt = OUTPUT_FORMAT.upper()
    
    if fmt == "PNG":
        output_image.save(output_image_path, format="PNG")
    elif fmt == "JPG":
        output_image.save(output_image_path, format="JPEG", quality=100)
    else:
        raise ValueError(f"Format error: {OUTPUT_FORMAT!r}")
        

# START PROCESSING    
for idx, image_in in enumerate(all_files):
    image_path = os.path.join(images_path, image_in)
    image_name = os.path.splitext(image_in)[0]
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()

    batch_images.append({'image_name': image_name, 'input_tensor': input_tensor})
    
    # Process the batch if the batch is full or this is the last image
    if len(batch_images) == BATCH_SIZE or idx == len(all_files) - 1:
        try:
            # Clear CUDA memory if CLEAN_CACHE = True (sometimes helps fit images into a batch)
            if CLEAN_CACHE: torch.cuda.empty_cache()
            
            # Check available GPU memory before merging images into a batch
            required_memory = sum(item['input_tensor'].element_size() * item['input_tensor'].nelement() for item in batch_images)
            free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            if free_memory < required_memory:
                raise RuntimeError("CUDA out of memory")
    
            batch_tensor = torch.cat([item['input_tensor'] for item in batch_images], dim=0)
            
            # Send image batch to model
            with torch.no_grad():
                output_tensor = model(batch_tensor)

            # Save processed images
            for i, item in enumerate(batch_images):
                image_name = item['image_name']
                save_image(image_name, output_tensor[i])
                
        except RuntimeError as e:
            # CUDA out-of-memory error
            if "CUDA out of memory" in str(e):
                print("Out-of-memory error, process images one by one...")
            
                for item in batch_images:
                    image_name = item['image_name']
                    single_tensor = item['input_tensor']
                
                    with torch.no_grad():
                        output_tensor = model(single_tensor)
                    save_image(image_name, output_tensor[0])
                    
            else:
                raise
                
        except Exception as e:  # Any other error
            print(f"Ошибка:\n{e}")
        
        finally:  # Clear batch
            batch_images.clear()


print("DONE.")


# Delete model and clear Cuda cache
del model
torch.cuda.empty_cache()