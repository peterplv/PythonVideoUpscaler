import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from spandrel import ImageModelDescriptor, ModelLoader


# ПАРАМЕТРЫ
# Папка с исходными изображениями
images_path = "/home/user/frames_in"

# Папка для сохранения результатов
output_path = "/home/user/frames_upscaled"
os.makedirs(output_path, exist_ok=True) # Создаем папку если не существует

# Формат готовых изображений
OUTPUT_FORMAT = "JPG"  # PNG, JPG

# Список исходных изображений
all_files = sorted(
    f for f in os.listdir(images_path)
    if os.path.isfile(os.path.join(images_path, f))
)

# Модель
model_path = "/home/user/models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth"

BATCH_SIZE = 2  # Размер батча
batch_images = []  # Для хранения изображений текущего батча

# Очищать torch.cuda.empty_cache(), True или False, иногда помогает вместить изображения в батч
CLEAN_CACHE = False

# Загрузка модели
model = ModelLoader().load_from_file(model_path)
assert isinstance(model, ImageModelDescriptor)
model.cuda().eval()


def save_image(image_name, output_tensor):
    ''' Преобразование тензора в изображение и сохранение в зависимости от заданного формата '''
    
    output_image = transforms.ToPILImage()(output_tensor.cpu().clamp(0, 1))
    output_image_path = os.path.join(output_path, f"{image_name}.{OUTPUT_FORMAT.lower()}")
    
    fmt = OUTPUT_FORMAT.upper()
    
    if fmt == "PNG":
        output_image.save(output_image_path, format="PNG")
    elif fmt == "JPG":
        output_image.save(output_image_path, format="JPEG", quality=100)
    else:
        raise ValueError(f"Ошибка формата: {OUTPUT_FORMAT!r}")
        

# СТАРТ ОБРАБОТКИ    
for idx, image_in in enumerate(all_files):
    image_path = os.path.join(images_path, image_in)
    image_name = os.path.splitext(image_in)[0]
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()

    batch_images.append({'image_name': image_name, 'input_tensor': input_tensor})
    
    # Если накопился полный батч или это последнее изображение - обрабатываем батч
    if len(batch_images) == BATCH_SIZE or idx == len(all_files) - 1:
        try:
            # Очищаем память Cuda если CLEAN_CACHE = True (иногда помогает вместить изображения в батч)
            if CLEAN_CACHE: torch.cuda.empty_cache()
            
            # Проверяем доступную память перед объединением изображений в батч
            required_memory = sum(item['input_tensor'].element_size() * item['input_tensor'].nelement() for item in batch_images)
            free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            if free_memory < required_memory:
                raise RuntimeError("CUDA out of memory")
    
            batch_tensor = torch.cat([item['input_tensor'] for item in batch_images], dim=0)
            
            # Отправляем модели батч с изображениями
            with torch.no_grad():
                output_tensor = model(batch_tensor)

            # Сохраняем обработанные изображения
            for i, item in enumerate(batch_images):
                image_name = item['image_name']
                save_image(image_name, output_tensor[i])
                
        except RuntimeError as e:
            # Ошибка доступной памяти VRAM
            if "CUDA out of memory" in str(e):
                print("Ошибка доступной памяти, обрабатываем изображения по одному...")
            
                for item in batch_images:
                    image_name = item['image_name']
                    single_tensor = item['input_tensor']
                
                    with torch.no_grad():
                        output_tensor = model(single_tensor)
                    save_image(image_name, output_tensor[0])
                    
            else:
                raise
                
        except Exception as e:  # Любая другая ошибка
            print(f"Ошибка:\n{e}")
        
        finally:  # Очищаем батч
            batch_images.clear()


print("ГОТОВО.")


# Выгружаем модель
del model
torch.cuda.empty_cache()