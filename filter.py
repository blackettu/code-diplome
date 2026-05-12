import os
import shutil
import argparse
from PIL import Image, ImageEnhance

def apply_filter(image, filter_name, factor):
    """Применяет указанный фильтр к изображению и возвращает результат."""
    if filter_name.lower() == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    elif filter_name.lower() == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    elif filter_name.lower() == "color":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    elif filter_name.lower() == "sharpness":
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    else:
        print(f"Фильтр '{filter_name}' не поддерживается.")
        return image


def preview_first_image(images_dir, filter_name, factor):
    """Показывает оригинал и результат применения фильтра для первого найденного изображения."""
    for filename in os.listdir(images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_path = os.path.join(images_dir, filename)
            try:
                image = Image.open(image_path)
                filtered = apply_filter(image, filter_name, factor)

                image.show(title="Оригинал")
                filtered.show(title="С фильтром")

                return  # Предпросмотр только для одного изображения
            except Exception as e:
                print(f"Ошибка предпросмотра файла {filename}: {e}")
                continue
    print("Изображения для предпросмотра не найдены.")


def process_images(input_dir, output_dir, filter_name, factor):
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_path = os.path.join(images_dir, filename)
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Ошибка при открытии файла {filename}: {e}")
                continue

            enhanced_image = apply_filter(image, filter_name, factor)

            base_name, ext = os.path.splitext(filename)
            new_image_name = f"{base_name}_{filter_name}{ext}"
            output_image_path = os.path.join(output_images_dir, new_image_name)
            enhanced_image.save(output_image_path)
            print(f"Сохранено изображение: {output_image_path}")

            label_filename = f"{base_name}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            if os.path.exists(label_path):
                new_label_name = f"{base_name}_{filter_name}.txt"
                output_label_path = os.path.join(output_labels_dir, new_label_name)
                shutil.copy(label_path, output_label_path)
                print(f"Сохранён лейбл: {output_label_path}")
            else:
                print(f"Лейбл для {filename} не найден.")


def main():
    parser = argparse.ArgumentParser(description="Apply a photometric filter to YOLO images.")
    parser.add_argument("--input-dir", required=True, help="Folder with images/ and labels/.")
    parser.add_argument("--output-dir", required=True, help="Output folder for augmented images/labels.")
    parser.add_argument(
        "--filter",
        required=True,
        choices=["contrast", "brightness", "color", "sharpness"],
        help="Filter name.",
    )
    parser.add_argument("--factor", type=float, required=True, help="Filter multiplier.")
    parser.add_argument("--preview", action="store_true", help="Show the first augmented image.")
    args = parser.parse_args()

    if args.preview:
        preview_first_image(os.path.join(args.input_dir, "images"), args.filter, args.factor)

    process_images(args.input_dir, args.output_dir, args.filter, args.factor)


if __name__ == "__main__":
    main()
