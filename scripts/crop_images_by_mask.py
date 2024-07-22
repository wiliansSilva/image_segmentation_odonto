import os
import math
from PIL import Image

if __name__ == '__main__':
    margin = 50
    data_path = 'datasets/panoramic/yolov8/out4/data/val'
    output_path = 'datasets/panoramic/yolov8/out6/data/val'
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                new_image_path = image_path.replace(data_path, output_path)
                image = Image.open(image_path)
                width, height = image.size

                yolov8_path = image_path.replace('.jpg', '.txt')
                new_yolov8_path = new_image_path.replace('.jpg', '.txt')
                xs = []
                ys = []
                with open(yolov8_path, 'r') as reader:
                    lines = reader.readlines()
                    for line in lines:
                        line = line.split(" ")
                        class_id = int(line.pop(0))
                        for i in range(0, len(line), 2):
                            x = float(line[i])
                            y = float(line[i + 1])
                            xs.append(x)
                            ys.append(y)

                left = math.floor(min(xs) * width) - margin
                top = math.floor(min(ys) * height) - margin
                right = math.floor(max(xs) * width) + margin
                bottom = math.floor(max(ys) * height) + margin

                cropped_image = image.crop((left, top, right, bottom))
                cropped_image.save(new_image_path)

                new_height = bottom - top
                new_width = right - left

                with open(yolov8_path, 'r') as reader:
                    lines = reader.readlines()
                    with open(new_yolov8_path, 'w') as writer:
                        for line in lines:
                            line = line.split(" ")
                            class_id = int(line.pop(0))
                            writer.write(f'{class_id}')
                            for i in range(0, len(line), 2):
                                x = (float(line[i]) * width - left) / new_width
                                y = (float(line[i + 1]) * height - top) / new_height
                                writer.write(f' {x} {y}')

                            writer.write('\n')