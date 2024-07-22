import tensorflow as tf

class Parser:
    def __init__(self, classes, masks_path):
        self.classes = classes
        self.masks_path = masks_path

    def __call__(self, img_filename):
        id = tf.strings.split(tf.strings.split(img_filename, os.sep)[-1], '.')[0]
        masks_tensor = None

        image = tf.io.read_file(img_filename)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if len(self.classes) > 1:
            masks = [ None for _ in self.classes ]

            for class_ in self.classes:
                mask_path = tf.strings.join([self.masks_path, id, f'{class_}.png'], os.sep)

                try:
                    mask = tf.io.read_file(mask_path)
                    mask = tf.io.decode_png(mask, channels=1)
                    mask = tf.image.convert_image_dtype(mask, tf.float32)
                    masks[self.classes.index(class_)] = mask
                except:
                    print(f'O arquivo {mask_path} não foi encontrado!')

                masks_tensor = tf.stack(masks, axis=-1)
                masks_tensor = tf.squeeze(masks_tensor, axis=-2)

        mask_path = tf.strings.join([MASKS_PATH, id, f'{CLASSES[0]}.png'], os.sep)
        mask = tf.io.read_file(mask_path)
        masks_tensor = tf.io.decode_png(mask, channels=1)
        masks_tensor = tf.image.convert_image_dtype(masks_tensor, tf.uint8)

        return image, masks_tensor