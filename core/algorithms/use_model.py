import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import keras


def get_loss(model, anchor, positive, negative):
    anchor_representation = model(anchor)
    positive_representation = model(positive)
    negative_representation = model(negative)

    ap_distance = tf.reduce_sum(
        (anchor_representation - positive_representation) ** 2, axis=-1
    )
    an_distance = tf.reduce_sum(
        (anchor_representation - negative_representation) ** 2, axis=-1
    )

    return ap_distance, an_distance


def get_triplet_loss(model, anchor, positive, negative):
    ap_distance, an_distance = get_loss(model, anchor, positive, negative)
    return tf.maximum(ap_distance - an_distance + 0.2, 0)


def read_image(path):
    image = tf.keras.preprocessing.image.load_img(
        path,
        color_mode="rgb",
        target_size=(300, 300),  # Resize for specific model input shape
        interpolation="bilinear",
    )
    image = tf.keras.preprocessing.image.img_to_array(image, dtype="float32")
    image = preprocess_input(np.expand_dims(image.copy(), axis=0))
    return image


def load_model(path):
    model = keras.models.load_model(path, safe_mode=False, compile=False, custom_objects={"tf": tf})
    return model

def main():
    model = load_model("model.h5")

    path_to_anchor = "../../data/test/anchor.png"
    path_to_positive = "../../data/test/positive.png"
    path_to_negative = "../../data/test/negative.png"

    anchor = read_image(path_to_anchor)
    positive = read_image(path_to_positive)
    negative = read_image(path_to_negative)

    anchor_re = model(anchor).numpy()[0]
    positive_re = model(positive).numpy()[0]
    negative_re = model(negative).numpy()[0]

    distance_with_positive = np.sqrt(np.sum((anchor_re - positive_re) ** 2))
    distance_with_negative = np.sqrt(np.sum((anchor_re - negative_re) ** 2))

    print(distance_with_positive)
    print(distance_with_negative)


if __name__ == "__main__":
    main()
