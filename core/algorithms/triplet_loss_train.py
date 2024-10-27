import os
import random
from collections.abc import Generator
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda
from core.datatypes.triplet import TrainingConfig, TripletPath, Path, TripletBatch, Triplet
import matplotlib.pyplot as plt
import yaml


def read_image(path: Path, target_size: int) -> np.ndarray:
    """
    Read image from path

    :param path:  File path
    :param target_size: target size for image

    :return: image as numpy array
    """
    image = tf.keras.preprocessing.image.load_img(
        path.__repr__(),
        color_mode="rgb",
        target_size=(target_size, target_size),  # Resize for specific model input shape
        interpolation="bilinear",
    )
    image = tf.keras.preprocessing.image.img_to_array(image, dtype="float32")
    return image


def split_dataset(directory: str, split: float = 0.9):
    """
    Split dataset into training and test set with a given split ratio
    \n
    Example of directory structure:
    \n
    data\n
    ├── category_1\n
    │   ├── image_1.jpg\n
    │   ├── image_2.jpg\n
    │   └── ...\n
    ├── category_2\n
    │   ├── image_1.jpg\n
    │   ├── image_2.jpg\n
    │   └── ...\n
    └── ...\n

    :param directory: directory with subfolders as categories
    :param split: ratio for training and test set
    :return:
    """

    folders = [
        name for name in os.listdir(directory) if not name.endswith(".DS_Store")
    ]  # exclude .DS_Store --> MacBook specific
    num_train = int(len(folders) * split)
    random.shuffle(folders)

    train_list, test_list = folders[:num_train], folders[num_train:]

    ### ------------- Only for information -------------
    train_data = []
    for folder in train_list:
        train_data.append(len(os.listdir(os.path.join(directory, folder))))
    if len(train_data) > 0:
        print(
            f"Training data with an average amount of samples per category: {np.mean(train_data)} and std: {np.std(train_data)}"
        )

    test_data = []
    for folder in test_list:
        test_data.append(len(os.listdir(os.path.join(directory, folder))))
    if len(test_data) > 0:
        print(
            f"Test data with an average amount of samples per category: {np.mean(test_data)} and std: {np.std(test_data)}"
        )
    ### -------------  End -------------

    return train_list, test_list


def create_triplets(
    directory: str, categories: List[str], max_files: int = 5
) -> List[TripletPath]:
    """
    Create triplets for training the model. A triplet consists of an anchor, a positive and a negative image.
    \n
    Triplets are the input for the model. The model learns to minimize the distance
    between the anchor and the positive image and maximize the distance between the anchor and the negative image.

    :param directory: path to data
    :param categories: list of categories
    :param max_files: maximum amount of files per category
    :return: list of triplets
    """
    triplets = []

    for category in tqdm(categories):  # For every category
        path = os.path.join(directory, category)
        files = list(os.listdir(path))[:max_files]  # Limit amount of files per category
        num_files = len(files)

        for i in range(num_files):
            for j in range(
                num_files
            ):  # Every file with every other file in the folder exept for itself
                if i == j:
                    continue  # Except for itself
                anchor = Path(path, files[i])  # folder and filename
                positive = Path(path, files[j])  # folder and filename

                neg_category = category  # Chose negative category --> any random category exept for itself
                while (
                    neg_category == category
                ):  # Make sure another category has been chosen
                    neg_category = random.choice(categories)

                neg_files = list(os.listdir(os.path.join(directory, neg_category)))
                neg_file = random.choice(
                    neg_files
                )  # Chose random file from negative folder
                path_to_neg_file = os.path.join(directory, neg_category)
                negative = Path(path_to_neg_file, neg_file)
                triplet = TripletPath(anchor, positive, negative)
                triplets.append(
                    triplet
                )  # Append tripplet to list

    random.shuffle(triplets)
    print(f"Amount of triplets: {len(triplets)}")
    return triplets


def get_batch(
    triplet_list: List[TripletPath],
    config: TrainingConfig,
    preprocessing: bool = False,
) -> Generator[TripletBatch]:
    batch_steps = len(triplet_list) // config.batch_size
    for i in range(batch_steps + 1):
        triplet_batch = []
        j = i * config.batch_size  # Staring index at (last batch size index)
        while j < (i + 1) * config.batch_size and j < len(
            triplet_list
        ):  # As long as the index j starting at (last batch size index) is below the index of ((last batch size index) + 1) * batch_size and as long as the end of the triplet list has not been reached

            current_triplet = Triplet(
                anchor=read_image(triplet_list[j].anchor, config.image_input_size),
                positive=read_image(triplet_list[j].positive, config.image_input_size),
                negative=read_image(triplet_list[j].negative, config.image_input_size),
            )
            triplet_batch.append(current_triplet)
            j += 1

        triplet_batch = TripletBatch(triplets=triplet_batch)
        if preprocessing:
            triplet_batch.preprocess()

        yield triplet_batch


def get_encoder(
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    pretrained_trainable_layers: int = 27,
) -> tf.keras.Model:
    """
    Get pretrained model with additional layers for triplet loss training and return model
    \n
    It uses the EfficientNetV2S model as backbone. The model is pretrained on ImageNet and the last layers are trainable.

    :param input_shape: input shape of the model
    :param pretrained_trainable_layers: amount of trainable layers
    :return: model
    """

    # Backbone Modell auf ImageNet vortrainiert
    pretrained_model = tf.keras.applications.Xception(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )

    # Einfrieren der Layer
    for i in range(len(pretrained_model.layers) - pretrained_trainable_layers):
        pretrained_model.layers[i].trainable = False

    # MLP
    encoder_model = Sequential([
        pretrained_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.05),
        BatchNormalization(),
        Dense(128, activation=None)], name="Encoder_Model")

    encoder_model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    return encoder_model


def evaluate(
    encoder: tf.keras.Model,
    summary_writer: tf.summary.SummaryWriter,
    test_triplet: List[TripletPath],
    mode: str = "",
    step: int = 0,
    config: TrainingConfig = None,
):
    ap_distances, an_distances = [], []

    for anchor, positive, negative in get_batch(
        test_triplet, config=training_config, preprocessing=True,
    ):
        if len(anchor.shape) > 1:
            anchor_representation = encoder(anchor)
            positive_representation = encoder(positive)
            negative_representation = encoder(negative)
            ap_distances.extend(
                tf.reduce_sum(
                    (anchor_representation - positive_representation) ** 2, axis=-1
                ).numpy()
            )
            an_distances.extend(
                tf.reduce_sum(
                    (anchor_representation - negative_representation) ** 2, axis=-1
                ).numpy()
            )

    test_accuracy = np.mean(
        [np.array(ap_distances) < config.margin, np.array(an_distances) > config.margin]
    )
    test_metric = np.mean(np.array(ap_distances) < np.array(an_distances))
    with summary_writer.as_default():
        tf.summary.scalar(f"{mode} Test accuracy", test_accuracy, step)
        tf.summary.scalar(
            f"{mode} Test negative distance bigger than positive distance",
            test_metric,
            step,
        )
        tf.summary.scalar(
            f"{mode} Test Mean anchor positive distance", np.mean(ap_distances), step
        )
        tf.summary.scalar(
            f"{mode} Test Mean anchor negative distance", np.mean(an_distances), step
        )
        tf.summary.scalar(
            f"{mode} Test STD anchor positive distance", np.std(ap_distances), step
        )
        tf.summary.scalar(
            f"{mode} Test STD anchor negative distance", np.std(an_distances), step
        )

    return test_accuracy


def train_triplet_loss_model(data_path: str,
                             training_config: TrainingConfig) -> tf.keras.Model:
    """
    Train model with triplet loss
    \n
    The model is trained with triplet loss. The model learns to minimize the distance between the anchor and the
    positive image and maximize the distance between the anchor and the negative image.
    \n
    The model is saved after each epoch if the accuracy is higher than the previous best accuracy.
    \n
    Example of directory structure:
    \n
    data\n
    ├── category_1\n
    │   ├── image_1.jpg\n
    │   ├── image_2.jpg\n
    │   └── ...\n
    ├── category_2\n
    │   ├── image_1.jpg\n
    │   ├── image_2.jpg\n
    │   └── ...\n
    └── ...\n

    :param data_path: path to data
    :param training_config: training configuration
    :return: trained model
    """

    learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=training_config.initial_learning_rate,
        first_decay_steps=training_config.first_decay_steps,
        t_mul=training_config.t_mul,
        m_mul=training_config.m_mul,
        alpha=training_config.alpha  #
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_list, test_list = split_dataset(data_path, split=training_config.split_value_for_dataset)
    train_triplet = create_triplets(data_path, train_list, max_files=training_config.max_images_num_per_category)
    test_triplet = create_triplets(data_path, test_list, max_files=training_config.max_images_num_per_test_category)

    encoder = get_encoder(input_shape=(training_config.image_input_size, training_config.image_input_size, 3), pretrained_trainable_layers=27)
    encoder.compile()
    encoder.summary()

    os.makedirs(training_config.save_path, exist_ok=True)

    log_dir = os.path.join(training_config.save_path, "logs/" + training_config.name)

    summary_writer = tf.summary.create_file_writer(log_dir)
    step = 0
    max_acc = 0

    for epoch in tqdm(range(1, training_config.epochs + 1)):
        for anchor, positive, negative in get_batch(
            train_triplet,config=training_config, preprocessing=True
        ):
            if len(anchor.shape) > 1:
                with tf.GradientTape() as tape:
                    anchor_representation = encoder(anchor)
                    positive_representation = encoder(positive)
                    negative_representation = encoder(negative)

                    ap_distance = tf.reduce_sum(
                        (anchor_representation - positive_representation) ** 2, axis=-1
                    )
                    an_distance = tf.reduce_sum(
                        (anchor_representation - negative_representation) ** 2, axis=-1
                    )

                    loss = tf.reduce_mean(
                        tf.maximum(ap_distance - an_distance + training_config.margin, 0.0)
                    )
                gradients = tape.gradient(loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

                train_accuracy = np.mean(
                    [np.array(ap_distance) < training_config.margin, np.array(an_distance) > training_config.margin]
                )
                train_test_metric = np.mean(np.array(ap_distance) < np.array(an_distance))
                with summary_writer.as_default():
                    tf.summary.scalar(f"Train accuracy", train_accuracy, step)
                    tf.summary.scalar(
                        f"Train negative distance bigger than positive distance",
                        train_test_metric,
                        step,
                    )
                    tf.summary.scalar(
                        f"Train Mean anchor positive distance", np.mean(ap_distance), step
                    )
                    tf.summary.scalar(
                        f"Train Mean anchor negative distance", np.mean(an_distance), step
                    )
                    tf.summary.scalar(
                        f"Train STD anchor positive distance", np.std(ap_distance), step
                    )
                    tf.summary.scalar(
                        f"Train STD anchor negative distance", np.std(an_distance), step
                    )

                    tf.summary.scalar("Loss", loss, step)

                if step % 100 == 0:
                    test_accuracy = evaluate(
                        encoder=encoder,
                        summary_writer=summary_writer,
                        test_triplet=test_triplet,
                        config=training_config,
                        step=step
                    )

                    if test_accuracy >= max_acc:
                        max_acc = test_accuracy
                        encoder.save(
                            f"{training_config.save_path}/best_encoder_{epoch}_step_acc_{str(round(max_acc, 4)).replace('.', '_')}.keras"
                        )

                step += 1
    test_model(encoder, test_triplet[0], training_config)
    return encoder


def convert_triplet_path_to_triplet(triplet_path: TripletPath, target_size) -> Triplet:
    return Triplet(
        anchor=read_image(triplet_path.anchor, target_size),
        positive=read_image(triplet_path.positive, target_size),
        negative=read_image(triplet_path.negative, target_size),
    )


def compute_distance(anchor, comparison):
    return np.sqrt(np.sum((anchor - comparison) ** 2))


def test_model(model, triplet_path: TripletPath, config: TrainingConfig):
    # Todo: change to the model config
    triplet = convert_triplet_path_to_triplet(triplet_path, config.image_input_size)
    triplet.preprocess()
    anchor_rep = model(np.expand_dims(triplet.anchor.copy(), axis=0)).numpy()[0]

    positive_rep = model(np.expand_dims(triplet.positive.copy(), axis=0)).numpy()[0]

    negative_rep = model(np.expand_dims(triplet.negative.copy(), axis=0)).numpy()[0]

    distance_with_positive = compute_distance(anchor_rep, positive_rep)
    distance_with_negative = compute_distance(anchor_rep, negative_rep)

    plt.figure()
    plt.suptitle("Result")
    plt.subplot(3, 1, 1)
    plt.title("Anchor")
    plt.imshow(cv2.imread(triplet_path.anchor.__repr__())[:, :, ::-1])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.title(f"Distance to positive: {distance_with_positive:.4f}")
    plt.imshow(triplet.positive)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 3)
    plt.title(f"Distance to negative: {distance_with_negative:.4f}")
    plt.imshow(cv2.imread(triplet_path.negative.__repr__())[:, :, ::-1])
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def read_yaml_training_config(file_path: str) -> TrainingConfig:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return TrainingConfig(**config)

if __name__ == "__main__":
    training_config = read_yaml_training_config("../../training_config.yaml")
    encoder = train_triplet_loss_model("../../data/train", training_config)
    encoder.save("model.keras")
    print("Training finished!")
