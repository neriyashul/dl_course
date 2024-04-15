from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import join
from skimage.transform import resize

base_dir = "/Users/neriya.shulman/content/chest_xray"
# base_dir = join(getcwd(), "chest_xray")


def load_q1_a_model():
    return load_model('my_model.keras')


def print_models_layers(model) -> None:
    for idx, layer in enumerate(model.layers):
        print(idx, layer)


def print_specific_layer_in_model_with_relevant_name(model, layer_base_name="conv2d") -> None:
    for idx, layer in enumerate(model.layers):
        if layer_base_name in model.layers[idx].name:
            print("layer: ", idx, model.layers[idx].name)


def print_all_layers_with_weights(model) -> None:
    for idx, layer in enumerate(model.layers):
        if model.layers[idx].get_weights():
            print("layer: ", idx, model.layers[idx].name)
            print("filter weights: ", model.layers[idx].get_weights()[0].shape)
            print("biases weights: ", model.layers[idx].get_weights()[1].shape)
            print()


def get_layer_filters(model, layer_name: str, include_prints) -> list:
    lst = []
    specific_layer = model.get_layer(layer_name)
    filters, biases = specific_layer.get_weights()
    if include_prints:
        print("layer: ", specific_layer.name)
        print("filter weights: ", filters.shape)
        print("output weights: ", biases.shape)
        print()
    for idx2 in range(filters.shape[3]):
        x = filters[:, :, :, idx2]
        lst.append(x)
    return lst


def display_first_layer_filters(filters_list: list) -> None:
    lst_width = 8
    lst_height = int(len(filters_list) / lst_width)
    fig, ax = plt.subplots(lst_height, lst_width, figsize=(12, 6))
    fig.suptitle(f"{len(filters_list)}x{'x'.join((str(b) for b in filters_list[0].shape))} Filters Displays")
    for i in range(lst_height):
        for j in range(lst_width):
            ax[i, j].imshow(filters_list[i * lst_width + j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()


def display_seconds_layer_filters(filters_list: list) -> None:
    lst = []
    for pic in filters_list:
        lst.append(np.concatenate(pic, axis=1))

    fig, ax = plt.subplots(len(lst), 1, figsize=(18, 36))
    fig.suptitle(f"{len(lst)}x{'x'.join((str(b) for b in filters_list[0].shape))} Filters Displays")
    for i, pic in enumerate(lst):
        ax[i].imshow(pic, cmap='gray')
        ax[i].axis('off')
    plt.show()


def display_layer_outputs_of_image(model, img_path: str, layer_name: str) -> None:
    layer_outputs = model.get_layer(layer_name).output
    activation_model = Model(inputs=model.inputs[0], outputs=layer_outputs)

    img = load_img(img_path, target_size=(250, 250)).convert('L')
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    activations = activation_model.predict(img_tensor)
    layer_activation = activations[0]

    width_size = 8
    height_size = int(layer_activation.shape[2] / width_size)

    fig, ax = plt.subplots(height_size, width_size, figsize=(width_size * 3, height_size * 3))
    fig.suptitle(f"Display output of layer {layer_name}")
    for height_idx in range(height_size):
        for width_idx in range(width_size):
            ax[height_idx, width_idx].imshow(layer_activation[:, :, height_idx * width_size + width_idx] * 255.,
                                             cmap='gray')
            ax[height_idx, width_idx].axis('off')
    plt.show()


def display_first_layer_outputs_of_normal_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "NORMAL"))
    tested_image_name = "IM-0007-0001.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d')


def display_first_layer_outputs_of_bacteria_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person141_bacteria_676.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d')


def display_first_layer_outputs_of_virus_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person3_virus_16.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d')


def display_second_layer_outputs_of_normal_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "NORMAL"))
    tested_image_name = "IM-0007-0001.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


def display_second_layer_outputs_of_bacteria_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person141_bacteria_676.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


def display_second_layer_outputs_of_virus_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person3_virus_16.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


def patch_images(orig_image, patch_size, stride):
    patches = []
    for counter, row in enumerate(range(0, orig_image.shape[0] - patch_size + 1, stride)):
        for col in range(0, orig_image.shape[1] - patch_size + 1, stride):
            edited_image = np.array(orig_image)
            edited_image[row: row + patch_size, col: col + patch_size] = 0
            patches.append(edited_image)
    return np.array(patches)


def evaluate_heatmap_from_image(model, image, orig_pred, patch_size, stride):
    patched_images = patch_images(image, patch_size, stride)
    pred_results = model.predict(patched_images)
    mse_results = np.mean(np.square(pred_results - orig_pred), axis=1)
    mse_results_as_img_shape = np.array(mse_results).reshape((
        int((image.shape[0] - patch_size) / stride) + 1,
        int((image.shape[1] - patch_size) / stride) + 1
    ))
    heatmap_img = resize(mse_results_as_img_shape, (image.shape[0], image.shape[1]))
    return heatmap_img


def get_image_and_prediction(model, img_path: str):
    img = load_img(img_path, target_size=(model.inputs[0].shape[1], model.inputs[0].shape[2])).convert('L')
    img = img_to_array(img)
    tensor_img = np.expand_dims(img, axis=0)
    img /= 255.
    prediction = model.predict(tensor_img)
    return img, prediction


def run_heatmap(model):
    img_paths = []
    filenames_normal = ["IM-0093-0001.jpeg", "IM-0007-0001.jpeg", "NORMAL2-IM-0278-0001.jpeg"]
    for filename in filenames_normal:
        img_paths.append(join(join(base_dir, join("test", "NORMAL")), filename))
    filenames_bacteria = ["person141_bacteria_676.jpeg", "person147_bacteria_706.jpeg", "person145_bacteria_696.jpeg"]
    for filename in filenames_bacteria:
        img_paths.append(join(join(base_dir, join("test", "PNEUMONIA")), filename))
    filenames_virus = ["person3_virus_16.jpeg", "person44_virus_93.jpeg", "person45_virus_95.jpeg"]
    for filename in filenames_virus:
        img_paths.append(join(join(base_dir, join("test", "PNEUMONIA")), filename))

    heatmap_images = []
    for img_path in img_paths:
        image, image_prediction = get_image_and_prediction(model, img_path)
        filter_size = 15
        stride_size = 3
        hitmap_image = evaluate_heatmap_from_image(model, image, image_prediction, filter_size, stride_size)
        heatmap_images.append((image, hitmap_image))

    height_size = 3
    width_size = int(len(heatmap_images) / height_size)

    fig, ax = plt.subplots(height_size, width_size, figsize=(width_size * 3, height_size * 3))
    for height_idx in range(height_size):
        for width_idx in range(width_size):
            hitmap_tuple = heatmap_images[height_idx * width_size + width_idx]
            ax[height_idx, width_idx].imshow(hitmap_tuple[0])
            ax[height_idx, width_idx].imshow(hitmap_tuple[1], cmap='jet', alpha=0.5)
            ax[height_idx, width_idx].axis('off')
    plt.show()


def main(include_prints: bool = False) -> None:
    q1_a_model = load_q1_a_model()

    if include_prints:
        print_models_layers(q1_a_model)
        print_specific_layer_in_model_with_relevant_name(q1_a_model, "conv2d")
        print_specific_layer_in_model_with_relevant_name(q1_a_model, "dense")
        print_all_layers_with_weights(q1_a_model)

    display_first_layer_filters(get_layer_filters(q1_a_model, "conv2d", include_prints))
    display_seconds_layer_filters(get_layer_filters(q1_a_model, "conv2d_1", include_prints))
    display_first_layer_outputs_of_normal_img(q1_a_model)
    display_first_layer_outputs_of_bacteria_img(q1_a_model)
    display_first_layer_outputs_of_virus_img(q1_a_model)
    display_second_layer_outputs_of_normal_img(q1_a_model)
    display_second_layer_outputs_of_bacteria_img(q1_a_model)
    display_second_layer_outputs_of_virus_img(q1_a_model)

    run_heatmap(q1_a_model)


if __name__ == "__main__":
    main()
