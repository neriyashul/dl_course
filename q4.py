from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import join

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


def display_first_layer_outputs_of_virus_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person1_virus_11.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d')


def display_first_layer_outputs_of_bacteria_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person91_bacteria_448.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d')


def display_second_layer_outputs_of_normal_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "NORMAL"))
    tested_image_name = "IM-0007-0001.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


def display_second_layer_outputs_of_virus_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person1_virus_11.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


def display_second_layer_outputs_of_bacteria_img(model) -> None:
    base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    tested_image_name = "person91_bacteria_448.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    display_layer_outputs_of_image(model, img_path, 'conv2d_1')


# def generate_heat_map(model, img_path: str) -> None:
#     img = load_img(img_path, target_size=(250, 250)).convert('L')
#     img_tensor = img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.
#
#     conv_output = model.get_layer("conv2d_1").output
#     pred_output = model.get_layer("dense_3").output
#     model2 = Model(inputs=model.inputs[0], outputs=[conv_output, pred_output])
#     conv, pred = model2.predict(img_tensor)
#
#     plt.figure(figsize=(16, 16))
#     for i in range(36):
#         plt.subplot(6, 6, i + 1)
#         plt.imshow(conv[0, :,:,i]*255., cmap='jet')
#     plt.show()
#
#     plt.figure(figsize=(16, 16))
#     for i in range(36):
#         plt.subplot(6, 6, i + 1)
#         plt.imshow(img.resize(conv[0,:,:,0].shape), cmap='gray')     # Resize original image to be same as the result filters.
#         plt.imshow(conv[0, :,:,i]*255., cmap='jet', alpha=0.6)
#     plt.show()
#
#     target = np.argmax(pred, axis=0).squeeze()
#     w, b = model2.get_layer("dense_3").weights
#     weights = w[:, target].numpy()
#     conv_sq = conv.squeeze()
#     heatmap1 = np.dot(conv_sq[:,:,:int(conv_sq.shape[2]/2)], weights)
#     heatmap2 = np.dot(conv_sq[:,:,int(conv_sq.shape[2]/2):], weights)
#
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img.resize(heatmap1.shape), cmap='gray')
#     plt.imshow(heatmap1, cmap='jet', alpha=0.5)
#     plt.show()
#
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img.resize(heatmap1.shape), cmap='gray')
#     plt.imshow(heatmap2, cmap='jet', alpha=0.5)
#     plt.show()
#
#
# def generate_heat_map_for_normal(model) -> None:
#     img_path = "C:\\Users\\Raviv\\PycharmProjects\\dl_course\\chest_xray\\test\\NORMAL\\IM-0001-0001.jpeg"
#     generate_heat_map(model, img_path)
#
#
# def generate_heat_map_for_pneumonia(model) -> None:
#     img_path = "C:\\Users\\Raviv\\PycharmProjects\\dl_course\\chest_xray\\test\\PNEUMONIA\\person1_virus_6.jpeg"
#     generate_heat_map(model, img_path)


from skimage.transform import resize
def generate_heat_map(model, original_img, image_array, original_prediction, predicted_class, window_size, stride):
    heatmaps = []
    for y in range(0, image_array.shape[1] - window_size + 1, stride):
        for x in range(0, image_array.shape[2] - window_size + 1, stride):
            print(x,y)
            # Create occluded image
            occluded_img = np.copy(image_array)
            occluded_img[:, y:y + window_size, x:x + window_size, :] = 0

            # if show_occluded_images:
            #     # Convert RGB image to grayscale
            #     image_gray = np.dot(occluded_img[0][..., :3], [0.2989, 0.5870, 0.1140])
            #
            #     # Display the grayscale image using Matplotlib
            #     plt.imshow(image_gray, cmap='gray')
            #     plt.axis('off')
            #     plt.show()

            # Predict on occluded image
            occlusion_prediction = model.predict(occluded_img)

            # Calculate difference in predictions caused by occlusion
            prediction_diff = occlusion_prediction - original_prediction

            # Calculate average difference in predictions
            avg_diff = np.mean(np.abs(prediction_diff))

            # Append average difference to heatmaps
            heatmaps.append(avg_diff)

    # Reshape average differences to match original image dimensions
    heatmap_img = np.array(heatmaps).reshape((int((image_array.shape[1] - window_size) / stride) + 1,
                                              int((image_array.shape[2] - window_size) / stride) + 1))

    # Resize heatmap to match the original image dimensions
    # resized_heatmap = heatmap_img.copy()
    # resized_heatmap.resize((image_array.shape[1], image_array.shape[2]))
    resized_heatmap = resize(heatmap_img, (image_array.shape[1], image_array.shape[2]))

    plt.imshow(original_img)
    plt.show()
    # Plot the original image with heatmap overlay
    plt.imshow(original_img)
    plt.imshow(resized_heatmap, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title(f'Heatmap Prediction: {predicted_class}')
    plt.axis('off')
    plt.show()


def import_image(model, img_path: str):
    img = load_img(img_path, target_size=(model.inputs[0].shape[1], model.inputs[0].shape[2])).convert('L')
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    prediction = model.predict(img_tensor)
    prediction_type = "NORMAL" if prediction < 0.5 else "PNEUMONIA"
    return img, img_tensor, prediction, prediction_type


def run_s(model, show_occluded_images=False):
    # base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
    # tested_image_name = "person91_bacteria_448.jpeg"
    base_tested_dir = join(base_dir, join("test", "NORMAL"))
    tested_image_name = "IM-0007-0001.jpeg"
    img_path = join(base_tested_dir, tested_image_name)
    original_img, image_array, original_prediction, predicted_class = import_image(model, img_path)
    window_size = 3
    stride = 2
    generate_heat_map(model, original_img, image_array, original_prediction, predicted_class, window_size, stride)


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
    display_first_layer_outputs_of_virus_img(q1_a_model)
    display_first_layer_outputs_of_bacteria_img(q1_a_model)
    display_second_layer_outputs_of_normal_img(q1_a_model)
    display_second_layer_outputs_of_virus_img(q1_a_model)
    display_second_layer_outputs_of_bacteria_img(q1_a_model)

    run_s(q1_a_model)
    # # Not working well. if can't improve it - delete it.
    # generate_heat_map_for_normal(q1_a_model)
    # generate_heat_map_for_pneumonia(q1_a_model)


if __name__ == "__main__":
    main()


#
#
# from skimage.transform import resize
# def display_heatmap(model):
#     base_tested_dir = join(base_dir, join("test", "PNEUMONIA"))
#     tested_image_name = "person91_bacteria_448.jpeg"
#     img_path = join(base_tested_dir, tested_image_name)
#
#     img = load_img(img_path, target_size=(model.inputs[0].shape[1], model.inputs[0].shape[2])).convert('L')
#     img_tensor = img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.
#     prediction_result = model.predict(img_tensor)
#
#     filter_size = 3
#     stride_size = 2
#
#     partial_predictions_set = []
#     for row_idx in range(0, img_tensor.shape[1] - filter_size + 1, stride_size):
#         for col_idx in range(0, img_tensor.shape[2] - filter_size + 1, stride_size):
#             print(row_idx, col_idx)
#             copied_tensor = np.copy(img_tensor)
#             copied_tensor[:, row_idx:row_idx + filter_size, col_idx:col_idx + filter_size, :] = 0
#             occlusion_prediction = model.predict(copied_tensor)
#             prediction_diff = occlusion_prediction - prediction_result
#             avg_diff = np.mean(np.abs(prediction_diff))
#             partial_predictions_set.append(avg_diff)
#
#     heatmap_img = np.array(partial_predictions_set).reshape(
#         (int((img_tensor.shape[1] - filter_size) / stride_size) + 1,
#          int((img_tensor.shape[2] - filter_size) / stride_size) + 1)
#     )
#
#     resized_heatmap = resize(heatmap_img, (img_tensor.shape[1], img_tensor.shape[2]))
#
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#
#     plt.imshow(img)
#     plt.imshow(resized_heatmap, cmap='jet', alpha=0.5)
#     plt.colorbar()
#     plt.axis('off')
#     plt.show()
#
