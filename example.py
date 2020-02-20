import os
from generate_matches import generate_matches_carla, ImageIndex
from display_matches import display_correspondences

if __name__ == '__main__':
    benchmark_folder = '/media/lukas/storage04/benchmarkpublic/GNNET_BENCHMARK_PUBLIC'  # set this to the folder you have downloaded the benchmark to.

    carla_folder = os.path.join(benchmark_folder, 'carla_training_validation')
    matches, img1, img2 = generate_matches_carla(benchmark_folder=carla_folder, all_weathers=True,
                                                 image_index_1=ImageIndex(0, 0, 10), image_index_2=ImageIndex(3, 4, 14),
                                                 use_dso_depths=False)
    display_correspondences(img1, img2, matches, 20)
