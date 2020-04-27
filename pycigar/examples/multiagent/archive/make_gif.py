import os
import imageio

png_dir = '/home/toanngo/ray_results/result_plot/'
images = []
filenames = sorted(os.listdir(png_dir))
for file_name in filenames:
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave(png_dir + 'result.gif', images, fps=1.5)
"""
print(filenames)
"""
