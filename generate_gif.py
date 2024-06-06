from PIL import Image
import imageio.v3 as imageio
import fire


def split_image_to_gif(
    image_path: str = "mnist_results.png",
    row_height: int = 28,
    gif_path: str = "result.gif",
    fps: int = 10,
):

    image = Image.open(image_path)
    image_width, image_height = image.size

    num_rows = image_height // row_height

    frames = []
    for i in range(num_rows):
        box = (0, i * row_height, image_width, (i + 1) * row_height)
        frame = image.crop(box)
        frames.append(frame)

    imageio.imwrite(gif_path, frames, duration=1 / fps)

    print(f"GIF saved at {gif_path}")


if __name__ == "__main__":
    fire.Fire(split_image_to_gif)
