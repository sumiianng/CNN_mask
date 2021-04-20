import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def show_label_on_image(root, all_preds, train=True,
                        filename=None, fontsize=12):
    if train:
        df = pd.read_csv('train.csv')
    else:
        df = pd.read_csv('test.csv')

    if filename is None:
        sample = np.random.randint(len(df), size=1)
        filename = list(df['filename'][sample])[0]

    df = df[df['filename'] == filename]
    img = Image.open(f"{root}/images/{filename}")

    co_map = {0: "green", 1: "blue", 2: "red"}
    label_map = {0: "good", 1: "none", 2: "bad"}

    draw = ImageDraw.Draw(img)

    for index, xmin, ymin, xmax, ymax in zip(df.index,
                                             df['xmin'], df['ymin'],
                                             df['xmax'], df['ymax']):
        pred = all_preds.argmax(dim=1)[index].item()
        co = co_map[pred]
        label = label_map[pred]

        # text size
        ft = ImageFont.truetype("ariblk.ttf", fontsize)
        left, top, right, bottom = ft.getmask(label).getbbox()
        ascent, descent = ft.getmetrics()

        xstart = xmin + (xmax-xmin-right)/2
        ystart = ymin - bottom

        draw.rectangle((xstart-5, ystart-5, xstart+right+5,
                        ystart+bottom+5), fill=co, width=4)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=co, width=10)
        draw.text((xstart, ystart-descent), label, afill=(0, 0, 0), font=ft)
        plt.imshow(img)
