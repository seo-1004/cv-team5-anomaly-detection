# visualize.py

import matplotlib.pyplot as plt
import numpy as np


def show_results(vis_data):
    """
    시각화용 함수
    vis_data = [{ filename, input, recon, heatmap, mask }]
    """

    for data in vis_data:
        filename = data['filename']
        input_img = data['input']
        recon_img = data['recon']
        heatmap = data['heatmap']
        mask = data['mask']

        # Convert [-1,1] → [0,1]
        input_vis = (input_img + 1) / 2
        recon_vis = (recon_img + 1) / 2

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"File: {filename}", fontsize=15)

        axes[0].imshow(input_vis)
        axes[0].set_title("Input (Defect)")
        axes[0].axis("off")

        axes[1].imshow(recon_vis)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        im = axes[2].imshow(heatmap, cmap='jet')
        axes[2].set_title("Error Map")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], shrink=0.8)

        axes[3].imshow(mask, cmap='gray')
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
