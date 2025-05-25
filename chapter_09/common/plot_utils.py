import matplotlib.pyplot as plt


def plot_faces(faces, labels, n_cols=5, display_handler=None):
    """
    Plot faces in a grid with flexible display handling

    Args:
        faces: Array of face images (shape: n_samples X 4096)
        labels: List of labels for each face
        n_cols: Number of columns iin grid
        display_handler: Function that handles display/saving
                         Signature: handler(fig) -> None
    """
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))

    for index, (face, label) in enumerate(zip(faces, labels)):
        ax = fig.add_subplot(n_rows, n_cols, index + 1)
        ax.imshow(face, cmap="gray")
        ax.axis("off")
        ax.set_title(str(label), fontsize=9)

    plt.tight_layout()

    if display_handler:
        display_handler(fig)
    else:
        plt.show()
    plt.close(fig)
