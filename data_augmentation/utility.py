import matplotlib.pyplot as plt


def show2(im1, im2):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(im1, cmap="gray")
    ax[1].imshow(im2, cmap="gray")
    plt.show()


def show3(im1, im2, im3):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(im1, cmap="gray")
    ax[1].imshow(im2, cmap="gray")
    ax[2].imshow(im3, cmap="gray")
    plt.show()


def show4(im1, im2, im3, im4):
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(im1, cmap="gray")
    ax[0][1].imshow(im2, cmap="gray")
    ax[1][0].imshow(im3, cmap="gray")
    ax[1][1].imshow(im4, cmap="gray")
    plt.show()
