import random
import numpy as np
import cv2


def display_correspondences(img0, img1, correspondences, n_corresp_show, ):
    img0_corr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img1_corr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    try:
        rand_idxs = random.sample(range(correspondences.shape[0]), n_corresp_show)
    except ValueError:
        print("Less correspondences than required.")
        return

    coordinates = np.array(correspondences[rand_idxs], dtype=np.uint16)

    # define color lambda function
    color = lambda: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    for i in range(n_corresp_show):
        marker = np.zeros([10, 10, 3])

        col = color()
        marker[:, :, 0] = col[0]
        marker[:, :, 1] = col[1]
        marker[:, :, 2] = col[2]
        try:
            img0_corr[coordinates[i, 1] - 5: coordinates[i, 1] + 5, coordinates[i, 0] - 5: coordinates[i, 0] + 5,
            :] = marker
            img1_corr[coordinates[i, 3] - 5: coordinates[i, 3] + 5, coordinates[i, 2] - 5: coordinates[i, 2] + 5,
            :] = marker

        except ValueError:
            print("Beyond borders.")

    cv2.imshow("Correspondences", np.concatenate([img0_corr, img1_corr], axis=1))
    cv2.waitKey(20000)
