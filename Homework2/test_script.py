import utils
import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from implementation_script import GeneralizedHoughTransform, nonMaxSuprression, calcBinaryMask


def mainProgram():
    # Load query image and template
    query = cv2.imread("data/query.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("data/template.jpg", cv2.IMREAD_GRAYSCALE)

    # Visualize images
    utils.show(query)
    utils.show(template)

    # Create search space and compute GHT
    angles = np.linspace(0, 360, 36)
    scales = np.linspace(0.9, 1.3, 10)
    ght = GeneralizedHoughTransform(query, template, angles, scales)

    # extract votes (correlation) and parameters
    votes, thetas, s = zip(*ght)

    # Visualize votes
    print("Hough votes")
    votes = np.stack(votes).max(0)
    plt.imshow(votes)
    plt.show()

    # nonMaxSuprression
    print("Filtered Hough votes")
    votes = nonMaxSuprression(votes, 20)
    plt.imshow(votes)
    plt.show()

    # Visualize n best matches
    n = 10
    coords = zip(*np.unravel_index(np.argpartition(votes, -n, axis=None)[-n:], votes.shape))
    vis = np.stack(3 * [query], 2)
    print("Detected Positions")
    for y, x in coords:
        print(x, y)
        vis = cv2.circle(vis, (x, y), 10, (255, 0, 0), 2)
    utils.show(vis)


def testGHT():
    # Load Images
    query = cv2.imread("data/query.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("data/template.jpg", cv2.IMREAD_GRAYSCALE)

    # GHT with search space
    angles = np.linspace(0, 360, 36)
    scales = np.linspace(0.9, 1.3, 10)
    ght = GeneralizedHoughTransform(query, template, angles, scales)

    # Visualize GHT votes
    votes, thetas, s = zip(*ght)
    votes = np.stack(votes).max(0)
    plt.imshow(votes)
    plt.show()

    # Visualize filtered points
    votes = nonMaxSuprression(votes, 20)
    plt.imshow(votes)
    plt.show()

    # Extract n points wiht highest voting score
    n = 10
    coords = list(zip(*np.unravel_index(np.argpartition(votes, -n, axis=None)[-n:], votes.shape)))
    vis = np.stack(3 * [query], 2)
    for y, x in coords:
        vis = cv2.circle(vis, (x, y), 10, (255, 0, 0), 2)
    utils.show(vis)

    # Compare with ground-truth centroids
    f = open("centroids.txt", "r")
    centroids = f.read()
    f.close()
    centroids = centroids.split("\n")[:-1]
    centroids = [centroid.split() for centroid in centroids]
    centroids = np.array([[int(centroid[0]), int(centroid[1])] for centroid in centroids])

    # Visualize centroids
    vis = np.stack(3 * [query], 2)
    for x, y in centroids:
        vis = cv2.circle(vis, (x, y), 10, (255, 0, 0), 2)
    utils.show(vis)

    # Compute Distances and apply threshold
    coords = np.array(coords)[:, ::-1]
    d = euclidean_distances(centroids, coords).min(1)
    correct_detections = np.count_nonzero((d < 10))
    score = {"scores": {"Correct_Detections": correct_detections}}

    print(json.dumps(score))


def testBinaryMask():
    query = cv2.imread("data/query.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("data/template.jpg", cv2.IMREAD_GRAYSCALE)
    query_res = calcBinaryMask(query)
    template_res = calcBinaryMask(template)
    plt.imshow(query_res)
    plt.show()
    plt.imshow(template_res)
    plt.show()


if __name__ == "__main__":
    # mainProgram()
    testGHT()
    # testBinaryMask()
