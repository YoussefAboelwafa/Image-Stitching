 
![png](outputs/output_2_0.png)
    


### **Getting Correspondences**



```python
def getMatches(img1, img2):

    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_grey, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_grey, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

    matched_img = cv2.drawMatches(
        img1_grey,
        keypoints_1,
        img2_grey,
        keypoints_2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    correspondences = [
        [keypoints_1[match.queryIdx].pt, keypoints_2[match.trainIdx].pt]
        for match in good_matches
    ]

    return correspondences, matched_img
```


    
![png](outputs/output_5_0.png)
    


### **Computing the homography using Direct Linear Transform (DLT)**



```python
def compute_H(correspondences):

    A = []
    for p, p_dash in correspondences:
        A.append(
            [-p[0], -p[1], -1, 0, 0, 0, p[0] * p_dash[0], p[1] * p_dash[0], p_dash[0]]
        )
        A.append(
            [0, 0, 0, -p[0], -p[1], -1, p[0] * p_dash[1], p[1] * p_dash[1], p_dash[1]]
        )

    A = np.array(A)
    U, D, V_transpose = svd(A)
    H = np.reshape(V_transpose[8], (3, 3))
    H /= H[2, 2]

    return H
```

### **Estimating homography using RANSAC**



```python
def RANSAC(correspondences):
    max_inliers = []

    for _ in range(500):

        random_indices = random.sample(range(0, len(correspondences)), 4)
        random_correspondences = [correspondences[i] for i in random_indices]

        H = compute_H(random_correspondences)

        curr_inliers = []
        for corr in correspondences:
            P = corr[0]
            mapped_P = tuple(
                map(
                    int,
                    (
                        np.dot(H, np.transpose([P[0], P[1], 1]))
                        / (np.dot(H, np.transpose([P[0], P[1], 1]))[2])
                    ).astype(int)[:2],
                )
            )
            e = np.linalg.norm(np.asarray(corr[1]) - np.asarray(mapped_P))
            if e < 5:
                curr_inliers.append(corr)

        if len(curr_inliers) > len(max_inliers):
            max_inliers = curr_inliers

    return compute_H(max_inliers)
```

### **Warping the image**

```python
def warp_image(warped_img, original_img, h):
    warped_y, warped_x, _ = warped_img.shape
    original_y, original_x, _ = original_img.shape

    warped_corner1, warped_corner2, warped_corner3, warped_corner4 = getCorners(
        warped_img, h
    )

    border_min_point, _ = get_borders(
        warped_corner1, warped_corner2, (0, 0), (0, original_y)
    )

    _, border_max_point = get_borders(
        warped_corner3,
        warped_corner4,
        (original_x, 0),
        (original_x, original_y),
    )

    height = border_max_point[1] - border_min_point[1]
    width = border_max_point[0] - border_min_point[0]
    final_img = np.zeros((height, width, 3))

    warped_min_border, warped_max_border = get_borders(
        warped_corner1, warped_corner2, warped_corner3, warped_corner4
    )

    corners = [
        (0, 0),
        (0, original_y),
        (original_x, original_y),
        (original_x, 0),
    ]

    if border_min_point[0] == 0 and border_min_point[1] == 0:
        final_img[0 : original_img.shape[0], 0 : original_img.shape[1]] = original_img
    else:
        shifted_corners = [
            (point[0] - warped_min_border[0], point[1] - warped_min_border[1])
            for point in corners
        ]
        final_img[
            shifted_corners[0][1] : shifted_corners[2][1],
            shifted_corners[0][0] : shifted_corners[2][0],
        ] = original_img

    h_inverse = np.linalg.inv(h)

    # Warp each channel separately
    for channel in range(3):
        for i in range(warped_min_border[0], warped_max_border[0]):
            for j in range(warped_min_border[1], warped_max_border[1]):
                p_dash = compute_p_dash(h_inverse, (i, j), True)
                if (
                    p_dash[0] < 0
                    or p_dash[0] > warped_x - 1
                    or p_dash[1] < 0
                    or p_dash[1] > warped_y - 1
                ):
                    continue
                new_x = abs(border_min_point[0] - i)
                new_y = abs(border_min_point[1] - j)

                final_img[new_y][new_x][channel] = interpolation(
                    p_dash[0], p_dash[1], warped_img[:, :, channel]
                )

    return final_img, warped_min_border
```

## **Stitching 2 images**



```python
def stitch_images(img1, img2):
    correspondences, _ = getMatches(img1, img2)
    h = RANSAC(correspondences)
    stitched_img, _ = warp_image(img1, img2, h)
    stitched_img = stitched_img.astype(np.uint8)
    return stitched_img
```

### **_Note: The order of stitching yields different results_**


#### **Right stitch**

    
![png](outputs/output_19_0.png)
    

#### **Left stitch**


![png](outputs/output_21_0.png)
    


#### Another example



    
![png](outputs/output_23_0.png)
    



    
![png](outputs/output_23_1.png)
    



    
![png](outputs/output_23_2.png)
    


## **Stitching 3 images**


    
![png](outputs/output_25_0.png)
    



    
![png](outputs/output_25_1.png)
    



    
![png](outputs/output_25_2.png)
    


#### Another example


    
![png](outputs/output_27_0.png)
    



    
![png](outputs/output_27_1.png)
    



    
![png](outputs/output_27_2.png)
    

