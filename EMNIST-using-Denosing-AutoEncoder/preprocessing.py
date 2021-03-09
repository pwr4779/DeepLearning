import cv2


img=cv2.imread('2.png', cv2.IMREAD_COLOR)

'노이즈 제거'
denoised_img1 = cv2.fastNlMeansDenoisingColored(img, None, 40, 40, 7, 21)  # NLmeans
denoised_img2 = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian
denoised_img3 = cv2.medianBlur(img, 5)  # Median
denoised_img4 = cv2.bilateralFilter(img, 5, 50, 50)  # Bilateral

cv2.imshow("before", img)
cv2.imshow("after(NLmeans)", denoised_img1)
cv2.imshow("after(Gaussian)", denoised_img2)
cv2.imshow("after(Median)", denoised_img3)
cv2.imshow("after(Bilateral)", denoised_img4)

cv2.waitKey(0)
cv2.destroyAllWindows()

