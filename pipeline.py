import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


def sho(img, title=""):
	if title != "": plt.title(title)
	plt.imshow(img)


def plot_2(images, titles, cmap='gray'):
	pass
	# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
	# ax1.imshow(images[0], cmap=cmap)
	# ax1.set_title(titles[0])
	# ax2.imshow(images[1], cmap=cmap)
	# ax2.set_title(titles[1])


images_test = glob.glob("test_images/test*.jpg")
images_straight = glob.glob("test_images/straight_lines*.jpg")

with open("camera_calibration_param.p", mode='rb') as f:
	calibration_param = pickle.load(f)
cameraMatrix, distCoeffs = calibration_param["mtx"], calibration_param["dist"]


def undistort(img):
	return cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)


img_raw = cv2.imread(images_test[1])
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

img_undistort = undistort(img_raw)
plot_2([img_raw, img_undistort], ['raw', 'undistorted'])


def gaussian_blur(img, kernel_size=5):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def grayscale_threshold(img, threshold=(130, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	binary_img = np.zeros_like(gray)

	binary_img[(gray > threshold[0]) & (gray < threshold[1])] = 1

	return binary_img


plot_2([img_undistort, grayscale_threshold(img_undistort)], ["undistorted", "greyscale treshold"])


def sobel_x_treshold(img, thresh_min=20, thresh_max=200):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

	abs_sobel_x = np.absolute(sobel_x)

	scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

	sobel_x_image = np.zeros_like(scaled_sobel_x)

	sobel_x_image[(scaled_sobel_x >= thresh_min) & (scaled_sobel_x <= thresh_max)] = 1

	return sobel_x_image


plot_2([img_undistort, sobel_x_treshold(img_undistort)], ["undistorted", "sobel_x_treshold"])


def hls_channel_treshold(img, channel_name='s', thresh_min=180, thresh_max=255):
	hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	hls_dic = {'h': 0, 'l': 1, 's': 2}
	hls_channel = hls_img[:, :, hls_dic.get(channel_name)]
	hls_channel_image = np.zeros_like(hls_channel)
	hls_channel_image[(hls_channel >= thresh_min) & (hls_channel <= thresh_max)] = 1
	return hls_channel_image


plot_2([img_undistort, hls_channel_treshold(img_undistort)], ["undistorted", "hls s treshold"])


def colors_treshold(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	lower_yellow = np.array([40 * 180 / 360, 20 * 256 / 100, 40 * 256 / 100], dtype=np.uint8)
	upper_yellow = np.array([71 * 180 / 360, 255, 255], dtype=np.uint8)

	lower_white = np.array([0, 0, 180], dtype=np.uint8)
	upper_white = np.array([179, 50, 255], dtype=np.uint8)

	img_whites = cv2.inRange(hsv_img, lower_white, upper_white)

	img_yellows = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

	color_selection_img = cv2.bitwise_or(img_whites, img_yellows)
	return color_selection_img


plot_2([img_undistort, colors_treshold(img_undistort)], ["undistorted", "yellow and white color selection treshold"])


def combined_tresholds(img):
	img_grayscale_threshold = grayscale_threshold(img)
	img_sobel_x = sobel_x_treshold(img)
	img_s_channel_treshold = hls_channel_treshold(img)
	img_colors_treshold = colors_treshold(img)
	img_combined_tresholds = np.zeros_like(img_sobel_x)
	img_combined_tresholds[((img_sobel_x == 1) & (img_grayscale_threshold == 1)) |
												 (img_s_channel_treshold == 1) |
												 (img_colors_treshold == 1)] = 1
	return img_combined_tresholds


img_combined_tresholds = combined_tresholds(img_undistort)
plot_2([img_undistort, img_combined_tresholds], ["undistorted", "combined tresholds"])


def crop(img):
	top_left = (img.shape[1] * 0.45, img.shape[0] * 0.55)
	top_right = (img.shape[1] * 0.55, img.shape[0] * 0.55)
	bot_left = (img.shape[1] * 0.10, img.shape[0])
	bot_right = (img.shape[1] * 0.95, img.shape[0])

	white = np.zeros_like(img)
	points = np.array([[bot_left, top_left, top_right, bot_right]], dtype=np.int32)
	cv2.fillPoly(white, points, 255)

	cropped_img = cv2.bitwise_and(img, white)
	return cropped_img


img_cropped = crop(img_combined_tresholds)
plot_2([img_combined_tresholds, img_cropped], ["undistorted", "cropped"])


def transform_perspective_init(img):
	img_size = (img.shape[1], img.shape[0])
	middle = img_size[0] / 2
	trapezoid_top_width = 135
	src = np.float32([[middle - trapezoid_top_width / 2, 450], [middle + trapezoid_top_width / 2, 450],
										[1250, 720], [40, 720]])
	dst = np.float32([[0, 0], [1280, 0],
										[1250, 720], [40, 720]])
	M = cv2.getPerspectiveTransform(src, dst)
	M_inv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, img_size)
	return warped, M, M_inv


def transform_perspective(image, M):
	return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


for image in images_straight:
	img_raw = cv2.imread(image)
	img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
	img_undistorted = undistort(img_raw)
	img_warped, M, M_inv = transform_perspective_init(img_undistorted)
	plot_2([img_undistorted, img_warped], ['undistorted', 'warped'])


def curve_finding(binary_warped, img_undistort, show=False):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0] / 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0] / nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
			nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
			nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	if show:
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	img_unwarped = transform_perspective(color_warp, M_inv)
	# Combine the result with the original image
	result = cv2.addWeighted(img_undistort, 1, img_unwarped, 0.3, 0)
	# plot_2([img_raw, result], ["undistorted", "result"])

	# Curvature in Pixels
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
	right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
	# print(left_curverad, right_curverad)

	# Curvature in Meters
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30 / 720  # meters per pixel in y dimension
	xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
	# Calculate the new radius of curvature
	left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * left_fit_cr[0])
	right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * right_fit_cr[0])
	# Now our radius of curvature is in meters
	# print(left_curverad, 'm', right_curverad, 'm')
	# Example values: 632.1 m    626.2 m

	# Center offset of car
	histogram_middle = int((leftx_base + rightx_base) / 2)
	middle_offset = (histogram_middle - midpoint) * xm_per_pix
	# print(middle_offset)

	return result, left_curverad, right_curverad, middle_offset


img_perspective = transform_perspective(img_cropped, M)
plot_2([img_cropped, img_perspective], ["binary", "warped"])

img_final = curve_finding(img_perspective, img_undistort, show=True)[0]
plot_2([img_undistort, img_final], ["undistorted", "final"])


def process_image(image):
	try:
		img_blur = gaussian_blur(image)

		img_undistort = undistort(img_blur)

		img_binary = combined_tresholds(img_undistort)

		img_cropped = crop(img_binary)

		img_warped = transform_perspective(img_cropped, M)

		final, cur_l, cur_r, mid_offset = curve_finding(img_warped, img_undistort)

		cv2.putText(final, "Lane Curvature: " + str(cur_l) + " (m)", (100, 100),
								cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
		cv2.putText(final, "Distance from center: " + str(mid_offset) + " (m)", (100, 150),
								cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

	except Exception as e:
		print(e)
		return image

	return final


from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process(image):
	return process_image(image)


output_vid = 'p4.mp4'
clip1 = VideoFileClip("./project_video.mp4")
fl_clip = clip1.fl_image(process)  # NOTE: this function expects color images!!

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_vid))
