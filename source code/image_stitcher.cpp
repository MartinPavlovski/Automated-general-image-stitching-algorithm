#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <string>
#include <fstream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <unordered_map>
#include <algorithm>
#include <limits>

using namespace cv;
using namespace std;

#define SIFT 0
#define SURF 1

int MIN_HESSIAN;
int FEATURE_DETECTOR;
int FEATURE_EXTRACTOR;
int MIN_DIST_FACTOR;
int HOMOGRAPHY_METHOD;
int NUM_PARTS;
int USE_ROTATION;

struct ImagePairCharacteristics {
	std::vector<cv::KeyPoint> keypoints_of_first_image;
	std::vector<cv::KeyPoint> keypoints_of_second_image;
	std::vector<cv::DMatch> good_matches;
} ;

struct ScaleFactor {
	double x_axis_factor;
	double y_axis_factor;
} ;

struct BackUpKeyPoint {
	int transparency;
	Point2f pt;
} ;

ImagePairCharacteristics global_ipc;

double euclidean_dist(Point2f p1, Point2f p2)
{
	return sqrt( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) );
}

bool compare_two_matches(DMatch first_match, DMatch second_match)
{
	return (first_match.distance < second_match.distance);
}

bool compare_two_backup_keypoints(BackUpKeyPoint first_bkp, BackUpKeyPoint second_bkp)
{
	return (first_bkp.transparency < second_bkp.transparency);
}

int half(int x)
{
	if (x % 2 == 0)
		return x / 2;
	else
		return (x / 2) + 1;
}

int round(double x)
{
	double remainder = x - floor(x);
	if (remainder < 0.5)
		return floor(x);
	else
		return ceil(x);
}









ImagePairCharacteristics get_image_pair_characteristics(cv::Mat image1, cv::Mat image2)
{
	cv::Mat img_1;
	cv::Mat img_2;
	cvtColor(image1, img_1, CV_RGB2GRAY);
	cvtColor(image2, img_2, CV_RGB2GRAY);
 
	int minHessian = MIN_HESSIAN;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	if (FEATURE_DETECTOR == SIFT)
	{
		cv::SiftFeatureDetector detector(minHessian);
		detector.detect(img_1, keypoints_1);
		detector.detect(img_2, keypoints_2);
	}
	else if (FEATURE_DETECTOR == SURF)
	{
		cv::SurfFeatureDetector detector(minHessian);
		detector.detect(img_1, keypoints_1);
		detector.detect(img_2, keypoints_2);
	}

	cv::Mat descriptors_1, descriptors_2;
	if (FEATURE_EXTRACTOR == SIFT)
	{
		cv::SiftDescriptorExtractor extractor;
		extractor.compute(img_1, keypoints_1, descriptors_1);
		extractor.compute(img_2, keypoints_2, descriptors_2);
	}
	else if (FEATURE_EXTRACTOR == SURF)
	{
		cv::SurfDescriptorExtractor extractor;
		extractor.compute(img_1, keypoints_1, descriptors_1);
		extractor.compute(img_2, keypoints_2, descriptors_2);
	}

	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	double max_dist = -1; double min_dist = std::numeric_limits<double>::max();

	for(int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	
	std::vector< cv::DMatch > good_matches;

	for(int i = 0; i < descriptors_1.rows; i++)
	{
		if(matches[i].distance <= MIN_DIST_FACTOR*min_dist)
			good_matches.push_back(matches[i]);
	}

	ImagePairCharacteristics image_pair_characteristics;
	image_pair_characteristics.keypoints_of_first_image = keypoints_1;
	image_pair_characteristics.keypoints_of_second_image = keypoints_2;
	image_pair_characteristics.good_matches = good_matches;
	return image_pair_characteristics;
}

Mat erase_trans(Mat image, int padding = 3, bool modify_keypoints = false)
{
	cv::Point left_border_point, right_border_point, upper_border_point, lower_border_point;
	left_border_point = right_border_point = upper_border_point = lower_border_point = cv::Point(-1, -1);

	bool break_flag = false;
	for (int j = 0; j < image.cols; j++)
	{
		for (int i = 0; i < image.rows; i++)
			if (image.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				left_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int j = image.cols - 1; j >= 0; j--)
	{
		for (int i = 0; i < image.rows; i++)
			if (image.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				right_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
			if (image.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				upper_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int i = image.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < image.cols; j++)
			if (image.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				lower_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	if (left_border_point.x >= padding)
		left_border_point.x -= padding;

	right_border_point.x += padding;

	if (upper_border_point.y >= padding)
		upper_border_point.y -= padding;

	lower_border_point.y += padding;
	
	image = cv::Mat(image, cv::Rect(left_border_point.x, upper_border_point.y,
				      right_border_point.x - left_border_point.x,
				      lower_border_point.y - upper_border_point.y));

	if (modify_keypoints)
	{
		for (int i = 0; i < global_ipc.keypoints_of_second_image.size(); i++)
		{
			global_ipc.keypoints_of_second_image[i].pt.x -= left_border_point.x;
			global_ipc.keypoints_of_second_image[i].pt.y -= upper_border_point.y;
		}
	}

	return image;
}

cv::Mat get_homography(cv::Mat image1, cv::Mat image2, ImagePairCharacteristics imagePairCharacteristics)
{
	std::vector< Point2f > good_keypoints_1;
	std::vector< Point2f > good_keypoints_2;
 
	for( int i = 0; i < imagePairCharacteristics.good_matches.size(); i++ )
	{
		good_keypoints_1.push_back( imagePairCharacteristics.keypoints_of_first_image[ imagePairCharacteristics.good_matches[i].queryIdx ].pt );
		good_keypoints_2.push_back( imagePairCharacteristics.keypoints_of_second_image[ imagePairCharacteristics.good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( good_keypoints_1, good_keypoints_2, HOMOGRAPHY_METHOD );

	return H;
}

cv::Mat get_warped_image_old(cv::Mat image1, cv::Mat image2, cv::Mat H)
{
	cv::Mat image2Warped;
	image2.copyTo(image2Warped);

	cv::warpPerspective(image2, image2Warped, H, image2Warped.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1.0,1.0,1.0,0.0));

	return image2Warped;
}

cv::Mat smooth_filter(Mat image)
{
	Mat imageAveraged;
	image.copyTo(imageAveraged);

	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
		{
			if (y >= 1 && y <= image.rows - 2 && 
				x >= 1 && x <= image.cols - 2)
			{
				if (image.at<cv::Vec4b>(y, x).val[3] == 0)
				{
					vector<Vec4b> neighbors;

					if (image.at<cv::Vec4b>(y - 1, x - 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y - 1, x - 1) );

					if (image.at<cv::Vec4b>(y - 1, x).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y - 1, x) );

					if (image.at<cv::Vec4b>(y - 1, x + 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y - 1, x + 1) );

					if (image.at<cv::Vec4b>(y, x - 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y, x - 1) );

					if (image.at<cv::Vec4b>(y, x).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y, x) );

					if (image.at<cv::Vec4b>(y, x + 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y, x + 1) );

					if (image.at<cv::Vec4b>(y + 1, x - 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y + 1, x - 1) );

					if (image.at<cv::Vec4b>(y + 1, x).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y + 1, x) );

					if (image.at<cv::Vec4b>(y + 1, x + 1).val[3] == 255)
						neighbors.push_back( image.at<cv::Vec4b>(y + 1, x + 1) );

					if (neighbors.size() > 0)
					{
						double mean_val_0 = 0;
						for (int n = 0; n < neighbors.size(); n++)
							mean_val_0 += (double)neighbors[n].val[0];
						mean_val_0 /= neighbors.size();


						double mean_val_1 = 0;
						for (int n = 0; n < neighbors.size(); n++)
							mean_val_1 += (double)neighbors[n].val[1];
						mean_val_1 /= neighbors.size();


						double mean_val_2 = 0;
						for (int n = 0; n < neighbors.size(); n++)
							mean_val_2 += (double)neighbors[n].val[2];
						mean_val_2 /= neighbors.size();


						double mean_val_3 = 0;
						for (int n = 0; n < neighbors.size(); n++)
							mean_val_3 += (double)neighbors[n].val[3];
						mean_val_3 /= neighbors.size();

						imageAveraged.at<cv::Vec4b>(y, x).val[0] = mean_val_0;
						imageAveraged.at<cv::Vec4b>(y, x).val[1] = mean_val_1;
						imageAveraged.at<cv::Vec4b>(y, x).val[2] = mean_val_2;
						imageAveraged.at<cv::Vec4b>(y, x).val[3] = mean_val_3;
					}
				}
			}
		}

	return imageAveraged;
}

cv::Mat get_warped_image(cv::Mat image1, cv::Mat image2, cv::Mat H)
{
	cv::Mat image2Warped;
	image2.copyTo(image2Warped);

	int margin = ceil(sqrt((double)(image2.size().width * image2.size().width + image2.size().height * image2.size().height)));
	cv::resize(image2Warped, image2Warped, cv::Size(image2.rows + 3 * margin, image2.cols + 3 * margin));

	for (int i = 0; i < image2Warped.rows; i++)
		for (int j = 0; j < image2Warped.cols; j++)
		{
			image2Warped.at<cv::Vec4b>(i,j).val[0] = 0;
			image2Warped.at<cv::Vec4b>(i,j).val[1] = 0;
			image2Warped.at<cv::Vec4b>(i,j).val[2] = 0;
			image2Warped.at<cv::Vec4b>(i,j).val[3] = 0;
		}

	for (int y = 0; y < image2.rows; y++)
		for (int x = 0; x < image2.cols; x++)
		{
			double new_x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2)) /
						(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));

			double new_y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2)) /
						(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));

			new_y += margin;
			new_x += margin;

			if (new_y >= 0 && new_y <= image2Warped.rows - 1 && 
				new_x >= 0 && new_x <= image2Warped.cols - 1)
					image2Warped.at<cv::Vec4b>(new_y, new_x) = image2.at<cv::Vec4b>(y,x);
		}

	for (int i = 0; i < global_ipc.keypoints_of_second_image.size(); i++)
	{
		double x = global_ipc.keypoints_of_second_image[i].pt.x;
		double y = global_ipc.keypoints_of_second_image[i].pt.y;

		double new_x = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2)) /
					(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));

		double new_y = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2)) /
					(H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2));
		
		new_y += margin;
		new_x += margin;

		global_ipc.keypoints_of_second_image[i].pt.x = new_x;
		global_ipc.keypoints_of_second_image[i].pt.y = new_y;
	}

	image2Warped = erase_trans(image2Warped, 3, true);

	Mat image2WarpedAveraged = smooth_filter(image2Warped);
	
	
	return image2WarpedAveraged;
}




void put_one_image_over_another(cv::Mat &background, cv::Mat &foreground, cv::Mat &output, cv::Point2f start_pt)
{
	background.copyTo(output);

	for (int i = 0; i < foreground.rows; i++)
		for (int j = 0; j < foreground.cols; j++)
		{
			if (0 <= (j + start_pt.x) && (j + start_pt.x) <= output.size().width - 1 &&
				0 <= (i + start_pt.y) && (i + start_pt.y) <= output.size().height - 1)
				{
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[0] = foreground.at<cv::Vec4b>(i, j).val[0];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[1] = foreground.at<cv::Vec4b>(i, j).val[1];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[2] = foreground.at<cv::Vec4b>(i, j).val[2];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[3] = foreground.at<cv::Vec4b>(i, j).val[3];
				}
		}
}

Mat remove_seam(Mat image, Point2f ver_seam_pt, Point2f hor_seam_pt)
{
	Mat imageAveraged;
	image.copyTo(imageAveraged);

	int num_neighbors = 0;
	int offset = 1;

	for (double i = ver_seam_pt.y; i <= hor_seam_pt.y; i++)
	{
		for (double j = ver_seam_pt.x - num_neighbors; j <= ver_seam_pt.x + num_neighbors; j++)
		{
			if (imageAveraged.at<cv::Vec4b>(i, j).val[3] != 0)
			{
				for (int channel = 0; channel <= 3; channel++)
					imageAveraged.at<cv::Vec4b>(i, j).val[channel] = (	
						image.at<cv::Vec4b>(i - 1, j - offset).val[channel] +
						image.at<cv::Vec4b>(i - 1, j + offset).val[channel] + 
						image.at<cv::Vec4b>(i, j - offset).val[channel] +
						image.at<cv::Vec4b>(i, j + offset).val[channel] +
						image.at<cv::Vec4b>(i + 1, j - offset).val[channel] +
						image.at<cv::Vec4b>(i + 1, j + offset).val[channel]) / 6;
			}
				
		}
	}

	for (double j = ver_seam_pt.x; j <= hor_seam_pt.x; j++)
	{
		for (double i = hor_seam_pt.y - num_neighbors; i <= hor_seam_pt.y + num_neighbors; i++)
		{
			if (imageAveraged.at<cv::Vec4b>(i, j).val[3] != 0)
			{
				for (int channel = 0; channel <= 3; channel++)
					imageAveraged.at<cv::Vec4b>(i, j).val[channel] = (	
						image.at<cv::Vec4b>(offset - 1, j - 1).val[channel] +
						image.at<cv::Vec4b>(offset - 1, j).val[channel] + 
						image.at<cv::Vec4b>(offset - 1, j + 1).val[channel] +
						image.at<cv::Vec4b>(offset + 1, j - 1).val[channel] +
						image.at<cv::Vec4b>(offset + 1, j).val[channel] +
						image.at<cv::Vec4b>(offset + 1, j + 1).val[channel]) / 6;
			}
		}

	}

	return imageAveraged;
}

cv::Mat overlay_images(Mat image1, vector<Mat> parts, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2)
{
	cv::Mat pano;
	image1.copyTo(pano);

	double max_width = -1, max_height = -1;
	for (int i = 0; i < parts.size(); i++)
	{
		if (parts[i].cols > max_width)
			max_width = parts[i].cols;

		if (parts[i].rows > max_height)
			max_height = parts[i].rows;
	}
	int margin = ceil(sqrt((double)(max_width * max_width + max_height * max_height)));

	cv::Point2f start_pt = cv::Point2f(margin, margin);

	cv::resize(pano, pano, cv::Size(image1.size().width + 2*(margin), image1.size().height + 2*(margin)));

	for (int i = 0; i < pano.rows; i++)
		for (int j = 0; j < pano.cols; j++)
		{
			pano.at<cv::Vec4b>(i,j).val[0] = 0;
			pano.at<cv::Vec4b>(i,j).val[1] = 0;
			pano.at<cv::Vec4b>(i,j).val[2] = 0;
			pano.at<cv::Vec4b>(i,j).val[3] = 0;
		}

	put_one_image_over_another(pano, image1, pano, start_pt);
	
	for (int k = 0; k < parts.size(); k++)
	{
		Mat image2;
		parts[k].copyTo(image2);

		cv::Mat image2_background;
		image2.copyTo(image2_background);

		start_pt = cv::Point(margin, margin);

		cv::resize(image2_background, image2_background, cv::Size(image2.size().width + 2*(margin), image2.size().height + 2*(margin)));

		for (int i = 0; i < image2_background.rows; i++)
			for (int j = 0; j < image2_background.cols; j++)
			{
				image2_background.at<cv::Vec4b>(i,j).val[0] = 0;
				image2_background.at<cv::Vec4b>(i,j).val[1] = 0;
				image2_background.at<cv::Vec4b>(i,j).val[2] = 0;
				image2_background.at<cv::Vec4b>(i,j).val[3] = 0;
			}


		put_one_image_over_another(image2_background, image2, image2_background, start_pt);


		Point2f mean_point_1 = keypoints_1[k].pt;
		Point2f mean_point_2 = keypoints_2[k].pt;		
	
		double angle_1 = keypoints_1[k].angle;
		double angle_2 = keypoints_2[k].angle;

		Point2f location = Point2f(margin + mean_point_2.x,
								   margin + mean_point_2.y);
	
		if (USE_ROTATION == 1)
		{
			double angle = -angle_2 + angle_1;
			angle = (-1) * angle;
			cv::Mat R = cv::getRotationMatrix2D(location, angle,1);
			cv::warpAffine(image2_background, image2_background, R, image2_background.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1.0,1.0,1.0,0.0));
		}


		start_pt = cv::Point2f(margin + mean_point_1.x - mean_point_2.x, margin + mean_point_1.y - mean_point_2.y);

		Point2f ver_seam_pt;
		Point2f hor_seam_pt;
		bool is_ver_seam_pt_set = false;

		for (int i = 0; i < image2.rows; i++)
			for (int j = 0; j < image2.cols; j++)
			{
				if (image2_background.at<cv::Vec4b>(margin + i, margin + j).val[3] != 0 && margin + j >= location.x)
				{
					pano.at<cv::Vec4b>(start_pt.y + i, start_pt.x + j).val[0] = image2_background.at<cv::Vec4b>(margin + i, margin + j).val[0];
					pano.at<cv::Vec4b>(start_pt.y + i, start_pt.x + j).val[1] = image2_background.at<cv::Vec4b>(margin + i, margin + j).val[1];
					pano.at<cv::Vec4b>(start_pt.y + i, start_pt.x + j).val[2] = image2_background.at<cv::Vec4b>(margin + i, margin + j).val[2];
					pano.at<cv::Vec4b>(start_pt.y + i, start_pt.x + j).val[3] = image2_background.at<cv::Vec4b>(margin + i, margin + j).val[3];

					if (is_ver_seam_pt_set == false)
					{
						ver_seam_pt.x = start_pt.x + j;
						ver_seam_pt.y = start_pt.y + i;
						is_ver_seam_pt_set = true;
					}

					hor_seam_pt.x = start_pt.x + j;
					hor_seam_pt.y = start_pt.y + i;
				}
			}

		pano = remove_seam(pano, ver_seam_pt, hor_seam_pt);
	}

	int padding = 3;

	pano = erase_trans(pano, padding);

	return pano;
}

cv::Mat stitch_pair(Mat image1, int index_of_second_image)
{

	return image1;
}

cv::Mat get_stitched_pair(Mat image1, Mat image2)
{
	ImagePairCharacteristics ipc = get_image_pair_characteristics(image1, image2);

	int part_height = image2.rows / NUM_PARTS;
	vector<Mat> parts;
	for (int k = 0; k < NUM_PARTS; k++)
	{
		parts.push_back( cv::Mat(image2, cv::Rect(0, k * part_height, image2.cols, part_height)) );
	}

	vector<int> best_indices;
	vector<KeyPoint> best_kps;
	for (int k = 0; k < NUM_PARTS; k++)
	{
		double min_dist = std::numeric_limits<double>::max();
		int best_index = -1;
		for (int i = 0; i < ipc.good_matches.size(); i++)
		{
			if (ipc.keypoints_of_second_image[ipc.good_matches[i].trainIdx].pt.y >= k*part_height &&
				ipc.keypoints_of_second_image[ipc.good_matches[i].trainIdx].pt.y <= (k+1)*part_height - 1)
				{
					if (ipc.good_matches[i].distance < min_dist)
					{
						min_dist = ipc.good_matches[i].distance;
						best_index = i;
					}
				}
		}

		best_indices.push_back(best_index);
		
		KeyPoint part_best_kp = ipc.keypoints_of_second_image[ipc.good_matches[best_index].trainIdx];
		part_best_kp.pt.y -= k * part_height;
		best_kps.push_back(part_best_kp);
	}

	Mat pano;
	image1.copyTo(pano);


	vector<KeyPoint> keypoints_1;
	for (int k = 0; k < NUM_PARTS; k++)
		keypoints_1.push_back(ipc.keypoints_of_first_image[ipc.good_matches[best_indices[k]].queryIdx]);
	

	pano = overlay_images(pano, parts, keypoints_1, best_kps);

	imwrite("Resources/Output/pano.png", pano);

	return pano;
}

Mat equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

int main( int argc, char** argv )
{
	// Set the parameters
	MIN_HESSIAN = 400;
	FEATURE_DETECTOR = SIFT;
	FEATURE_EXTRACTOR = SIFT;
	MIN_DIST_FACTOR = 3;
	//HOMOGRAPHY_METHOD = 8;
	NUM_PARTS = 2;
	USE_ROTATION = 0;

	// Load the images
	vector<Mat> partial_images;

	partial_images.push_back(imread( "Resources/Input/S1.jpg" ));
	partial_images.push_back(imread( "Resources/Input/S2.jpg" ));
	partial_images.push_back(imread( "Resources/Input/S3.jpg" ));
	partial_images.push_back(imread( "Resources/Input/S4.jpg" ));
	partial_images.push_back(imread( "Resources/Input/S5.jpg" ));
	
	for (int i = 0; i < partial_images.size(); i++) 
		partial_images[i] = equalizeIntensity(partial_images[i]);
	
	for (int i = 0; i < partial_images.size(); i++)
		cvtColor(partial_images[i], partial_images[i], CV_RGB2RGBA);

	Mat pano = partial_images[0];
	for (int i = 1; i < partial_images.size(); i++)
		pano = get_stitched_pair(pano, partial_images[i]);

	for (int i = 0; i < 10; i++)
	{
		pano = smooth_filter(pano);
		imwrite("Resources/Output/pano.png", pano);
	}

	std::cout << "Done!" << std::endl;
	waitKey();
	int in;
	std::cin >> in;
	return 0;
}