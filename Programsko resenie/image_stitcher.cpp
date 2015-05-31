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

#define PANORAMIC_STITCHING 0
#define IMAGE_MOSAIC 1

#define STANDARD 0
#define FAST 1
#define COMPROMISED 2

#define KNN_NUMBER_OF_GOOD_MATCHES 0
#define KNN_HIST 1

#define SIFT 0
#define SURF 1

struct ImagePairCharacteristics {
	std::vector<cv::KeyPoint> keypoints_of_first_image;
	std::vector<cv::KeyPoint> keypoints_of_second_image;
	std::vector<cv::DMatch> good_matches;
	int min_dist_good_match_index;
} ;

struct SimilarImage {
	ImagePairCharacteristics image_pair_characteristics;
	int similar_image_index;
} ;

struct ScaleFactor {
	double x_axis_factor;
	double y_axis_factor;
} ;



cv::Mat stitch_pair(cv::Mat first_image, cv::Mat second_image)
{
	std::vector<cv::Mat> pair_of_partial_images;
	cv::Mat pano;
	pair_of_partial_images.push_back(first_image);
	pair_of_partial_images.push_back(second_image);

	cv::Stitcher stitcher = cv::Stitcher::createDefault();
	stitcher.stitch(pair_of_partial_images, pano);

	return pano;
}

void put_one_image_over_another(cv::Mat &background, cv::Mat &foreground, cv::Mat &output, cv::Point2i start_pt)
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

cv::Mat overlay_images(cv::Mat image1, cv::Mat image2, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
		       std::vector<cv::DMatch> good_matches, int overlay_kp_index)
{
	cv::Mat pano;
	image1.copyTo(pano);


	cv::Point start_pt, location;

	int margin = image2.size().width;
	if (image2.size().height > margin)
		margin = image2.size().height;

	start_pt = cv::Point(margin, margin);

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


	// =========================


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


	// =========================

	
	location = cv::Point(margin + keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.x,
			     margin + keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.y); 
	
	double angle = keypoints_2[good_matches[overlay_kp_index].trainIdx].angle - keypoints_1[good_matches[overlay_kp_index].queryIdx].angle;
	cv::Mat R = cv::getRotationMatrix2D(location, angle,1);
	cv::warpAffine(image2_background, image2_background, R, image2_background.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1.0,1.0,1.0,0.0));


	// =========================


	start_pt = cv::Point(keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.x - keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.x,
			     keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.y - keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.y);

	put_one_image_over_another(pano, image2_background, pano, start_pt);
	


	// Popolnuvanje na transparentnite triagolnici so pikseli od prvata slika
	start_pt.x += location.x - keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.x;
	start_pt.y += location.y - keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.y;

	for (int i = 0; i < image1.rows; i++)
		for (int j = 0; j < image1.cols; j++)
		{
			if (pano.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[3] == 0)
			{
				pano.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[0] = image1.at<cv::Vec4b>(i, j).val[0];
				pano.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[1] = image1.at<cv::Vec4b>(i, j).val[1];
				pano.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[2] = image1.at<cv::Vec4b>(i, j).val[2];
				pano.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[3] = image1.at<cv::Vec4b>(i, j).val[3];
			}
		}



	// Brisenje na transparentniot prostor
	cv::Point left_border_point, right_border_point, upper_border_point, lower_border_point;
	left_border_point = right_border_point = upper_border_point = lower_border_point = cv::Point(-1, -1);

	bool break_flag = false;
	for (int j = 0; j < pano.cols; j++)
	{
		for (int i = 0; i < pano.rows; i++)
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				left_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int j = pano.cols - 1; j >= 0; j--)
	{
		for (int i = 0; i < pano.rows; i++)
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				right_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int i = 0; i < pano.rows; i++)
	{
		for (int j = 0; j < pano.cols; j++)
		if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
		{
			upper_border_point = cv::Point(j,i);
			break_flag = true;
			break;
		}

		if (break_flag)
			break;
	}

	break_flag = false;
	for (int i = pano.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < pano.cols; j++)
		if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
		{
			lower_border_point = cv::Point(j,i);
			break_flag = true;
			break;
		}

		if (break_flag)
			break;
	}

	left_border_point.x += 2;
	left_border_point.y += 2;
	right_border_point.x += 2;
	right_border_point.y += 2;
	upper_border_point.x += 2;
	upper_border_point.y += 2;
	lower_border_point.x += 2;
	lower_border_point.y += 2;
	
	pano = cv::Mat(pano, cv::Rect(left_border_point.x, upper_border_point.y,
				      right_border_point.x - left_border_point.x,
				      lower_border_point.y - upper_border_point.y));

	return pano;
}

cv::Mat stitch_pair_by_overlaying(cv::Mat image1, cv::Mat image2, ImagePairCharacteristics image_pair_characteristics)
{
	cv::Mat pano = overlay_images(image1, image2, image_pair_characteristics.keypoints_of_first_image,
				      image_pair_characteristics.keypoints_of_second_image,
				      image_pair_characteristics.good_matches,
				      image_pair_characteristics.min_dist_good_match_index);
	
	return pano;
}



ImagePairCharacteristics get_image_pair_characteristics(cv::Mat image1, cv::Mat image2, int feature_detector, int feature_extractor)
{
	cv::Mat img_1;
	cv::Mat img_2;
	// Convert to Grayscale
	cvtColor(image1, img_1, CV_RGB2GRAY);
	cvtColor(image2, img_2, CV_RGB2GRAY);
 
	//-- Step 1: Detect the keypoints using SURF/SIFT Detector 
	int minHessian = 100;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	if (feature_detector == SIFT)
	{
		cv::SiftFeatureDetector detector(minHessian);
		detector.detect(img_1, keypoints_1);
		detector.detect(img_2, keypoints_2);
	}
	else if (feature_detector == SURF)
	{
		cv::SurfFeatureDetector detector(minHessian);
		detector.detect(img_1, keypoints_1);
		detector.detect(img_2, keypoints_2);
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::Mat descriptors_1, descriptors_2;
	if (feature_extractor == SIFT)
	{
		cv::SiftDescriptorExtractor extractor;
		extractor.compute(img_1, keypoints_1, descriptors_1);
		extractor.compute(img_2, keypoints_2, descriptors_2);
	}
	else if (feature_extractor == SURF)
	{
		cv::SurfDescriptorExtractor extractor;
		extractor.compute(img_1, keypoints_1, descriptors_1);
		extractor.compute(img_2, keypoints_2, descriptors_2);
	}

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for(int i = 0; i < descriptors_1.rows; i++)
	{ 
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< cv::DMatch > good_matches;

	for(int i = 0; i < descriptors_1.rows; i++)
	{
		if(matches[i].distance <= std::max(2*min_dist, 0.02))
			good_matches.push_back(matches[i]);
	}
	
	min_dist = 1000000;
	int min_dist_good_match_index = 0;
	for (int i = 0; i < good_matches.size(); i++)
		if (good_matches[i].distance < min_dist)
		{
			min_dist = good_matches[i].distance;
			min_dist_good_match_index = i;
		}

	//-- Return only "good" matches
	ImagePairCharacteristics image_pair_characteristics;
	image_pair_characteristics.keypoints_of_first_image = keypoints_1;
	image_pair_characteristics.keypoints_of_second_image = keypoints_2;
	image_pair_characteristics.good_matches = good_matches;
	image_pair_characteristics.min_dist_good_match_index = min_dist_good_match_index;
	return image_pair_characteristics;
}

cv::Mat calculate_hist(cv::Mat input_image)
{
	cv::Mat image;
	input_image.copyTo(image);
	cv::cvtColor(image, image, CV_BGR2HSV);
	int channels[] = { 0, 1 };
	cv::Mat hist;
	int h_bins = 50; int s_bins = 60;
    	int hist_size[] = { h_bins, s_bins };
	float h_ranges[] = { 0, 180 };
    	float s_ranges[] = { 0, 256 };
    	const float* ranges[] = { h_ranges, s_ranges };

	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 2, hist_size, ranges);
	cv::normalize(hist, hist);
	return hist;
}

SimilarImage find_similar_image(cv::Mat pano, std::vector<cv::Mat> partial_images, std::vector<bool> processed_images,
				int knn_method, int feature_detector, int feature_extractor)
{
	SimilarImage most_similar_image;
	if (knn_method == KNN_NUMBER_OF_GOOD_MATCHES)
	{
		int max_number_of_good_matches = -1;
		for (int i = 0; i < partial_images.size(); i++)
		{
			if (!processed_images[i])
			{
				ImagePairCharacteristics current_image_pair_characteristics = get_image_pair_characteristics(pano, partial_images[i],
															     feature_detector, feature_extractor);
				int number_of_good_matches = current_image_pair_characteristics.good_matches.size();
				if (number_of_good_matches > max_number_of_good_matches)
				{
					max_number_of_good_matches = number_of_good_matches;
					most_similar_image.image_pair_characteristics = current_image_pair_characteristics;
					most_similar_image.similar_image_index = i;
				}
			}
		}
	}
	else
	{
		cv::Mat hist = calculate_hist(pano);
		double max_hist_value = -1;
		for (int i = 0; i < partial_images.size(); i++)
		{
			if (!processed_images[i])
			{
				ImagePairCharacteristics current_image_pair_characteristics = get_image_pair_characteristics(pano, partial_images[i],
															     feature_detector, feature_extractor);
				cv::Mat current_partial_image;
				partial_images[i].copyTo(current_partial_image);
				double hist_value = cv::compareHist(hist, calculate_hist(current_partial_image), CV_COMP_INTERSECT);
				if (hist_value > max_hist_value)
				{
					max_hist_value = hist_value;
					most_similar_image.image_pair_characteristics = current_image_pair_characteristics;
					most_similar_image.similar_image_index = i;
				}
			}
		}
	}

	return most_similar_image;
}

bool is_pano_completed(std::vector<bool> processed_images)
{
	for (int i = 0; i < processed_images.size(); i++)
		if (!processed_images[i])
			return false;

	return true;
}



void stitch_images(std::string image_set_info_file_path, int stitching_type, int stitching_method, int image_similarity_method,
	           int feature_detector, int feature_extractor, std::string result_pano_file_path)
{
	if (stitching_type == PANORAMIC_STITCHING)
		std::cout << "\t===== PANORAMIC STITCHING OF IMAGES =====\n\n";
	else
		std::cout << "\t===== CREATION OF AN IMAGE MOSAIC =====\n\n";

	std::vector<cv::Mat> partial_images, scaled_partial_images;
	std::vector<ScaleFactor> scale_factors_of_partial_images;
	std::vector<bool> processed_images;
	cv::Mat result_pano, scaled_result_pano;
	ScaleFactor scale_factor_of_result_pano;

	std::cout << "Loading images into memory:" << std::endl;
	std::ifstream partial_images_info_file(image_set_info_file_path);
	std::string partial_image_path;
	while (std::getline(partial_images_info_file, partial_image_path))
	{
		std::cout << partial_image_path << std::endl;

		cv::Mat current_partial_image = cv::imread(partial_image_path);
		if (stitching_type == IMAGE_MOSAIC)
		{
			cv::cvtColor(current_partial_image, current_partial_image, CV_RGB2RGBA);
		}
		if (stitching_method != FAST)
		{
			partial_images.push_back(current_partial_image);
		}
		
		double current_partial_image_width, current_partial_image_height;
		if (stitching_method == COMPROMISED)
		{
			current_partial_image_width = current_partial_image.cols;
			current_partial_image_height = current_partial_image.rows;
		}
		if (stitching_method != STANDARD)
		{
			cv::resize(current_partial_image, current_partial_image, cv::Size(640, 480));
			scaled_partial_images.push_back(current_partial_image);
		}

		if (stitching_type == IMAGE_MOSAIC && stitching_method == COMPROMISED)
		{
			ScaleFactor current_scale_factor;
			current_scale_factor.x_axis_factor = current_partial_image_width / 640;
			current_scale_factor.y_axis_factor = current_partial_image_height / 480;
			scale_factors_of_partial_images.push_back(current_scale_factor);
		}

		processed_images.push_back(false);
	}
	partial_images_info_file.close();



	int start_index = 0;
	if (stitching_method == FAST)
		scaled_result_pano = scaled_partial_images[start_index];
	else
		result_pano = partial_images[start_index];

	processed_images[start_index] = true;
	std::cout << std::endl;

	std::cout << "Index of corresponding image\t" << "Number of \"good\" matches" << std::endl;
	while (!is_pano_completed(processed_images))
	{
		try
		{
			if (stitching_method == COMPROMISED)
			{
				result_pano.copyTo(scaled_result_pano);
				cv::resize(scaled_result_pano, scaled_result_pano, cv::Size(640, 480));

				if (stitching_type == IMAGE_MOSAIC)
				{
					scale_factor_of_result_pano.x_axis_factor = (double)result_pano.cols / 640;
					scale_factor_of_result_pano.y_axis_factor = (double)result_pano.rows / 480;
				}
			}

			SimilarImage most_similar_image;
			if (stitching_method == STANDARD)
			{
				most_similar_image = find_similar_image(result_pano, partial_images, processed_images,
								        image_similarity_method, feature_detector, feature_extractor);
			}
			else
			{
				most_similar_image = find_similar_image(scaled_result_pano, scaled_partial_images, processed_images,
									image_similarity_method, feature_detector, feature_extractor);
			}

			std::cout << most_similar_image.similar_image_index << "\t\t\t\t";

			if (stitching_type == IMAGE_MOSAIC && stitching_method == COMPROMISED)
			{
				most_similar_image.image_pair_characteristics.keypoints_of_first_image[
					most_similar_image.image_pair_characteristics.good_matches[
						most_similar_image.image_pair_characteristics.min_dist_good_match_index
					].queryIdx
				].pt.x *= scale_factor_of_result_pano.x_axis_factor;
				most_similar_image.image_pair_characteristics.keypoints_of_first_image[
					most_similar_image.image_pair_characteristics.good_matches[
						most_similar_image.image_pair_characteristics.min_dist_good_match_index
					].queryIdx
				].pt.y *= scale_factor_of_result_pano.y_axis_factor;

				most_similar_image.image_pair_characteristics.keypoints_of_second_image[
					most_similar_image.image_pair_characteristics.good_matches[
						most_similar_image.image_pair_characteristics.min_dist_good_match_index
					].trainIdx
				].pt.x *= scale_factors_of_partial_images[most_similar_image.similar_image_index].x_axis_factor;
				most_similar_image.image_pair_characteristics.keypoints_of_second_image[
					most_similar_image.image_pair_characteristics.good_matches[
						most_similar_image.image_pair_characteristics.min_dist_good_match_index
					].trainIdx
				].pt.y *= scale_factors_of_partial_images[most_similar_image.similar_image_index].y_axis_factor;
			}

			std::cout << most_similar_image.image_pair_characteristics.good_matches.size() << std::endl;
			if (stitching_type == PANORAMIC_STITCHING)
			{
				if (stitching_method == FAST)
					scaled_result_pano = stitch_pair(scaled_result_pano, scaled_partial_images[most_similar_image.similar_image_index]);
				else
					result_pano = stitch_pair(result_pano, partial_images[most_similar_image.similar_image_index]);
			}
			else
			{
				if (stitching_method == FAST)
					scaled_result_pano = stitch_pair_by_overlaying(scaled_result_pano,
										       scaled_partial_images[most_similar_image.similar_image_index],
										       most_similar_image.image_pair_characteristics);
				else
					result_pano = stitch_pair_by_overlaying(result_pano,
										partial_images[most_similar_image.similar_image_index],
										most_similar_image.image_pair_characteristics);
			}
			
			processed_images[most_similar_image.similar_image_index] = true;
		}
		catch (std::exception e)
		{
			for (int i = 0; i < processed_images.size(); i++)
				processed_images[i] = false;

			start_index++;
			if (stitching_method == FAST)
				scaled_result_pano = scaled_partial_images[start_index];
			else
				result_pano = partial_images[start_index];

			processed_images[start_index] = true;
			std::cout << std::endl;
		}
	}

	if (stitching_method == FAST)
		cv::imwrite(result_pano_file_path, scaled_result_pano);
	else
		cv::imwrite(result_pano_file_path, result_pano);

	std::cout << "\n\n";
}



int main()
{
	stitch_images("Resources/PartialImagesInfo.csv", PANORAMIC_STITCHING, COMPROMISED, KNN_NUMBER_OF_GOOD_MATCHES,
		       SIFT, SIFT, "Resources/partial_images/Pano(panoramic stitching).png");

	stitch_images("Resources/PartialImagesInfo.csv", IMAGE_MOSAIC, STANDARD, KNN_NUMBER_OF_GOOD_MATCHES,
		       SIFT, SIFT, "Resources/partial_images/Pano(image mosaic).png");
	
	cv::waitKey(0);
	std::cout << "Done!" << std::endl;
	std::cin >> (new char[1]);
	return 0;
}