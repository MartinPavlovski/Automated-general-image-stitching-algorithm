#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <string>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define PARTIAL_IMAGES_INFO_PATH "Resources/PartialImagesInfo.csv"

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

int find_match(cv::Mat pano, std::vector<cv::Mat> partial_images, std::vector<bool> processed_images)
{
	cv::Mat hist = calculate_hist(pano);
	double max_hist_value = -1;
	int matching_index = -1;
	for (int i = 0; i < partial_images.size(); i++)
	{
		if (!processed_images[i])
		{
			cv::Mat current_partial_image;
			partial_images[i].copyTo(current_partial_image);
			double hist_value = cv::compareHist(hist, calculate_hist(current_partial_image), CV_COMP_INTERSECT);
			if (hist_value > max_hist_value)
			{
				max_hist_value = hist_value;
				matching_index = i;
			}
		}
	}

	return matching_index;
}

bool is_pano_completed(std::vector<bool> processed_images)
{
	for (int i = 0; i < processed_images.size(); i++)
	{
		if (!processed_images[i])
			return false;
	}
	return true;
}

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


int main()
{
	std::vector<cv::Mat> partial_images;
	std::vector<bool> processed_images;
	cv::Mat result_pano;

	std::ifstream partial_images_info_file(PARTIAL_IMAGES_INFO_PATH);
	std::string partial_image_path;
	while (std::getline(partial_images_info_file, partial_image_path))
	{
		std::cout << partial_image_path << std::endl;
		partial_images.push_back( cv::imread(partial_image_path) );
		processed_images.push_back(false);
	}
	partial_images_info_file.close();

	int start_index = 0;
	result_pano = partial_images[start_index];
	processed_images[start_index] = true;

	std::cout << std::endl;
	while (!is_pano_completed(processed_images))
	{
		try
		{
			int matching_index = find_match(result_pano, partial_images, processed_images);
			std::cout << "Matching index: " << matching_index << std::endl;

			result_pano = stitch_pair(result_pano, partial_images[matching_index]);
			processed_images[matching_index] = true;
		}
		catch (std::exception e)
		{
			for (int i = 0; i < processed_images.size(); i++)
				processed_images[i] = false;

			start_index++;
			result_pano = partial_images[start_index];
			processed_images[start_index] = true;
			std::cout << std::endl;
		}
	}

	cv::imshow("pano", result_pano);
	
	std::cout << "DONE!" << std::endl;
	cv::waitKey();
	return 0;
}