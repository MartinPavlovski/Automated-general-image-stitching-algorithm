#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <string>
#include <fstream>

#define PARTIAL_IMAGES_INFO_PATH "Resources/PartialImagesInfo.csv"

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

cv::Mat stitch_partial_images(cv::Mat pano, std::vector<cv::Mat> partial_images, int index)
{
	if (index == partial_images.size())
		return pano;

	std::cout << index << std::endl;
	pano = stitch_pair(pano, partial_images[index]);
	return stitch_partial_images(pano, partial_images, ++index);
}

int main()
{
	std::vector<cv::Mat> partial_images;
	cv::Mat result_pano;

	std::ifstream partial_images_info_file(PARTIAL_IMAGES_INFO_PATH);
	std::string partial_image_path;
	bool is_first_partial_image = true;
	while (std::getline(partial_images_info_file, partial_image_path))
	{
		if (is_first_partial_image)
		{
			result_pano = cv::imread(partial_image_path);
			is_first_partial_image = false;
		}
		else
		{
			std::cout << partial_image_path << std::endl;
			partial_images.push_back( cv::imread(partial_image_path) );
		}
	}
	partial_images_info_file.close();


	result_pano = stitch_partial_images(result_pano, partial_images, 0);
	cv::imshow("Stitching result", result_pano);
	
	std::cout << "DONE!" << std::endl;
	cv::waitKey();
	return 0;
}