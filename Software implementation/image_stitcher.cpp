/******************************************************************************
** Avtori:		Martin Pavlovski IKI 115048
**			Vladimir Ilievski IKI 115028
**			Tamara Dimitrova KNI 111051
**
** Opis:		Proekt po predmetot Masinska vizija
** 
******************************************************************************/

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


/**
	Struktura koja cuva karakteristicni tocki i najdobri sovpaganja megu dve sliki.
*/
struct ImagePairCharacteristics {
	/* Karakteristicni tocki za prvata slika */
	std::vector<cv::KeyPoint> keypoints_of_first_image;
	/* Karakteristicni tocki za vtorata slika */
	std::vector<cv::KeyPoint> keypoints_of_second_image;
	/* rastojanie megu karakteristicnite tocki */
	std::vector<cv::DMatch> good_matches;
	/* minimalnoto rastojanie od site rastojanija */
	int min_dist_good_match_index;
} ;

/**
	Struktura koja cuva podatoci za toa koja slika e najslicna so koja od ostanatite
*/
struct SimilarImage {
	/* Karakteristikite na dvete najslicni sliki */
	ImagePairCharacteristics image_pair_characteristics;
	/* indeksot na najslicnata slika vo nizata sliki */
	int similar_image_index;
} ;

/**
	Struktura koja cuva podatoci za faktorot na skaliranje na edna slika, po x i po y oska
*/
struct ScaleFactor {
	/* faktor na skaliranje po x oska */
	double x_axis_factor;
	/* faktor na skaliranje po y oska */
	double y_axis_factor;
} ;


/**
	Funkcija koja sto spojuva dve sliki, spored nivnite karakteristicni tocki
*/
cv::Mat stitch_pair(cv::Mat first_image, cv::Mat second_image)
{
	/* vektor od dve sliki koi kje se obideme da gi spoime */
	std::vector<cv::Mat> pair_of_partial_images;
	/* rezultatot od spojuvanjeto na sliki */
	cv::Mat pano;
	pair_of_partial_images.push_back(first_image);
	pair_of_partial_images.push_back(second_image);

	cv::Stitcher stitcher = cv::Stitcher::createDefault();
	/* spoj gi dvete sliki i stavi gi vo izleznata slika pano */
	stitcher.stitch(pair_of_partial_images, pano);

	return pano;
}

/**

	Funkcija so koja sto vrz edna pogolema slika(background) koja e najcesto crna, se postavuva druga slika(foreground),
	zapocnuvajki od zadadena tocka(start_pt)
*/
void put_one_image_over_another(cv::Mat &background, cv::Mat &foreground, cv::Mat &output, cv::Point2i start_pt)
{
	/* Prvata slika ja kopirame vo rezultantnata slika */
	background.copyTo(output);

	/* ja izminuvame drugata slika po redici i koloni */
	for (int i = 0; i < foreground.rows; i++)
		for (int j = 0; j < foreground.cols; j++)
		{
			/* ako momentalnata koordinata sobrana so koordinatata na pocetnata tocka e vo granici na dimenziite izleznata slika*/
			if (0 <= (j + start_pt.x) && (j + start_pt.x) <= output.size().width - 1 &&
				0 <= (i + start_pt.y) && (i + start_pt.y) <= output.size().height - 1)
				{
					/* kopiraj gi vrednostite za boite od vtorata slika vo rezultantnata */
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[0] = foreground.at<cv::Vec4b>(i, j).val[0];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[1] = foreground.at<cv::Vec4b>(i, j).val[1];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[2] = foreground.at<cv::Vec4b>(i, j).val[2];
					output.at<cv::Vec4b>(i + start_pt.y, j + start_pt.x).val[3] = foreground.at<cv::Vec4b>(i, j).val[3];
				}
		}
}

/**
	Funkcija so koja sto se dobiva edna slika, koja sto se dobiva od dve sliki koi imaat eden zaednicki del,
	odnosno ednata slika se nadopolnuva so delot od drugata slika koj sto ne se naoga vo prvata. Karektristicnite tocki
	za dvete sliki se veke presmetani, kako i onie tocki sto se smetaat za dobro sovpaganje.
*/
cv::Mat overlay_images(cv::Mat image1, cv::Mat image2, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2,
		       std::vector<cv::DMatch> good_matches, int overlay_kp_index)
{
	cv::Mat pano;
	image1.copyTo(pano);


	cv::Point start_pt, location;

	/* Se bara maksimumot od visinata i sirinata od vtorata slika, za da znaeme za kolku maksimum da ja 
	prosirime prvata slika */
	int margin = image2.size().width;
	if (image2.size().height > margin)
		margin = image2.size().height;

	start_pt = cv::Point(margin, margin);

	/* Na sekoja od dimenziite na rezultantnata slika dodavame dva pati od maksimalnata dimenzija na vtorata slika. 
	Toa go pravime bidejki ne znaeme kade dvete sliki kje se sovpadnat, dali vo dolniot, gorniot, leviot ili
	desniot del od prvata slika*/
	cv::resize(pano, pano, cv::Size(image1.size().width + 2*(margin), image1.size().height + 2*(margin)));

	/* Rezultantnata slika ja boime vo crno */
	for (int i = 0; i < pano.rows; i++)
		for (int j = 0; j < pano.cols; j++)
		{
			pano.at<cv::Vec4b>(i,j).val[0] = 0;
			pano.at<cv::Vec4b>(i,j).val[1] = 0;
			pano.at<cv::Vec4b>(i,j).val[2] = 0;
			pano.at<cv::Vec4b>(i,j).val[3] = 0;
		}

	/* vrz crnata slika ja postavuvame prvata slika */
	put_one_image_over_another(pano, image1, pano, start_pt);


	// =========================

	/* Istata postapka ja pravime i za vtorata slika */
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

	/* ja presmetuvame tockata okolu koja treba da se zarotira vtorata slika */
	location = cv::Point(margin + keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.x,
			     margin + keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.y); 
	
	/* Go presmetuvame agolot pod koj treba da se zarotira vtorata slika za da soodvetstvuva na delot koj sto se poklopuva so prvata slika*/
	double angle = keypoints_2[good_matches[overlay_kp_index].trainIdx].angle - keypoints_1[good_matches[overlay_kp_index].queryIdx].angle;
	cv::Mat R = cv::getRotationMatrix2D(location, angle,1);
	cv::warpAffine(image2_background, image2_background, R, image2_background.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(1.0,1.0,1.0,0.0));


	// =========================

	/* ja presmetuvame pocetnata tocka, odnosno tockata najlevo najgore kade sto dvete sliki zapocnuvaat da se sovpagaat*/
	start_pt = cv::Point(keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.x - keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.x,
			     keypoints_1[good_matches[overlay_kp_index].queryIdx].pt.y - keypoints_2[good_matches[overlay_kp_index].trainIdx].pt.y);

	put_one_image_over_another(pano, image2_background, pano, start_pt);
	


	/* Popolnuvanje na transparentnite triagolnici so pikseli od prvata slika */
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



	/* Brisenje na transparentniot prostor. Posle prilepuvanjeto na dvete sliki okolu niv ostanuva mnogu crn prazen prostor,
	pa istiot prostor go briseme. Prvo treba da gi odredime granicite kade sto slikata ima sodrzin, a ne crn prostor*/
	cv::Point left_border_point, right_border_point, upper_border_point, lower_border_point;
	left_border_point = right_border_point = upper_border_point = lower_border_point = cv::Point(-1, -1);

	/* odreduvanje na levata granica */
	bool break_flag = false;
	for (int j = 0; j < pano.cols; j++)
	{
		for (int i = 0; i < pano.rows; i++)
			/* se dodeka alfa ne bide 255 */
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				left_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	/* odreduvanje na desnata granica */
	break_flag = false;
	for (int j = pano.cols - 1; j >= 0; j--)
	{
		for (int i = 0; i < pano.rows; i++)
			/* se dodeka alfa ne bide 255 */
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				right_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	/* odreduvanje na gornata granica */
	break_flag = false;
	for (int i = 0; i < pano.rows; i++)
	{
		for (int j = 0; j < pano.cols; j++)
			/* se dodeka alfa ne bide 255 */
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				upper_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	/* odreduvanje na dolnata granica */
	break_flag = false;
	for (int i = pano.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < pano.cols; j++)
			/* se dodeka alfa ne bide 255 */
			if (pano.at<cv::Vec4b>(i,j).val[3] == 255)
			{
				lower_border_point = cv::Point(j,i);
				break_flag = true;
				break;
			}

		if (break_flag)
			break;
	}

	/* dodavame dve edinici na sekoja od granicnite tocki, kako eden vid tolerancija */
	left_border_point.x += 2;
	left_border_point.y += 2;
	right_border_point.x += 2;
	right_border_point.y += 2;
	upper_border_point.x += 2;
	upper_border_point.y += 2;
	lower_border_point.x += 2;
	lower_border_point.y += 2;
	
	/* Spoenata slika so novite granici */
	pano = cv::Mat(pano, cv::Rect(left_border_point.x, upper_border_point.y,
				      right_border_point.x - left_border_point.x,
				      lower_border_point.y - upper_border_point.y));

	return pano;
}

/**
	Funkcija so koja sto se spojuvaat dve sliki, no site potrebni karakteristiki za slikite se smesteni vo
	strukturata ImagePairCharacteristics
*/
cv::Mat stitch_pair_by_overlaying(cv::Mat image1, cv::Mat image2, ImagePairCharacteristics image_pair_characteristics)
{
	/* Spojuvanje na dvete sliki */
	cv::Mat pano = overlay_images(image1, image2, image_pair_characteristics.keypoints_of_first_image,
				      image_pair_characteristics.keypoints_of_second_image,
				      image_pair_characteristics.good_matches,
				      image_pair_characteristics.min_dist_good_match_index);
	
	return pano;
}


/**
	Ekstraktiranje na karakteristikite na dve sliki so odreden feature detektor i feature ekstraktor. Rezultatite gi
	smestuvame vo strukturata ImagePairCharacteristics.
*/
ImagePairCharacteristics get_image_pair_characteristics(cv::Mat image1, cv::Mat image2, int feature_detector, int feature_extractor)
{
	/* Nadolu sleduva standardna procedura so koja se ekstraktiraat karakteristicnite tocki i deskriptori */

	cv::Mat img_1;
	cv::Mat img_2;
	// Konverzija na slikite vo Grayscale
	cvtColor(image1, img_1, CV_RGB2GRAY);
	cvtColor(image2, img_2, CV_RGB2GRAY);
 
	// Cekor 1: Detektiranje na klucni tocki koristejkji SURF/SIFT detektor
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

	// Cekor 2: Presmetuvanje na deskriptori (vektori na karakteristiki)
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

	// Cekor 3: Povrzuvanje na vektorite na deskriptorite koi si soodvetstvuvaat koristejkji FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	double max_dist = 0; double min_dist = 100;

	// Brza kalkulacija na maksimalnoto i minimalnoto rastojanie pomegju klucnite tocki
	for(int i = 0; i < descriptors_1.rows; i++)
	{ 
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	
	// Pronaogjanje na "good" match-ovi (t.e. onie match-ovi cie rastojanie e pomalo ili ednakvo na 2*min_dist,
	// ili pomalo od nekoja mala zadadena vrednost (0.02) vo slucaj koga rastojanieto min_dist e mnogu malo
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

	// Se vrakja struktura koja gi sodrzi najvaznite karakteristiki na parot od sliki
	ImagePairCharacteristics image_pair_characteristics;
	image_pair_characteristics.keypoints_of_first_image = keypoints_1;
	image_pair_characteristics.keypoints_of_second_image = keypoints_2;
	image_pair_characteristics.good_matches = good_matches;
	image_pair_characteristics.min_dist_good_match_index = min_dist_good_match_index;
	return image_pair_characteristics;
}


/**
	Standardna funkcija za presmetuvanje na histogram na edna slika po h i s vrednostite od hsv
	prostorot na boi
*/
cv::Mat calculate_hist(cv::Mat input_image)
{
	/* Slikata za koja kje presmetuvame histogram */
	cv::Mat image;
	input_image.copyTo(image);
	/* Konverzija na slikata vo hsv prostor na boi */
	cv::cvtColor(image, image, CV_BGR2HSV);
	int channels[] = { 0, 1 };

	/* Rezultantna slika */
	cv::Mat hist;
	/* Broj na binovi za h i s vrednostite */
	int h_bins = 50; int s_bins = 60;
    	int hist_size[] = { h_bins, s_bins };
    	/* Rang na vrednostite */
	float h_ranges[] = { 0, 180 };
    	float s_ranges[] = { 0, 256 };
    	const float* ranges[] = { h_ranges, s_ranges };

	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 2, hist_size, ranges);
	cv::normalize(hist, hist);
	return hist;
}

/**
	Funkcija za naoganje najslicna slika od mnozestvoto neobraboteni sliki so rezultantnata slika. Najslicna slika
	mozeme da barame spored dva metodi: najgolem broj na dobri sovpaganja ili najslicen histogram. REzultatot go
	smestuvame vo struktura od tipot SimilarImage
*/
SimilarImage find_similar_image(cv::Mat pano, std::vector<cv::Mat> partial_images, std::vector<bool> processed_images,
				int knn_method, int feature_detector, int feature_extractor)
{
	SimilarImage most_similar_image;
	/* Ako barame najslicni sliki spored brojot na dobri sovpaganja, taka sto za najslicni gi smetame onie sliki koi sto
	imaat najgolem broj na dobri sovpaganja */
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
	/* Ako barame najslicni sliki spored najsicen histogram, taka sto dve sliki se najslicni ako nivnite soodvetni
	histogrami imaat najvisok koeficient na slicnost*/
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

/* Funkcija koja sto proveruva dali site sliki se obraboteni */
bool is_pano_completed(std::vector<bool> processed_images)
{
	for (int i = 0; i < processed_images.size(); i++)
		if (!processed_images[i])
			return false;

	return true;
}


/**
	Ova e glavnata funkcija koja sto se povikuva. Parametri koi sto gi prosleduvame se: relativna pateka do mnozestvoto sliki
	(image_set_info_file_path), tip na spojuvanje na slikite(stitching_type) odnosno dali slikite kje bidat spoeni kako panorama
	ili kako mozaik, metod na spojuvanje na slikite(stitching_method) odnosno dali kje se spjuvaat brzo, standardno ili so kompromis
	megu brzo i standardno spojuvanje. Ako metodot na spojuvanje e fast, togas slikite se skaliraat na dimenzii od 640x480, i na vaka
	skaliranite sliki barame deskriptori i istite sliki gi spojuvame vo edna. Ako metodot e standarden, togas
	rabotime so originalnite sliki, dodeka ako e compromised togas barame deskriptori na skaliranite sliki i ovie deskriptori
	gi primenuvame na golemite sliki. Drugi parametri se metod za naoganje slicnost megu slikite(image_similarity_method), koj moze da bide so maksimalen
	broj na dobri sovpaganja ili so histogram. Na kraj se prosleduva koj deskriptor kje se koristi kako i pateka kade sto
	rezultantnata slika kje bide zacuvana.
*/
void stitch_images(std::string image_set_info_file_path, int stitching_type, int stitching_method, int image_similarity_method,
	           int feature_detector, int feature_extractor, std::string result_pano_file_path)
{
	if (stitching_type == PANORAMIC_STITCHING)
		std::cout << "\t===== PANORAMIC STITCHING OF IMAGES =====\n\n";
	else
		std::cout << "\t===== CREATION OF AN IMAGE MOSAIC =====\n\n";

	/* lista kade sto kje se cuvaat slikite sto treba da se spojat, kako i skaliranite sliki */
	std::vector<cv::Mat> partial_images, scaled_partial_images;
	/* faktor na skaliranje na slikite */
	std::vector<ScaleFactor> scale_factors_of_partial_images;
	/* lista od bulovi vrednosti za oznacuvanje koi sliki se vekje obraboteni */
	std::vector<bool> processed_images;
	/* rezultantnata slika i skaliranata rezultantna slika */
	cv::Mat result_pano, scaled_result_pano;
	/* faktor na skaliranje na rezultantnata slika */
	ScaleFactor scale_factor_of_result_pano;

	/* Vcituvanje na slikite vo programata */
	std::cout << "Loading images into memory:" << std::endl;
	std::ifstream partial_images_info_file(image_set_info_file_path);
	std::string partial_image_path;
	/* citanje red po red od fajlot partial_images_info_file, taka sto eden red ja sodrzi patekata do slikata*/
	while (std::getline(partial_images_info_file, partial_image_path))
	{
		std::cout << partial_image_path << std::endl;

		/* vcituvanje na tekovnata slika */
		cv::Mat current_partial_image = cv::imread(partial_image_path);
		/* ako slikite gi spojuvame mozaicno, konvertiraj ja slikata vo rgba prostor na boi*/
		if (stitching_type == IMAGE_MOSAIC)
		{
			cv::cvtColor(current_partial_image, current_partial_image, CV_RGB2RGBA);
		}

		/* ako metodot na spojuvanje ne e brz stavi ja slikata vo listata sliki za obrabotuvanje */
		if (stitching_method != FAST)
		{
			partial_images.push_back(current_partial_image);
		}
		
		/* ako metodot na spojuvanje e compromised, zemi gi sirinata i visinata na tekovnata slika */
		double current_partial_image_width, current_partial_image_height;
		if (stitching_method == COMPROMISED)
		{
			current_partial_image_width = current_partial_image.cols;
			current_partial_image_height = current_partial_image.rows;
		}


		/* ako metodot na spojuvanje ne e standarden, ja skalirame tekovnata slika na dimenzii od 640x480*/
		if (stitching_method != STANDARD)
		{
			cv::resize(current_partial_image, current_partial_image, cv::Size(640, 480));
			scaled_partial_images.push_back(current_partial_image);
		}

		/* ako slikite gi spojuvame mozaicno i ako metod na spojuvanje e compromised,
		treba da se zacuva faktorot na spojuvanje */
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



	/* ako metodot na spojuvanje e fast, togas kje rabotime so skaliranite sliki, vo sprotivno so originalnite sliki */
	int start_index = 0;
	if (stitching_method == FAST)
		scaled_result_pano = scaled_partial_images[start_index];
	else
		result_pano = partial_images[start_index];

	processed_images[start_index] = true;
	std::cout << std::endl;

	/* Zapocnuvame so spojuvanje na slikata */
	std::cout << "Index of corresponding image\t" << "Number of \"good\" matches" << std::endl;
	while (!is_pano_completed(processed_images))
	{
		try
		{
			/* ako metodot na spojuvanje e compromised */
			if (stitching_method == COMPROMISED)
			{
				result_pano.copyTo(scaled_result_pano);
				cv::resize(scaled_result_pano, scaled_result_pano, cv::Size(640, 480));

				/* Ako tipot na spojuvanje e mosaic */
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