#pragma warning(disable:4996)
#include <iostream>
#include <thread>
#include "config.h"
#include "net_recog.h"
#include "objectBoxNormal.h"
#include "face_recog.h"
#include "utils.h"
#include "CameraFunction.h"
#include "trackerBoard.h"
#define MAX_POINT 6

char ky;
bool got_roi = false;
Mat frame_draw_roi;
Point points_array[MAX_POINT];
Rect ROI_RECT;
vector<Point>  co_ordinates;

void  mouse_click(int event, int x, int y, int flags, void *param)
{
	static float min_dist = 8;
	static int count = 0;
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		/*
		switch (count)	// number of set Point 
		{
		case 0:
			cout << "Select top-right    point" << endl;
			break;
		case 1:
			cout << "Select bottom-right point" << endl;
			break;
		case 2:
			cout << "Select bottom-left  point" << endl << endl;
			break;
		default:
			break;
		}
		*/
		if (!got_roi) // you are not select ROI yet!
		{
			points_array[count] = Point(x, y);
			double dist_e2s = sqrt((points_array[count].x - points_array[0].x)*(points_array[count].x - points_array[0].x)
				+ (points_array[count].y - points_array[0].y)*(points_array[count].y - points_array[0].y));
			// cout << "distance: " << dist_e2s << endl;
			if (count >= 1)
			{
				if (dist_e2s < min_dist)
				{
					points_array[count] = points_array[0];
					line(frame_draw_roi, points_array[count - 1], points_array[0], Scalar(0, 255, 0), 2, CV_AA);
				}
				else
				{
					circle(frame_draw_roi, points_array[count], 2, Scalar(0, 255, 0), 2);	//show points on image
					line(frame_draw_roi, points_array[count - 1], points_array[count], Scalar(0, 255, 0), 2, CV_AA);
				}
			}
			else
				circle(frame_draw_roi, points_array[count], 2, Scalar(0, 255, 0), 2);	//show points on image
			imshow("camera", frame_draw_roi);
			count++;
			if (count == MAX_POINT || dist_e2s < min_dist && count != 1) // if select 4 point finished
			{
				cout << "ROI x & y points :" << endl;
				for (int i = 0; i < count; i++)
					cout << points_array[i] << endl;
				cout << endl << "ROI Saved You can continue with double press any keys except 'c' " << endl << "once press 'c' or 'C' to clear points and retry select ROI " << endl << endl;
				ky = waitKey(0) & 0xFF;

				if (ky == 99 || ky == 67)  // c or C to clear
				{
					for (int i = 0; i < MAX_POINT; i++)
						points_array[i] = Point(0, 0);
					imshow("camera", frame_draw_roi);
					count = 0;
					cout << endl << endl << endl << "@---------------------	 Clear Points!	------------------@ " << endl << endl << endl;
				}
				else // user accept points & dosn't want to clear them
				{
					//min_x = std::min(points_array[0].x, points_array[3].x);	//find rectangle for minimum ROI surround it!
					//max_x = std::max(points_array[1].x, points_array[2].x);
					//min_y = std::min(points_array[0].y, points_array[1].y);
					//max_y = std::max(points_array[3].y, points_array[2].y);
					//height_roi = max_y - min_y;
					//width_roi = max_x - min_x;
					//ROI_RECT = Rect(min_x, min_y, width_roi, height_roi);
					got_roi = true;
					for (int i = 0; i < count; i++)
						co_ordinates.push_back(points_array[i]);
					ROI_RECT = boundingRect(co_ordinates);
				}
			}


		}
		else { // if got_roi se true => select roi before
			cout << endl << "You Select ROI Before " << endl << "if you want to clear point press 'c' or double press other keys to continue" << endl << endl;
		}
		break;
	}
	}

}

int main()
{
	// 提示信息
	cout << "\n 请按提示输入你要选择的模式: \n"
		<< " Press 1: 用户注册\n"
		<< " Press 2: 智能看护.\n"
		<< " Press 3: 宝宝笑脸抓拍\n"
		<< " Press 4: 精彩视频片段剪辑(人与宠物互动)\n"
		<< " Press 0: 退出.\n"
		<< " Loading ......" << endl;

	dnn::Net net_detection = readNetFromCaffe(model_detect_C8_prototxt, model_detect_C8_caffemodel);
	dnn::Net net_AG = readNetFromCaffe(model_AG_prototxt, model_AG_caffemodel);
	dnn::Net net_emotion = readNetFromCaffe(model_emotion_prototxt, model_emotion_caffemodel);
	dnn::Net net_MFN = readNetFromCaffe(MobileFaceNet_prototxt, MobileFaceNet_caffemodel);
	//dnn::Net net_LCNNs = readNetFromCaffe(lcnns_model_prototxt, lcnns_model_caffemodel);
	dnn::Net net_face_5p = dnn::readNetFromCaffe(face_point_prototxt, face_point_caffemodel);

	string feature_dir = "../features";
	Face_recog face_recog = Face_recog(feature_dir, net_MFN);

	int function_nb = -1;
	while (function_nb != 1 && function_nb != 2 || function_nb == 3 || function_nb == 4)
	{
		cout << " Please press the button of 1, 2, 3, 4: ";
		cin >> function_nb;
		// 注册新用户
		if (function_nb == 1)
		{
			cout << " Choose the method of register: 1 take photo  2  choose local picture" << endl;
			int register_method = 1;
			cin >> register_method;
			if (register_method == 1)
			{
				VideoCapture cap;
				cap = VideoCapture(0);
				if (!cap.isOpened())
				{
					cout << " Couldn't find camera: " << endl;
					return -1;
				}

				cout << " To confirm the photo, please press the blank!" << endl;
				int press_int = -1;
				for (;;)
				{
					Mat frame, img_resize;
					cap >> frame; // get a new frame from camera/video or read image

					if (frame.empty())
					{
						waitKey();
						break;
					}

					if (frame.channels() == 4)
						cvtColor(frame, frame, COLOR_BGRA2BGR);

					Mat frame_clone = frame.clone();
					vector<DetectResult> detect_results = getDetectResult(frame_clone, net_detection, min_confidence_detect_c8);
					detect_results = throwOverlapBox(detect_results);
					/*for (int i = 0; i < detect_results.size(); i++)
					{
						DetectResult detect_result = detect_results[i];
						if (detect_result.label == 1)
						{
							rectangle(frame_clone, detect_result.objectBox, Scalar(0, 255, 0), 2);
							String label_name = format("%s: %.2f", classNames_Detect_C8[detect_result.label], detect_result.confidence);
							int baseLine = 3;
							Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							int top = max(detect_result.objectBox.y, labelSize.height);
							rectangle(frame_clone, Point(detect_result.objectBox.x, top - labelSize.height),
								Point(detect_result.objectBox.x + labelSize.width, top + baseLine),
								Scalar(255, 255, 255), CV_FILLED);
							putText(frame_clone, label_name, Point(detect_result.objectBox.x, top),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

						}
					}*/

					vector<Rect> rect_faces = getObjectBox(detect_results, 2);
					if (rect_faces.size() == 0)
					{
						cout << " Waining 1: Please Show Your Face in the Camera!" << endl;
					}
					else if (rect_faces.size() > 1)
					{
						cout << " Waining 2: Only One Face in the Camera!" << endl;
					}
					else
					{
						Rect face_rect = rect_faces[0];
						//vector<Mat> faces_onet;
						//faces_onet.clear();
						//mt.findFace(frame, face_t, faces_onet);
						Mat face_img;
						bool face_align_sucess = get_face_align(frame, face_img, face_rect, net_face_5p);
						if (face_align_sucess)
						{//getSquare(face_rect, face_square, frame.size());
							rectangle(frame, face_rect, Scalar(0, 255, 0));
							String label_name = format("face");
							int baseLine = 3;
							Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							int top = max(face_rect.y, labelSize.height);
							rectangle(frame, Point(face_rect.x, top - labelSize.height),
								Point(face_rect.x + labelSize.width, top + baseLine),
								Scalar(255, 255, 255), CV_FILLED);
							putText(frame, label_name, Point(face_rect.x, top),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
							String line_tip = " To confirm the photo please press the blank!";
							putText(frame, line_tip, Point(10, 20),
								FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
							if (press_int == 32)
							{
								cout << "Please input your name: ";
								string name_you;
								cin >> name_you;
								//Mat feature_face = getFeatureLCNNs(face_img, net_MFN);
								Mat feature_face = getFeatureMobileFaceNet(face_img, net_MFN);
								float feature_[FEATURE_NUM] = { 0 }, feature_l2[FEATURE_NUM] = { 0 };
								for (size_t f = 0; f < FEATURE_NUM; f++)
								{
									feature_[f] = feature_face.at<float>(0, f);
								}
								int age = -1;
								cout << "Input the person's age: ";
								cin >> age;
								if (face_recog.face_register(feature_, FEATURE_NUM, age, name_you))
								{
									cout << name_you << " register successful!" << endl;
								}
							}
						}
					}
					namedWindow("camera");
					imshow("camera", frame);
					press_int = waitKey(1);
					if (press_int == 27)
					{
						cap.release();
						destroyWindow("camera");
						cout << "Exit Register Function!" << endl;
						function_nb = -1;
						break;
					}
				}
			}
			else if (register_method == 2)
			{
				cout << "Input the picture's path: (Input 0 to exist register)";
				string picture_path = "";
				while (cin >> picture_path)
				{
					if (picture_path == "0")
					{
						cout << "Exit Register Function!" << endl;
						function_nb = -1;
						break;
					}
					Mat image = imread(picture_path);
					if (image.empty())
					{
						cout << "Picture open failed! Please check the path!" << endl;
						continue;
					}
					if (image.channels() == 4)
						cvtColor(image, image, COLOR_BGRA2BGR);

					Mat image_clone = image.clone();
					vector<DetectResult> detect_results = getDetectResult(image_clone, net_detection, min_confidence_detect_c8);
					detect_results = throwOverlapBox(detect_results);
					vector<Rect> rect_faces = getObjectBox(detect_results, 2);
					if (rect_faces.size() == 0)
					{
						cout << " Waining 1: Please Show Your Face in the Camera!" << endl;
					}
					else if (rect_faces.size() > 1)
					{
						cout << " Waining 2: Only One Face in the Camera!" << endl;
					}
					else
					{
						Rect face_rect = rect_faces[0];
						//vector<Mat> faces_onet;
						//faces_onet.clear();
						//mt.findFace(frame, face_t, faces_onet);
						Mat face_img;
						bool face_align_sucess = get_face_align(image_clone, face_img, face_rect, net_face_5p);
						if (face_align_sucess)
						{
							rectangle(image, face_rect, Scalar(0, 255, 0));
							String label_name = format("face");
							int baseLine = 3;
							Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							int top = max(face_rect.y, labelSize.height);
							rectangle(image, Point(face_rect.x, top - labelSize.height),
								Point(face_rect.x + labelSize.width, top + baseLine),
								Scalar(255, 255, 255), CV_FILLED);
							putText(image, label_name, Point(face_rect.x, top),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
							//String line_tip = " To confirm the photo please press the blank!";
							//putText(image, line_tip, Point(10, 20),
								//FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);


							cout << "Please input your name: ";
							string name_you;
							cin >> name_you;
							//Mat feature_face = getFeatureLCNNs(face_rect, net_MFN);
							Mat feature_face = getFeatureMobileFaceNet(face_img, net_MFN);
							float feature_[FEATURE_NUM] = { 0 }, feature_l2[FEATURE_NUM] = { 0 };
							for (size_t f = 0; f < FEATURE_NUM; f++)
							{
								feature_[f] = feature_face.at<float>(0, f);
							}
							int age = -1;
							cout << "Input the person's age: ";
							cin >> age;
							if (face_recog.face_register(feature_, FEATURE_NUM, age, name_you))
							{
								cout << name_you << " register successful!" << endl;
							}
						}
					}
					namedWindow("camera");
					imshow("camera", image);
					waitKey(0);
					destroyWindow("camera");
					cout << "Input the picture's path: (Input 0 to exist register)";
				}
			}
		}
		else if (function_nb == 2)
		{
			namedWindow("camera");
			string video_path = "../baby_climb.mp4";
			//VideoCapture cap("./climb.mp4");
			VideoCapture cap(video_path);
			//VideoCapture cap(0);

			cap >> frame_draw_roi;
			imshow("camera", frame_draw_roi);
			setMouseCallback("camera", mouse_click, 0);
			std::cout << "Select top-left point" << std::endl; // Lets give instructions on selecting the first point
			waitKey(0);
			destroyWindow("camera");

			cout << "选择该区域的模式："<<"\n键入 1：看护目标限定活动区域"<<"\n键入 2： 看护目标禁止触碰区域\n请选择：";
			int region_method = 1;
			cin >> region_method;
			int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			Face_recog face_recog("../features", net_MFN);

			face_recog.feature_read();
			vector<string> names = face_recog.getPersonName();
			cout << "选择被看护者的ID, 依次输入: " << endl;
			for (int i = 1; i < names.size(); i++)
			{
				cout << i << "  " << names[i] << endl;
			}
			vector<string> watch_persons;
			int watch_person_id;
			while (cin >> watch_person_id)
			{
				if (watch_person_id > 0 && watch_person_id <= names.size())
				{
					watch_persons.push_back(names[watch_person_id]);
				}
				else if (watch_person_id == 0)
					break;
			}
			int press_int = -1;
			int track_align = 0;
			bool initTrackBox = true;

			//动态数组存放 KCFTracker
			TrackerBoard trackerBoard;

			vector<DetectResult> detect_results_1;
			vector<DetectResult> detect_results_2;
			vector<DetectResult> detect_results_tracker;
			detect_results_1.clear();
			detect_results_2.clear();
			

			// get the width and height of this video
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

			//获取帧率
			double rate = cap.get(CV_CAP_PROP_FPS);
			string video_name = getFileNameWithoutType(video_path);
			string video_path_detect = "../videoSave/" + video_name + "_detection.mp4";
			VideoWriter video_w;
			video_w.open(video_path_detect, CV_FOURCC('M', 'P', '4', '2'), rate/2, Size(frame_width, frame_height), true);

			if (!cap.isOpened())
			{
				cout << "Couldn't find camera: " << endl;
				return -1;
			}

			int face_count = 0;
			for (;;)
			{
				Mat frame, img_resize;
				cap >> frame; // get a new frame from camera/video or read image

				if (frame.empty())
				{
					cout << "Exit Take Care Baby Function!" << endl;
					function_nb = -1;
					video_w.release();
					destroyWindow("camera");
					break;
				}

				if (frame.channels() == 4)
					cvtColor(frame, frame, COLOR_BGRA2BGR);

				Mat frame_clone = frame.clone();

				if (track_align <= 10)
				{
					++track_align;
				}
				else
				{
					if (track_align == 11)
						detect_results_1 = getDetectResult(frame, net_detection, min_confidence_detect_c8);
			
					if (track_align == 12)
					{
						detect_results_2 = getDetectResult(frame, net_detection, min_confidence_detect_c8);

						vector<DetectResult> detect_results = getRealBox_2(detect_results_1, detect_results_2);

						std::vector<PersonFace> person_faces = matchPersonAndFace(frame, detect_results);

						if (initTrackBox)
						{
							if (person_faces.size() > 0)
							{
								for (int f = 0; f < person_faces.size(); f++)
								{
									Rect rect_box = person_faces[f].person_box; 
									string face_name = "UnKnown";
									int age = -1;
									if (person_faces[f].face_box.area() != 0)
									{
										Mat face_img;
										bool squareLegitimate = get_face_align(frame, face_img, person_faces[f].face_box, net_face_5p);
										if (squareLegitimate)
										{
											double face_recog_conf = -1;
											face_name = face_recog.recognize(face_img, age, face_recog_conf, 1);
										}
									}
									trackerBoard.trackerAddAge(1, frame, rect_box, age, face_name.c_str());
									cv::rectangle(frame_clone, rect_box, Scalar(0, 255, 255), 1, 8);
									String label_name = format("%s", face_name.c_str());
									int baseLine = 3;
									Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
									int top = max(rect_box.y, labelSize.height);
									cv::rectangle(frame_clone, Point(rect_box.x, top - labelSize.height),
										Point(rect_box.x + labelSize.width, top + baseLine),
										Scalar(255, 255, 255), CV_FILLED);
									cv::putText(frame_clone, label_name, Point(rect_box.x, top),
										FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
								}
								initTrackBox = false;
							}
						}
						else
						{
							trackerBoard.trackerUpdateNet(frame, person_faces, net_face_5p, face_recog);
						}
						track_align = 0;
					}
					++track_align;
				}

				// Update
				detect_results_tracker.clear();
				if (!initTrackBox)
				{
					if (track_align != 1)
						trackerBoard.trackerUpdate(frame);
					for (int i = 0; i < trackerBoard.trackerinfos.size(); i++)
					{
						if (trackerBoard.trackerinfos[i].update_avail)
						{
							Rect track_result = trackerBoard.trackerinfos[i].tracker_box;
							cv::rectangle(frame_clone, track_result, Scalar(0, 255, 255), 1, 8);
							//resultsFile << track_result.x << "," << track_result.y << "," << track_result.width << "," << track_result.height << endl;

							//String label_name = format("%s", classNames_C8[trackerBoard.trackerClasses[i]]);
							String label_name = format("%s AGE: %d", trackerBoard.trackerinfos[i].name.c_str(), trackerBoard.trackerinfos[i].age);
							// std::cout<<"name: "<<trackerBoard.trackerinfos[i].name<<std::endl;
							int baseLine = 3;
							Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							int top = max(track_result.y, labelSize.height);
							cv::rectangle(frame_clone, Point(track_result.x, top - labelSize.height),
								Point(track_result.x + labelSize.width, top + baseLine),
								Scalar(255, 255, 255), CV_FILLED);
							cv::putText(frame_clone, label_name, Point(track_result.x, top),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

							//DetectResult detect_tracker = { trackerBoard.trackerClasses[i], trackerBoard.trackerBoxes[i], 1.0f };
							DetectResult detect_tracker;
							detect_tracker.label = 1;
							detect_tracker.objectBox = trackerBoard.trackerinfos[i].tracker_box;
							detect_tracker.confidence = 1.0f;
							detect_results_tracker.push_back(detect_tracker);
						}
					}
				}

				Scalar judge_color(0,255,0);
				for (int i = 0; i < trackerBoard.trackerinfos.size(); i++)
				{
					TrackerInfo tracker_info = trackerBoard.trackerinfos[i];
					if (tracker_info.update_avail)
					{
						if (nameInList(tracker_info.name, watch_persons))
						{
							if (tracker_info.age < AG_child)
							{
								Rect track_box = tracker_info.tracker_box;
								float warn_threshold = 0.16;
								int warn_w = track_box.width*warn_threshold;
								int warn_h = track_box.height *warn_threshold;
								Point center_UL(track_box.x + warn_w, track_box.y + warn_h);
								Point center_UR(track_box.x + track_box.width - warn_w, track_box.y + warn_h);
								Point center_DL(track_box.x + warn_w, track_box.y + track_box.y - warn_h);
								Point center_DR(track_box.x + track_box.width - warn_w, track_box.y + track_box.y - warn_h);
								int judge_UL = pointPolygonTest(co_ordinates, center_UL, false);
								int judge_UR = pointPolygonTest(co_ordinates, center_UR, false);
								int judge_DL = pointPolygonTest(co_ordinates, center_DL, false);
								int judge_DR = pointPolygonTest(co_ordinates, center_DR, false);
								if (judge_UL != -1 && judge_UR != -1 && judge_DL != -1 && judge_DR != -1)
									judge_color = Scalar(0, 255, 0);
								else
								{
									judge_color = Scalar(0, 0, 255);
									break;
								}
							}
						}
						else
						{
							if (tracker_info.tracker_box.width*1.5 > tracker_info.tracker_box.height)
							{
								Rect track_box = tracker_info.tracker_box;
								float warn_threshold = 0.16;
								int warn_w = track_box.width*warn_threshold;
								int warn_h = track_box.height *warn_threshold;
								Point center_UL(track_box.x + warn_w, track_box.y + warn_h);
								Point center_UR(track_box.x + track_box.width - warn_w, track_box.y + warn_h);
								Point center_DL(track_box.x + warn_w, track_box.y + track_box.y - warn_h);
								Point center_DR(track_box.x + track_box.width - warn_w, track_box.y + track_box.y - warn_h);
								int judge_UL = pointPolygonTest(co_ordinates, center_UL, false);
								int judge_UR = pointPolygonTest(co_ordinates, center_UR, false);
								int judge_DL = pointPolygonTest(co_ordinates, center_DL, false);
								int judge_DR = pointPolygonTest(co_ordinates, center_DR, false);
								if (judge_UL != -1 && judge_UR != -1 && judge_DL != -1 && judge_DR != -1)
									judge_color = Scalar(0, 255, 0);
								else
								{
									judge_color = Scalar(0, 0, 255);
									break;
								}
							}
						}
					}
				}

				for (int i = 1; i < co_ordinates.size(); i++)
				{
					line(frame_clone, points_array[i - 1], points_array[i], judge_color, 2, CV_AA);

				}
				line(frame_clone, points_array[co_ordinates.size() - 1], points_array[0], judge_color, 2, CV_AA);

				namedWindow("camera");
				imshow("camera", frame_clone);
				video_w.write(frame_clone);
				press_int = waitKey(1);
				if (press_int == 27)
				{
					cap.release();
					destroyWindow("camera");
					video_w.release();
					cout << "Exit Take Care Baby Function!" << endl;
					function_nb = -1;
					break;
				}
			}
		}
		else if (function_nb == 3)
		{
			int press_int = -1;
			string video_path = "../smile_girl_2_1.mp4";
			VideoCapture cap;
			cap = VideoCapture(video_path);
			// cap = VideoCapture(0);
			if (!cap.isOpened())
			{
				cout << "Couldn't find camera: " << endl;
				return -1;
			}

			// get the width and height of this video
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

			//获取帧率
			double rate = cap.get(CV_CAP_PROP_FPS);
			string video_name = getFileNameWithoutType(video_path);
			string video_path_detect = "../videoSave/" + video_name + "_detection.mp4";
			VideoWriter video_w;
			video_w.open(video_path_detect, CV_FOURCC('M', 'P', '4', '2'), rate/2, Size(frame_width, frame_height), true);

			// 记录连续帧出现微笑的次数
			long smile_count = 0, frame_count = 0;
			bool smile_saved = false;
			int face_count = 0;
			for (;;)
			{
				frame_count += 1;
				Mat frame, img_resize;
				cap >> frame; // get a new frame from camera/video or read image
				if (frame.empty())
				{
					cap.release();
					destroyWindow("camera");
					video_w.release();
					cout << "End! Exit Save Baby Smiling Recoder Module!" << endl;
					function_nb = -1;
					break;
				}

				if (frame.channels() == 4)
					cvtColor(frame, frame, COLOR_BGRA2BGR);

				Mat frame_clone = frame.clone();
				vector<DetectResult> detect_results = getDetectResult(frame_clone, net_detection, min_confidence_detect_c8);
				detect_results = throwOverlapBox(detect_results);
				for (int i = 0; i < detect_results.size(); i++)
				{
					DetectResult detect_result = detect_results[i];
					if (detect_result.label == 1)
					{
						rectangle(frame_clone, detect_result.objectBox, Scalar(0, 255, 0), 2);
						String label_name = format("%s: %.2f", classNames_Detect_C8[detect_result.label], detect_result.confidence);
						int baseLine = 3;
						Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
						int top = max(detect_result.objectBox.y, labelSize.height);
						rectangle(frame_clone, Point(detect_result.objectBox.x, top - labelSize.height),
							Point(detect_result.objectBox.x + labelSize.width, top + baseLine),
							Scalar(255, 255, 255), CV_FILLED);
						putText(frame_clone, label_name, Point(detect_result.objectBox.x, top),
							FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

					}
				}

				vector<Rect> rect_faces = getObjectBox(detect_results, 2);

				for (int i = 0; i < rect_faces.size(); i++)
				{
					Rect rect_face = rect_faces[i];
					Mat face_img;
					bool face_align_sucess = get_face_align(frame, face_img, rect_face, net_face_5p);
					if (face_align_sucess)
					{
						rectangle(frame_clone, rect_face, Scalar(0, 255, 0), 2);
						String label_name = "face";
						int baseLine = 3;
						Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
						int top = max(rect_face.y, labelSize.height);
						rectangle(frame_clone, Point(rect_face.x, top - labelSize.height),
							Point(rect_face.x + labelSize.width, top + baseLine),
							Scalar(255, 255, 255), CV_FILLED);
						putText(frame_clone, label_name, Point(rect_face.x, top),
							FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

						int gender = -1, age = -1;
						AGRecog_Face(face_img, net_AG, gender, age);

						int smiling = -1;
						double smiling_conf = -1;
						emotionRecognize_face(face_img, net_emotion, smiling, smiling_conf);

						String label_age = format("%d", age);
						Size labelSize_age = getTextSize(label_age, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
						//cout << top << endl;
						int top_age = max(top - labelSize_age.height - baseLine, labelSize_age.height);
						//cout << top_age << endl;
						rectangle(frame_clone, Point(rect_face.x, top_age - labelSize_age.height),
							Point(rect_face.x + labelSize_age.width, top_age),
							Scalar(250, 235, 235), CV_FILLED);
						putText(frame_clone, label_age, Point(rect_face.x, top_age),
							FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

						String label_emotion = format("%s: %.2f", classNames_Emotion[smiling], smiling_conf);
						Size labelSize_emotion = getTextSize(label_emotion, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
						int top_emotion = max(top_age - labelSize_emotion.height - baseLine, labelSize_emotion.height);
						//cout << top_emotion << endl << endl;
						rectangle(frame_clone, Point(rect_face.x, top_emotion - labelSize_emotion.height),
							Point(rect_face.x + labelSize_emotion.width, top_emotion),
							Scalar(250, 235, 235), CV_FILLED);
						putText(frame_clone, label_emotion, Point(rect_face.x, top_emotion),
							FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));


						String line_tip = " Take Photo";
						if (smiling > 0)
						{
							smile_count++;
							if (age < AG_teen)
							{
								if (smile_saved)
								{
									if (smile_count % 10 == 0)
									{
										cout << "current time: " << getCurrentSystemTime() << endl;
										string save_name = format("../smiling_save/baby_smile_%s_%d.jpg", getCurrentSystemTime().c_str(), smile_count);
										imwrite(save_name, frame);
										putText(frame_clone, line_tip, Point(10, 20),
											FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
										cout << save_name << "  saved" << endl;
									}
								}
								else
								{
									if (smile_count > 5)
									{
										cout << "current time: " << getCurrentSystemTime() << endl;
										string save_name = format("../smiling_save/baby_smile_%s_%d.jpg", getCurrentSystemTime().c_str(), smile_count);
										imwrite(save_name, frame);
										smile_saved = true;
										putText(frame_clone, line_tip, Point(10, 20),
											FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
										cout << save_name << "  saved" << endl;
									}
								}
							}

						}
						else
						{
							smile_count = 0;
							smile_saved = false;
						}
					}
				}
				namedWindow("camera");
				imshow("camera", frame_clone);
				video_w.write(frame_clone);
				press_int = waitKey(2);
				if (press_int == 27)
				{
					cap.release();
					destroyWindow("camera");
					video_w.release();
					cout << "Exit Save Baby Smiling Recoder Module!" << endl;
					function_nb = -1;
					break;
				}
			}
		}
		else if (function_nb == 4)
		{
			cout << " You Choosed Video Detection.\n Input the video path: ";
			string video_path = "";
			cin >> video_path;
			int event_type = 0;
			cout << " Input the event you want to monitor:\n"
				<< " Press 1: Person and pet interaction event\n"
				<< " Input the event type: ";
			cin >> event_type;
			face_recog.feature_read();
			VideoCapture video;
			video = VideoCapture(video_path);
			if (!video.isOpened())
			{
				cout << " Couldn't open this video: " << video_path << endl;
			}

			//获取整个帧数
			long totalFrameNumber = video.get(CV_CAP_PROP_FRAME_COUNT);
			cout << "整个视频共" << totalFrameNumber << "帧" << endl;

			// get the width and height of this video
			int frame_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);

			//设置开始帧()
			long frameToStart = 0;
			video.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
			cout << "Read from the" << frameToStart << "th frame" << endl;

			//设置结束帧
			long frameToStop = totalFrameNumber;

			if (frameToStop < frameToStart)
			{
				cout << "The end frame is smaller than the start frame, the program error is about to exit!" << endl;
				return -1;
			}
			else
			{
				cout << "The end frame is：frame " << frameToStop << endl;
			}

			//获取帧率
			double rate = video.get(CV_CAP_PROP_FPS);
			cout << "FPS is :" << rate << endl;

			// 设置跳帧数
			int process_frame_rate = 10;
			int frame_jump = int(rate / process_frame_rate) > 0 ? int(rate / process_frame_rate) : 1;
			string video_name = getFileNameWithoutType(video_path);
			string video_path_detect = "../videoSave/" + video_name + "_detection.mp4";
			VideoWriter video_w;
			video_w.open(video_path_detect, CV_FOURCC('M', 'P', '4', '2'), process_frame_rate, Size(frame_width, frame_height), true);


			// 设置视频规定动作提前与滞后的帧数。
			int videoBufferFrame = rate * 2;
			// 定义保存的视频在原视频中的起始和终止帧
			int video_save_start = 0, video_save_end = 0, video_event_frame = 0;

			//定义一个用来控制读取视频循环结束的变量
			bool stop = false;

			//承载每一帧的图像
			Mat frame;

			//显示每一帧的窗口
			//namedWindow( "Extractedframe" );

			//两帧间的间隔时间:
			int delay = 1000 / rate;

			//利用while循环读取帧
			//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
			long currentFrame = frameToStart;

			//记录时间连续帧数和时间延时帧数
			int event_count = 0;
			int event_delayed = process_frame_rate * 0.5;
			bool start_event = false;
			while (!stop)
			{
				//读取下一帧
				video >> frame;
				if (frame.empty())
				{
					cout << "Read video failed!" << endl;
					video.release();
					destroyWindow("camera");
					video_w.release();
					break;
				}
				//Rotate frame
				//frame = matRotateCounterClockWise90(frame);
				Mat frameClone = frame.clone(), img_resize;
				//此处为跳帧操作
				if (currentFrame % frame_jump == 0)
				{
					cout << "Now processing the " << currentFrame << "th frame" << endl;
					if (frame.channels() == 4)
						cvtColor(frame, frame, COLOR_BGRA2BGR);

					Mat frame_clone = frame.clone();
					vector<DetectResult> detect_results = getDetectResult(frame_clone, net_detection, min_confidence_detect_c8);
					detect_results = throwOverlapBox(detect_results);
					
					vector<Box_SSD> boxs_ssd;
					boxs_ssd.clear();
					int selete_classes[] = {1, 3, 4};
					int selete_classes_num = 3;
					boxs_ssd = filterDetect_Box(detect_results, selete_classes, selete_classes_num);
					for (int i = 0; i < boxs_ssd.size(); i++)
					{
						Box_SSD box_ssd = boxs_ssd[i];
						//if (box_ssd.classId == 1)
						{
							rectangle(frame_clone, box_ssd.box_rect, Scalar(0, 255, 0), 2);
							String label_name = format("%s", classNames_Detect_C8[box_ssd.classId]);
							int baseLine = 3;
							Size labelSize = getTextSize(label_name, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							int top = max(box_ssd.box_rect.y, labelSize.height);
							rectangle(frame_clone, Point(box_ssd.box_rect.x, top - labelSize.height),
								Point(box_ssd.box_rect.x + labelSize.width, top + baseLine),
								Scalar(255, 255, 255), CV_FILLED);
							putText(frame_clone, label_name, Point(box_ssd.box_rect.x, top),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

						}
					}
					namedWindow("camera");
					imshow("camera", frame_clone);
					video_w.write(frame_clone);
					if (getEventMonitor(boxs_ssd, event_type) == event_type)
					{
						event_count++;
						if (event_count >= event_delayed)
						{
							if (!start_event)
							{
								video_save_start = currentFrame - videoBufferFrame;
								start_event = true;
							}
							video_event_frame = currentFrame;
						}
					}
					else
					{
						if (start_event)
						{
							if ((currentFrame - video_event_frame) > videoBufferFrame)
							{
								start_event = false;
								video_save_end = currentFrame;
								time_t tt = time(NULL);
								tm* t = localtime(&tt);
								string video_save_name = "../videoSave/" + video_name + "_" + classType[event_type - 1] + "_" +
									to_string(t->tm_year + 1900) + "_" + to_string(t->tm_mon + 1) + "_" + to_string(t->tm_mday) + "_" +
									to_string(t->tm_hour) + " " + to_string(t->tm_min) + "_" + to_string(t->tm_sec) + ".mp4";
								thread video_save = thread(videoSave, video_path, video_save_name, video_save_start, video_save_end);
								video_save.detach();
							}
						}
						event_count = 0;
					}

				}

				//当时间结束前没有按键按下时，返回值为-1；否则返回按键
				//按下ESC或者到达指定的结束帧后退出读取视频
				if ((char)waitKey(delay) == 27 || currentFrame >= frameToStop - 1)
				{
					video.release();
					destroyWindow("camera");
					video_w.release();
					if (start_event)
					{
						time_t tt = time(NULL);
						tm* t = localtime(&tt);

						string video_save_name = "../videoSave/" + video_name + "_" + classType[event_type - 1] + "_" +
							to_string(t->tm_year + 1900) + "_" + to_string(t->tm_mon + 1) + "_" + to_string(t->tm_mday) + "_" +
							to_string(t->tm_hour) + "_" + to_string(t->tm_min) + "_" + to_string(t->tm_sec) + ".mp4";
						video_save_end = currentFrame;
						thread video_save = thread(videoSave, video_path, video_save_name, video_save_start, video_save_end);
						video_save.detach();
					}
					cout << "Exit Video Detection!" << endl;
					function_nb = -1;
					stop = true;
				}
				currentFrame++;
			}
		}
		else if (function_nb == 0)
			return 0;
	}
}