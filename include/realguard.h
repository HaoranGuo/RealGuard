#ifndef REALGUARD_H
#define REALGUARD_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/full_object_detection.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <librealsense2/rs.hpp>


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


class RealGuard{
private:
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    anet_type facerec;

public:
    bool find_depth_from(
    rs2::depth_frame const & frame,
    float const depth_scale,
    dlib::full_object_detection const & face,
    int markup_from, int markup_to,
    float * p_average_depth);

    bool validate_face(
    rs2::depth_frame const & frame,
    float const depth_scale,
    dlib::full_object_detection const & face);

    dlib::matrix<float,0,1> RealGuard::get_128d_features_of_face(char* image_path);
};

#endif // REALGUARD_H