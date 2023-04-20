// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
// Modified by: Haoran Guo on 04/19/2023
// Add some basic functions to recognize the face

#include "../include/realguard.h"

bool RealGuard::find_depth_from(
    rs2::depth_frame const & frame,
    float const depth_scale,
    dlib::full_object_detection const & face,
    int markup_from, int markup_to,
    float * p_average_depth){
        uint16_t const * data = reinterpret_cast<uint16_t const *>(frame.get_data());

    float average_depth = 0;
    size_t n_points = 0;
    for( int i = markup_from; i <= markup_to; ++i )
    {
        auto pt = face.part( i );
        auto depth_in_pixels = *(data + pt.y() * frame.get_width() + pt.x());
        if( !depth_in_pixels )
            continue;
        average_depth += depth_in_pixels * depth_scale;
        ++n_points;
    }
    if( !n_points )
        return false;
    if( p_average_depth )
        *p_average_depth = average_depth / n_points;
    return true;
}

bool RealGuard::validate_face(
    rs2::depth_frame const & frame,
    float const depth_scale,
    dlib::full_object_detection const & face){
    float left_ear_depth = 100, right_ear_depth = 100;
    if( !find_depth_from( frame, depth_scale, face, 0, 1, &right_ear_depth )
        && !find_depth_from( frame, depth_scale, face, 15, 16, &left_ear_depth ) ){
        return false;
    }
    float ear_depth = std::min( right_ear_depth, left_ear_depth );

    float chin_depth;
    if( !find_depth_from( frame, depth_scale, face, 7, 9, &chin_depth ) ){
        return false;
    }

    float nose_depth;
    if( !find_depth_from( frame, depth_scale, face, 29, 30, &nose_depth ) ){
        return false;
    }

    float right_eye_depth;
    if( !find_depth_from( frame, depth_scale, face, 36, 41, &right_eye_depth ) ){
        return false;
    }
    float left_eye_depth;
    if( !find_depth_from( frame, depth_scale, face, 42, 47, &left_eye_depth ) ){
        return false;
    }
    float eye_depth = std::min( left_eye_depth, right_eye_depth );

    float mouth_depth;
    if( !find_depth_from( frame, depth_scale, face, 48, 67, &mouth_depth ) ){
        return false;
    }
    
    if( nose_depth >= eye_depth ){
        return false;
    }
    if( eye_depth - nose_depth > 0.07f ){
        return false;
    }
    if( ear_depth <= eye_depth ){
        return false;
    }
    if( mouth_depth <= nose_depth ){
        return false;
    }
    if( mouth_depth > chin_depth ){
        return false;
    }

    float x = std::max( { nose_depth, eye_depth, ear_depth, mouth_depth, chin_depth } );
    float n = std::min( { nose_depth, eye_depth, ear_depth, mouth_depth, chin_depth } );

    if( x - n > 0.20f ){
        return false;
    }
    if( x - n < 0.02f ){
        return false;
    }

    return true;
}

// def get_128d_features_of_face(image_path):
//     image = cv2.imread(image_path)
//     faces = detector(image, 1)

//     if len(faces) != 0:
//         shape = predictor(image, faces[0])
//         face_descriptor = facerec.compute_face_descriptor(image, shape)
//     else:
//         face_descriptor = 0
//     return face_descriptor
dlib::matrix<float,0,1> RealGuard::get_128d_features_of_face(char* image_path){
    dlib::matrix<float,0,1> face_descriptor;
    dlib::array2d<dlib::rgb_pixel> image;
    dlib::load_image(image, image_path);
    std::vector<dlib::rectangle> faces = detector(image);
    if(faces.size() != 0){
        dlib::full_object_detection shape = predictor(image, faces[0]);
        face_descriptor = facerec.compute_face_descriptor(image, shape, 1);
    }
    else{
        face_descriptor = 0;
    }
    return face_descriptor;
}