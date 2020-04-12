#ifndef I3D_LINE3D_PP_VIEW_H_
#define I3D_LINE3D_PP_VIEW_H_

/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// check libs
//#include "configLIBS.h"

// std
#include <map>
#include <iostream>
#include <memory>
// external
#include "Eigen/Eigen"
#include <Eigen/Core>
#include<Eigen/Geometry>
#include<Eigen/StdVector>
// opencv

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include<pcl/common/distances.h>
// internal
#include "commons.h"

/**
 * Line3D++ - View Class
 * ====================
 * Holds all relevant data for one
 * specific image.
 * ====================
 * Author: M.Hofer, 2016
 */
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Isometry3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2i)
namespace L3DPP
{
    class View
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using Ptr=std::shared_ptr<View >;
        View(const unsigned int id, std::vector<Eigen::Vector4f> lines,
             const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
             const Eigen::Vector3d& t,
             const unsigned int width, const unsigned int height,
             const float median_depth,const std::string image_name);
        ~View();

        void loadDepthAndRawImg(cv::Mat& depth,cv::Mat& rawImg);
        // compute spatial regularizer (from pixel value)
        void computeSpatialRegularizer(const float r);
        float getSpecificSpatialReg(const float r);

        // find collinear segments
        void findCollinearSegments(const float dist_t, bool useGPU);

        // draws lines into image
        void drawLineImage(cv::Scalar& color);
        void drawCloud(std::vector<Eigen::Vector3d>& cloud,cv::Scalar& color);
        void drawSingleLine(const unsigned int id,
                            const cv::Scalar& color);
        void drawPoint(Eigen::Vector3d& p,const cv::Scalar& color);

        void drawEpipolarLine(const Eigen::Vector3d& epi, cv::Mat& img);
        void drawLabel(std::vector<Eigen::Vector3d>& cloud,cv::Scalar& color,int id);

        double line2lineDistance(unsigned int id,Eigen::Vector4f& line);
        double line2lineDistanceIn3D(L3DPP::SegmentEnd3D& seg1,unsigned int id);
        double computeLineLen(unsigned int id);
        double computeLineLen(Eigen::Vector3d& end1,Eigen::Vector3d& end2);
        double computeLineCrossLen(Eigen::Vector3d& end1,Eigen::Vector3d& end2,unsigned int id);
        Eigen::Vector3d projPToLine(Eigen::Vector3d& end1,Eigen::Vector3d& end,Eigen::Vector3d& p);
        // access to collinear segments
        std::list<unsigned int> collinearSegments(const unsigned int segID);

        // get ray from 2D point (normalized)
        Eigen::Vector3d getNormalizedRay(const Eigen::Vector3d& p);
        Eigen::Vector3d getNormalizedRay(const Eigen::Vector2d& p);
        Eigen::Vector3d getNormalizedLinePointRay(const unsigned int lID,
                                                  const bool pt1);

        void setSegment3DPos(unsigned int id,L3DPP::SegmentEnd3D& seg3d);

        bool isPointInterSegment(Eigen::Vector3d& end1,Eigen::Vector3d& end2,Eigen::Vector3d& p);
        // projects a 3D point into image
        Eigen::Vector2d project(const Eigen::Vector3d& P);
        Eigen::Vector3d projectWithCheck(const Eigen::Vector3d& P);
        void projectCloud(std::vector<Eigen::Vector3d>& cloud,
                          std::vector<Eigen::Vector3d>& img_points,
                          bool valid_c);
        double computeValidEndPoint(Eigen::Vector3d& end1,Eigen::Vector3d& end2);
        // projects points on line to word
        bool unprojectLine(unsigned int id,std::vector<Eigen::Vector3d>& word_p,bool valid_check);
        std::vector<Eigen::Vector3d> unprojectLineEndP(unsigned int id);
        // get optical axis
        Eigen::Vector3d getOpticalAxis();
        double computeLineCrossLenCore(Eigen::Vector3d l1end1,Eigen::Vector3d l1end2,
                                             Eigen::Vector3d l2end1,Eigen::Vector3d l2end2);
        //get two line angle
        double getLineAngle(unsigned int id,Eigen::Vector4f& line);
        // angle between views or view and segment (in rad)
        double opticalAxesAngle(L3DPP::View::Ptr v);
       // double segmentQualityAngle(const L3DPP::Segment3D& seg3D,
        //                           const unsigned int segID);

        // computes a projective visual neighbor score (to ensure bigger baselines)
        float distanceVisualNeighborScore(L3DPP::View::Ptr v);

        Eigen::Vector3d computePlaneC(const std::vector<double>& coffes);

        L3DPP::SegmentEnd3D get3DLineByID(unsigned int id);

        // baseline between views
        float baseLine(L3DPP::View::Ptr v);

        // set new regularization depth
        void update_median_depth(const float d,
                                 const float sigmaP,
                                 const float med_scene_depth)
        {
            median_depth_ = d;

            if(sigmaP > 0.0f)
            {
                // fixed sigma
                k_ = sigmaP/med_scene_depth;
            }

            median_sigma_ = k_*median_depth_;
        }

        // compute k when fixed sigmaP is used
        void update_k(const float sigmaP, const float med_scene_depth)
        {
            k_ = sigmaP/med_scene_depth;
        }

        // compute regularizer with respect to given 3D point
        float regularizerFrom3Dpoint(const Eigen::Vector3d& P);

        // get coordinates of a specific line segment
        Eigen::Vector4f getLineSegment2D(const unsigned int id);

        // translate view by a fixed vector
        void translate(const Eigen::Vector3d& t);

        // data access
        unsigned int id() const {return id_;}
        Eigen::Vector3d C() const {return C_;}
        Eigen::Matrix3d K() const {return K_;}
        Eigen::Matrix3d Kinv() const {return Kinv_;}
        Eigen::Matrix3d R() const {return R_;}
        Eigen::Matrix3d Rt() const {return Rt_;}
        Eigen::Matrix3d RtKinv() const {return RtKinv_;}
        Eigen::Vector3d t() const {return t_;}
        Eigen::Vector3d pp() const {return pp_;}
        unsigned int width() const {return width_;}
        unsigned int height() const {return height_;}
        float diagonal() const {return diagonal_;}

        size_t num_lines() const {return lines_.size();}
        float k() const {return k_;}
        float median_depth() const {return median_depth_;}
        float median_sigma() const {return median_sigma_;}
        int getLineSize(){ return lines_.size();}
        cv::Mat getImage(){return raw_img_;}
        void resetImage(){bkimg_.copyTo(raw_img_);}
        L3DPP::SegmentEnd3D getSeg3DById(unsigned int id){return seg3d_vec_[id];}
        std::string image_name_;
        std::vector<int> valid_flag_;
        std::vector<int> unmatch_lines_;
        std::vector<L3DPP::SegmentEnd3D> seg3d_vec_;
        cv::Mat bkimg_;
    private:
        // find collinear segments
        void findCollinCPU();

        // checks if a point is on a segment (only approximately!)
        // Note: use only cor collinearity estimation!
        bool pointOnSegment(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                            const Eigen::Vector3d& x);

        // collinearity helper function: point to line distance 2D
        float distance_point2line_2D(const Eigen::Vector3d& line, const Eigen::Vector3d& p);

        // smaller angle between two lines [0,pi/2]
        float smallerAngle(const Eigen::Vector2d& v1, const Eigen::Vector2d& v2);

        // lines
        std::vector<Eigen::Vector4f,Eigen::aligned_allocator<Eigen::Vector4f> > lines_;

        // camera
        unsigned int id_;
        Eigen::Matrix3d K_;
        Eigen::Matrix3d Kinv_;
        Eigen::Matrix3d R_;
        Eigen::Matrix3d Rt_;
        Eigen::Matrix3d RtKinv_;
        Eigen::Vector3d t_;
        Eigen::Vector3d C_;
        Eigen::Vector3d pp_;
        unsigned int width_;
        unsigned int height_;
        float diagonal_;
        float min_line_length_;

        // regularizer
        float k_;
        float initial_median_depth_;
        float median_depth_;
        float median_sigma_;

        // collinearity
        float collin_t_;
        std::vector<std::list<unsigned int> > collin_;
        cv::Mat raw_img_;
        cv::Mat depth_;

    };
}

#endif //I3D_LINE3D_PP_VIEW_H_
