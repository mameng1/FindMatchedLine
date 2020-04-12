#include "view.h"

namespace L3DPP
{
    //------------------------------------------------------------------------------
    View::View(const unsigned int id, std::vector<Eigen::Vector4f> lines,
               const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
               const Eigen::Vector3d& t,
               const unsigned int width, const unsigned int height,
               const float median_depth,const std::string image_name) :
        id_(id), lines_(lines), K_(K), R_(R), t_(t),
        width_(width), height_(height), initial_median_depth_(fmax(fabs(median_depth),L3D_EPS))
    {
        // init
        diagonal_ = sqrtf(float(width_*width_+height_*height_));
        min_line_length_ = diagonal_*L3D_DEF_MIN_LINE_LENGTH_FACTOR;

        collin_t_ = 0.0f;

        image_name_=image_name;
        // camera
        pp_ = Eigen::Vector3d(K_(0,2),K_(1,2),1.0);

        Kinv_ = K_.inverse();
        Rt_ = R_.transpose();
        RtKinv_  = Rt_*Kinv_;
        C_ = Rt_ * (-1.0 * t_);

        k_ = 0.0f;
        median_depth_ = 0.0f;
        median_sigma_ = 0.0f;

        valid_flag_.resize(lines.size(),0);
        unmatch_lines_.resize(lines.size(),0);
        seg3d_vec_.resize(lines.size());
    }

    //------------------------------------------------------------------------------
    View::~View()
    {

    }

    //------------------------------------------------------------------------------
    void View::loadDepthAndRawImg(cv::Mat& depth,cv::Mat& rawImg)
    {
        depth_=depth;
        raw_img_=rawImg;
        raw_img_.copyTo(bkimg_);
    }
    //-----------------------------------------------------------------------------
    bool View::unprojectLine(unsigned int id,std::vector<Eigen::Vector3d>& word_p,bool valid_check=false)
    {
        Eigen::Vector4f line=lines_[id];
        double x1=line[0],y1=line[1],x2=line[2],y2=line[3];
        double x_len=std::abs(x1-x2);
        double y_len=std::abs(y1-y2);
        Eigen::Vector3d line_c;
        line_c[0]=y1-y2;
        line_c[1]=-(x1-x2);
        line_c[2]=x1*y2-x2*y1;

        std::vector<cv::Vec2d> valid_p;
        if(x_len>y_len)
        {
            double max_x=std::max(x1,x2);
            double min_x=std::min(x1,x2);
            for(double x=min_x;x<=max_x;x++)
            {
                double y=-(line_c[0]*x+line_c[2])/line_c[1];
                valid_p.push_back(cv::Vec2d(x,y));
            }
        }
        else
        {
            double max_y=std::max(y1,y2);
            double min_y=std::min(y1,y2);
            for(double y=min_y;y<=max_y;y++)
            {
                double x=-(line_c[1]*y+line_c[2])/line_c[0];
                valid_p.push_back(cv::Vec2d(x,y));
            }
        }
        double all_pn=valid_p.size();
        double valid_pn=0;
        double depthScale = 1000.0;
        for(int p=0;p<valid_p.size();p++)
        {
            int u=valid_p[p][0];
            int v=valid_p[p][1];
            double d = depth_.ptr<unsigned short> ( v )[u]; // 深度值
            double d1 = depth_.ptr<unsigned short> ( v-1 )[u]; // 深度值
            double d2 = depth_.ptr<unsigned short> ( v+1 )[u-1]; // 深度值
            double d3 = depth_.ptr<unsigned short> ( v )[u-1]; // 深度值
            double d4 = depth_.ptr<unsigned short> ( v )[u+1]; // 深度值
            std::vector<double> depth_vec;
            depth_vec.push_back(d);
            depth_vec.push_back(d1);
            depth_vec.push_back(d2);
            depth_vec.push_back(d3);
            depth_vec.push_back(d4);
            double max_d=*std::max_element(depth_vec.begin(),depth_vec.end());
            double min_d=*std::min_element(depth_vec.begin(),depth_vec.end());
            if ( d==0 || d1==0 || d2==0 || d3==0 || (max_d-min_d)>100) continue; // 为0表示没有测量到
            if ( d >= 7000 ) continue; // 深度太大时不稳定，去掉
            valid_pn++;
            double ratio=double(d)/depthScale;
            Eigen::Vector3d point;
            point[0]=u;
            point[1]=v;
            point[2]=1;
            point=Kinv_*point*ratio;
            Eigen::Vector3d pointWorld = Rt_*point+C_;
            word_p.push_back(pointWorld);
         }
        if(valid_check)
        {
            double ratio=valid_pn/all_pn;
            if(ratio<0.5)
                return true;
        }
        return false;
    }
    //----------------------------------------------------------------------------
    L3DPP::SegmentEnd3D View::get3DLineByID(unsigned int id)
    {
        return seg3d_vec_[id];
    }
    //-----------------------------------------------------------------------------
    void View::projectCloud(std::vector<Eigen::Vector3d>& cloud,
                      std::vector<Eigen::Vector3d>& img_points,
                      bool valid_c)
    {
        for(size_t i=0;i<cloud.size();i++)
        {
            Eigen::Vector3d& pointWorld=cloud[i];
            if(pointWorld[2]>0)
            {
                Eigen::Vector3d ip=projectWithCheck(pointWorld);
                double u=ip[0];
                double v=ip[1];
                if(valid_c)
                {
                    if(u>=0 && u<width_ && v>=0 && v<height_)
                    {
                     img_points.push_back(ip);
                    }
                }
                else
                    img_points.push_back(ip);
            }
        }
    }
    //-------------------------------------------------------------------------
    void View::drawLabel(std::vector<Eigen::Vector3d>& cloud,cv::Scalar& color,int id)
    {
        double mediu_x,mediu_y;
        for(size_t i=0;i<cloud.size();i++)
        {
            mediu_x+=cloud[i][0];
            mediu_y+=cloud[i][1];
        }
        mediu_x/=cloud.size();
        mediu_y/=cloud.size();

        cv::putText(raw_img_,std::to_string(id), cv::Point2d(mediu_x,mediu_y), cv::FONT_HERSHEY_PLAIN, 2, color);
    }
    //-------------------------------------------------------------------------
    Eigen::Vector3d View::computePlaneC(const std::vector<double>& coffes)
    {
        double u1=coffes[0];
        double v1=coffes[1];
        double u2=coffes[2];
        double v2=coffes[3];

        Eigen::Vector3d p1,p2;
        p1<<u1,v1,1.0;
        p2<<u2,v2,1.0;
        p1=Kinv_*p1;
        p2=Kinv_*p2;
        Eigen::Vector3d norm_vec=p1.cross(p2);
        norm_vec=Rt_*norm_vec;
        norm_vec.normalize();
        return norm_vec;
    }
    //------------------------------------------------------------------------
    std::vector<Eigen::Vector3d> View::unprojectLineEndP(unsigned int id)
    {
        Eigen::Vector4f line=lines_[id];
        Eigen::Vector3d end1,end2;
        end1<<line[0],line[1],1;
        end2<<line[2],line[3],1;
        end1=Kinv_*end1;
        end2=Kinv_*end2;
        end1=Rt_*end1;
        end2=Rt_*end2;
        end1.normalize();
        end2.normalize();
        std::vector<Eigen::Vector3d> end_vec;
        end_vec.push_back(end1);
        end_vec.push_back(end2);
        return end_vec;
    }
    //------------------------------------------------------------------------------
    double View::line2lineDistance(unsigned int id,Eigen::Vector4f& l2)
    {
          Eigen::Vector4f l1=lines_[id];

          Eigen::Vector3d p[2];
          p[0] = Eigen::Vector3d(l1[0],l1[1],1.0f);
          p[1] = Eigen::Vector3d(l1[2],l1[3],1.0f);
          Eigen::Vector3d line1 = p[0].cross(p[1]);

          Eigen::Vector3d q[2];
          q[0] = Eigen::Vector3d(l2[0],l2[1],1.0f);
          q[1] = Eigen::Vector3d(l2[2],l2[3],1.0f);
          Eigen::Vector3d line2 = q[0].cross(q[1]);

          // compute distances
          float d1 = fmax(distance_point2line_2D(line1,q[0]),
                          distance_point2line_2D(line1,q[1]));
          float d2 = fmax(distance_point2line_2D(line2,p[0]),
                          distance_point2line_2D(line2,p[1]));

        double d=fmax(d1,d2);
        return d;
    }
    //-----------------------------------------------------------------
    double View::line2lineDistanceIn3D(L3DPP::SegmentEnd3D& seg1,unsigned int id)
    {
        Eigen::Vector3d norm_vec1=seg1.e1_-seg1.e2_;
        norm_vec1.normalize();
        float x1,y1,z1,nx1,ny1,nz1;
        x1=seg1.e1_[0],y1=seg1.e1_[1],z1=seg1.e1_[2];
        nx1=norm_vec1[0],ny1=norm_vec1[1],nz1=norm_vec1[2];
        Eigen::VectorXf line1(6);
        line1<<x1,y1,z1,nx1,ny1,nz1;

        L3DPP::SegmentEnd3D& seg2=seg3d_vec_[id];
        Eigen::Vector3d norm_vec2=seg2.e1_-seg2.e2_;
        norm_vec2.normalize();
        float x2,y2,z2,nx2,ny2,nz2;
        x2=seg2.e1_[0],y2=seg2.e1_[1],z2=seg2.e1_[2];
        nx2=norm_vec2[0],ny2=norm_vec2[1],nz2=norm_vec2[2];
        Eigen::VectorXf line2(6);
        line2<<x2,y2,z2,nx2,ny2,nz2;

        Eigen::Vector4f p1,p2;
        pcl::lineToLineSegment(line1,line2,p1,p2);
        double len=(p1-p2).norm();
        return len;
    }
    //----------------------------------------------------------------
    void View::setSegment3DPos(unsigned int id,L3DPP::SegmentEnd3D& seg3d)
    {
        if(id<seg3d_vec_.size())
            seg3d_vec_[id]=seg3d;
    }
    double View::computeLineLen(unsigned int id)
    {
        Eigen::Vector4f& l1=lines_[id];
        double len=std::sqrt((l1[0]-l1[2])*(l1[0]-l1[2])+(l1[1]-l1[3])*(l1[1]-l1[3]));
        return len;
    }
    double View::computeLineLen(Eigen::Vector3d& end1,Eigen::Vector3d& end2)
    {
        double len=std::sqrt((end1[0]-end2[0])*(end1[0]-end2[0])+(end1[1]-end2[1])*(end1[1]-end2[1]));
        return len;
    }
    //------------------------------------------------------------------
    bool View::isPointInterSegment(Eigen::Vector3d& end1,Eigen::Vector3d& end2,Eigen::Vector3d& p)
    {
        Eigen::Vector3d vec1=p-end1;
        Eigen::Vector3d vec2=p-end2;
        double len1=vec1.norm();
        double len2=vec2.norm();
        if(len1 <1e-6 || len2<1e-6)
            return true;
        double vecdot=vec1[0]*vec2[0]+vec1[1]*vec2[1];
        double costheta=vecdot/(len1*len2);
        bool flag=false;
        if(costheta<=0)
            flag=true;
        return flag;
    }
    double View::computeLineCrossLenCore(Eigen::Vector3d l1end1,Eigen::Vector3d l1end2,
                                         Eigen::Vector3d l2end1,Eigen::Vector3d l2end2)
    {
        Eigen::Vector3d p1=Kinv_*l1end1;
        Eigen::Vector3d p2=Kinv_*l1end2;

        Eigen::Vector3d blend1=l2end1,blend2=l2end2;
        Eigen::Vector3d lend1=l2end1,lend2=l2end2;
        double x1=blend1[0],y1=blend1[1],x2=blend2[0],y2=blend2[1];

        lend1=Kinv_*lend1;
        lend2=Kinv_*lend2;
        Eigen::Vector3d pp1=projPToLine(lend1,lend2,p1);
        Eigen::Vector3d pp2=projPToLine(lend1,lend2,p2);

        if(isPointInterSegment(lend1,lend2,pp1) || isPointInterSegment(lend1,lend2,pp2) ||
        isPointInterSegment(pp1,pp2,lend1)   || isPointInterSegment(pp1,pp2,lend2))
        {
            pp1=K_*pp1;
            pp2=K_*pp2;
            double xmin=std::min(x1,x2);
            double xmax=std::max(x1,x2);
            double ymin=std::min(y1,y2);
            double ymax=std::max(y1,y2);

            std::vector<Eigen::Vector3d> p_vec(4);
            p_vec[0]=pp1;
            p_vec[1]=pp2;
            p_vec[2]=blend1;
            p_vec[3]=blend2;

            double xlen=std::abs(xmax-xmin);
            double ylen=std::abs(ymax-ymin);
            if(xlen>ylen)
                std::sort(p_vec.begin(),p_vec.end(),[](Eigen::Vector3d& p1,Eigen::Vector3d& p2)
                {
                    return p1[0]<p2[0];
                });
            else
                std::sort(p_vec.begin(),p_vec.end(),[](Eigen::Vector3d& p1,Eigen::Vector3d& p2)
                {
                    return p1[1]<p2[1];
                });
            Eigen::Vector3d e1=p_vec[0],e2=p_vec[1],e3=p_vec[2],e4=p_vec[3];
            double max_len=std::sqrt((e1[0]-e4[0])*(e1[0]-e4[0])+(e1[1]-e4[1])*(e1[1]-e4[1]));
            double min_len=std::sqrt((e2[0]-e3[0])*(e2[0]-e3[0])+(e2[1]-e3[1])*(e2[1]-e3[1]));
            return min_len/max_len;


        }
        else
            return 0;
    }
    //--------------------------------------------------------------------
    double View::computeLineCrossLen(Eigen::Vector3d& end1,Eigen::Vector3d& end2,unsigned int id)
    {
        //Eigen::Vector3d p1=Kinv_*end1;
        //Eigen::Vector3d p2=Kinv_*end2;
        Eigen::Vector4f line=lines_[id];
        double x1=line[0],y1=line[1],x2=line[2],y2=line[3];
        Eigen::Vector3d lend1,lend2;
        lend1<<x1,y1,1;
        lend2<<x2,y2,1;
        double ratio1=computeLineCrossLenCore(end1,end2,lend1,lend2);
        double ratio2=computeLineCrossLenCore(lend1,lend2,end1,end2);
        return std::min(ratio1,ratio2);
    }
    Eigen::Vector3d View::projPToLine(Eigen::Vector3d& end1,Eigen::Vector3d& end2,Eigen::Vector3d& p)
    {
        double len=(end1-end2).norm();
        Eigen::Vector3d end1end2=end2-end1;
        Eigen::Vector3d end1p=p-end1;
        double ratio=end1p[0]*end1end2[0]+end1p[1]*end1end2[1];
        ratio=ratio/(len*len);
        return end1+end1end2*ratio;
    }
    //------------------------------------------------------------------------------
    void View::drawLineImage(cv::Scalar& color)
    {
        //img = cv::Mat::zeros(height_,width_,CV_8UC3);

        for(size_t i=0; i<lines_.size(); ++i)
        {
            Eigen::Vector4f coords = lines_[i];

            cv::Point p1(coords[0],coords[1]);
            cv::Point p2(coords[2],coords[3]);
            cv::line(raw_img_,p1,p2,color,2);
        }
    }
    //------------------------------------------------------------------------------
    void View::drawCloud(std::vector<Eigen::Vector3d>& cloud,cv::Scalar& color)
    {
        for(size_t i=0; i<cloud.size(); ++i)
        {
            Eigen::Vector3d coords = cloud[i];

            cv::Point p1(coords[0],coords[1]);
            if(p1.x>=0 && p1.x<width_ && p1.y>=0 && p1.y<height_)
                cv::circle(raw_img_,p1,2,color,1);
        }
    }
    //---------------------------------------------------------------------------
    void View::drawPoint(Eigen::Vector3d& p,const cv::Scalar& color)
    {
        cv::Point p1(p[0],p[1]);
        if(p1.x>=0 && p1.x<width_ && p1.y>=0 && p1.y<height_)
            cv::circle(raw_img_,p1,3,color,2);
    }
    //------------------------------------------------------------------------------
    void View::drawSingleLine(const unsigned int id,
                              const cv::Scalar& color)
    {
        if(id < lines_.size())
        {
            Eigen::Vector4f coords = lines_[id];
            cv::Point p1(coords[0],coords[1]);
            cv::Point p2(coords[2],coords[3]);
            cv::line(raw_img_,p1,p2,color,3);
        }
    }

    //------------------------------------------------------------------------------
    void View::drawEpipolarLine(const Eigen::Vector3d& epi, cv::Mat& img)
    {
        // intersect with image borders
        Eigen::Vector3d p1(0,0,1);
        Eigen::Vector3d p2(img.cols,0,1);
        Eigen::Vector3d p3(img.cols,img.rows,1);
        Eigen::Vector3d p4(0,img.rows,1);

        Eigen::Vector3d borders[4];
        borders[0] = p1.cross(p2);
        borders[1] = p2.cross(p3);
        borders[2] = p3.cross(p4);
        borders[3] = p4.cross(p1);

        std::vector<Eigen::Vector3d> intersections;
        for(size_t i=0; i<4; ++i)
        {
            Eigen::Vector3d I = borders[i].cross(epi);
            if(fabs(I.z()) > L3D_EPS)
            {
                I /= I.z();
                I(2) = 1.0;

                // check position
                if(I.x() > -1.0 && I.x() < img.cols+1 &&
                        I.y() > -1.0 && I.y() < img.rows+1)
                {
                    intersections.push_back(I);
                }
            }
        }

        if(intersections.size() < 2)
            return;

        // find intersections that are farthest apart
        double max_dist = 0.0f;
        Eigen::Vector3d e_p1(0,0,0);
        Eigen::Vector3d e_p2(0,0,0);

        for(size_t i=0; i<intersections.size()-1; ++i)
        {
            Eigen::Vector3d _p = intersections[i];
            for(size_t j = i+1; j<intersections.size(); ++j)
            {
                Eigen::Vector3d _q = intersections[j];
                double len = (_p-_q).norm();
                if(len > max_dist)
                {
                    max_dist = len;
                    e_p1 = _p;
                    e_p2 = _q;
                }
            }
        }

        cv::Point pt1(e_p1.x(),e_p1.y());
        cv::Point pt2(e_p2.x(),e_p2.y());
        cv::line(img,pt1,pt2,cv::Scalar(0,255,255),3);
    }

    //------------------------------------------------------------------------------
    void View::findCollinearSegments(const float dist_t, bool useGPU)
    {
        if(fabs(dist_t-collin_t_) < L3D_EPS)
        {
            // already computed
            return;
        }

        if(dist_t > L3D_EPS)
        {

            collin_t_ = dist_t;

            findCollinCPU();
        }
    }

    
    //------------------------------------------------------------------------------
    void View::findCollinCPU()
    {
        // reset
        collin_ = std::vector<std::list<unsigned int> >(lines_.size());

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int r=0; r<collin_.size(); ++r)
        {
            Eigen::Vector3d p[2];
            Eigen::Vector4f l1 = lines_[r];

            p[0] = Eigen::Vector3d(l1[0],l1[1],1.0f);
            p[1] = Eigen::Vector3d(l1[2],l1[3],1.0f);
            Eigen::Vector3d line1 = p[0].cross(p[1]);

            for(size_t c=0; c<lines_.size(); ++c)
            {
                if(r == c)
                    continue;

                // line data
                Eigen::Vector4f l2 = lines_[c];

                Eigen::Vector3d q[2];
                q[0] = Eigen::Vector3d(l2[0],l2[1],1.0f);
                q[1] = Eigen::Vector3d(l2[0],l2[1],1.0f);
                Eigen::Vector3d line2 = q[0].cross(q[1]);

                // check location (overlap)
                if(pointOnSegment(p[0],p[1],q[0]) ||
                        pointOnSegment(p[0],p[1],q[1]) ||
                        pointOnSegment(q[0],q[1],p[0]) ||
                        pointOnSegment(q[0],q[1],p[1]))
                {
                    // overlap -> not collinear
                    continue;
                }

                // compute distances
                float d1 = fmax(distance_point2line_2D(line1,q[0]),
                                distance_point2line_2D(line1,q[1]));
                float d2 = fmax(distance_point2line_2D(line2,p[0]),
                                distance_point2line_2D(line2,p[1]));

                if(fmax(d1,d2) < collin_t_)
                {
                    collin_[r].push_back(c);
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    float View::distance_point2line_2D(const Eigen::Vector3d& line, const Eigen::Vector3d& p)
    {
        return fabs((line.x()*p.x()+line.y()*p.y()+line.z())/sqrtf(line.x()*line.x()+line.y()*line.y()));
    }

    //------------------------------------------------------------------------------
    float View::smallerAngle(const Eigen::Vector2d& v1, const Eigen::Vector2d& v2)
    {
        float angle = acos(fmax(fmin(v1.dot(v2),1.0f),-1.0f));
        if(angle > L3D_PI_1_2)
            angle = M_PI-angle;

        return angle;
    }

    //------------------------------------------------------------------------------
    std::list<unsigned int> View::collinearSegments(const unsigned int segID)
    {
        if(collin_.size() == lines_.size() && segID < lines_.size())
            return collin_[segID];
        else
            return std::list<unsigned int>();
    }

    //------------------------------------------------------------------------------
    bool View::pointOnSegment(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                              const Eigen::Vector3d& x)
    {
        Eigen::Vector2d v1(p1.x()-x.x(),p1.y()-x.y());
        Eigen::Vector2d v2(p2.x()-x.x(),p2.y()-x.y());
        return (v1.dot(v2) < L3D_EPS);
    }

    //------------------------------------------------------------------------------
    void View::computeSpatialRegularizer(const float r)
    {
        k_ = getSpecificSpatialReg(r);
    }

    //------------------------------------------------------------------------------
    float View::getSpecificSpatialReg(const float r)
    {
        Eigen::Vector3d pp_shifted = pp_+Eigen::Vector3d(r,0.0,0.0);
        Eigen::Vector3d ray_pp = getNormalizedRay(pp_);
        Eigen::Vector3d ray_pp_shifted = getNormalizedRay(pp_shifted);
        double alpha = acos(fmin(fmax(double(ray_pp.dot(ray_pp_shifted)),-1.0),1.0));
        return sin(alpha);
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedRay(const Eigen::Vector3d& p)
    {
        Eigen::Vector3d ray = RtKinv_*p;
        return ray.normalized();
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedRay(const Eigen::Vector2d& p)
    {
        return getNormalizedRay(Eigen::Vector3d(p.x(),p.y(),1.0));
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getNormalizedLinePointRay(const unsigned int lID,
                                                    const bool pt1)
    {
        Eigen::Vector3d ray(0,0,0);
        if(lID < lines_.size())
        {
            Eigen::Vector3d p;
            if(pt1)
            {
                // ray through P1
                p = Eigen::Vector3d(lines_[lID].x(),
                                    lines_[lID].y(),1.0);
            }
            else
            {
                // ray through P2
                p = Eigen::Vector3d(lines_[lID].z(),
                                    lines_[lID].w(),1.0);
            }

            return getNormalizedRay(p);
        }
        return ray;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector2d View::project(const Eigen::Vector3d& P)
    {
        Eigen::Vector3d q = (R_*P + t_);

        // projection to unit focal plane
        double xn = (1.0 * q[0] + 0.0 * q[2]) / q[2];
        double yn = (1.0 * q[1] + 0.0 * q[2]) / q[2];

        // projection function
        q[0] = xn;
        q[1] = yn;
        q[2] = 1;
        q = K_*q;

        Eigen::Vector2d res;
        res(0) = q(0)/q(2);
        res(1) = q(1)/q(2);
        return res;
    }
    //------------------------------------------------------------------------
     double View::computeValidEndPoint(Eigen::Vector3d& end1,Eigen::Vector3d& end2)
     {
        Eigen::Vector3d lt,lb,rt,rb;
        lt<<0,0,1;
        lb<<0,(height_-1),1;
        rt<<(width_-1),0,1;
        rb<<(width_-1),(height_-1),1;
        lt=Kinv_*lt;
        lb=Kinv_*lb;
        rt=Kinv_*rt;
        rb=Kinv_*rb;

        /*Eigen::Vector3d left_b,right_b,top_b,bottom_b;
        left_b<<1,0,0;
        right_b<<1,0,(-width_+1);
        top_b<<0,1,0;
        bottom_b<<0,1,(-height_+1);
        right_b.normalize();
        bottom_b.normalize();*/
        Eigen::Vector3d nend1,nend2;
        nend1=Kinv_*end1;
        nend2=Kinv_*end2;
        Eigen::Vector3d line=nend1.cross(nend2);
        Eigen::Vector3d left_b,right_b,top_b,bottom_b;

        left_b=lt.cross(lb);
        right_b=rt.cross(rb);
        top_b=lt.cross(rt);
        bottom_b=lb.cross(rb);

        Eigen::Vector3d left_p=line.cross(left_b);
        Eigen::Vector3d right_p=line.cross(right_b);
        Eigen::Vector3d top_p=line.cross(top_b);
        Eigen::Vector3d bottom_p=line.cross(bottom_b);

        left_p[2]+=L3D_EPS;
        right_p[2]+=L3D_EPS;
        top_p[2]+=L3D_EPS;
        bottom_p[2]+=L3D_EPS;

        left_p/=left_p[2];
        right_p/=right_p[2];
        top_p/=top_p[2];
        bottom_p/=bottom_p[2];

        std::vector<Eigen::Vector3d> point_v(6);
        point_v[0]=left_p;
        point_v[1]=right_p;
        point_v[2]=top_p;
        point_v[3]=bottom_p;
        point_v[4]=nend1;
        point_v[5]=nend2;
        double xmax=-std::numeric_limits<double>::max(),ymax=-std::numeric_limits<double>::max();
        double xmin,ymin;
        xmin=ymin=std::numeric_limits<double>::max();
        for(size_t i=0;i<point_v.size();i++)
        {
            Eigen::Vector3d& p=point_v[i];
            if(p[0]<xmin)
                xmin=p[0];
            if(p[0]>xmax)
                xmax=p[0];
            if(p[1]<ymin)
                ymin=p[1];
            if(p[1]>ymax)
                ymax=p[1];
        }
        double xlen=std::abs(xmax-xmin);
        double ylen=std::abs(ymax-ymin);
        if(xlen>ylen)
            std::sort(point_v.begin(),point_v.end(),[](Eigen::Vector3d& p1,Eigen::Vector3d& p2)
            {
                return p1[0]<p2[0];
            });
        else
            std::sort(point_v.begin(),point_v.end(),[](Eigen::Vector3d& p1,Eigen::Vector3d& p2)
            {
                return p1[1]<p2[1];
            });
        end1=K_*point_v[2];
        end2=K_*point_v[3];

        double len=std::sqrt((end2[0]-end1[0])*(end2[0]-end1[0])+(end2[1]-end1[1])*(end2[1]-end1[1]));
        return len;
     }
    //------------------------------------------------------------------------------
    Eigen::Vector3d View::projectWithCheck(const Eigen::Vector3d& P)
    {
        Eigen::Vector3d q = (R_*P + t_);

        // projection to unit focal plane
        double xn = (1.0 * q[0] + 0.0 * q[2]) / q[2];
        double yn = (1.0 * q[1] + 0.0 * q[2]) / q[2];

        // projection function
        q[0] = xn;
        q[1] = yn;
        q[2] = 1;
        q = K_*q;

        Eigen::Vector3d res(0,0,-1);

        if(fabs(q(2)) > L3D_EPS)
        {
            res(0) = q(0)/q(2);
            res(1) = q(1)/q(2);
            res(2) = 1;
        }

        return res;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f View::getLineSegment2D(const unsigned int id)
    {
        Eigen::Vector4f coords(0,0,0,0);
        if(id < lines_.size())
        {
            coords = lines_[id];
        }
        return coords;
    }
    //------------------------------------------------------------------------------
    double View::getLineAngle(unsigned int id,Eigen::Vector4f& line)
    {
        Eigen::Vector4f& line_local=lines_[id];
        Eigen::Vector3f line1,line2;
        line1<<line_local[0]-line_local[2],line_local[1]-line_local[3],1;
        line2<<line[0]-line[2],line[1]-line[3],1;
        float line1_len,line2_len;
        line1_len=std::sqrt(line1[0]*line1[0]+line1[1]*line1[1]);
        line2_len=std::sqrt(line2[0]*line2[0]+line2[1]*line2[1]);
        float angle=(line1[0]*line2[0]+line1[1]*line2[1])/(line1_len*line2_len);
        return std::abs(angle);
    }
    //------------------------------------------------------------------------------
    float View::regularizerFrom3Dpoint(const Eigen::Vector3d& P)
    {
        return (P-C_).norm()*k_;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d View::getOpticalAxis()
    {
        return getNormalizedRay(pp_);
    }

    //------------------------------------------------------------------------------
    double View::opticalAxesAngle(L3DPP::View::Ptr v)
    {
        Eigen::Vector3d r1 = getNormalizedRay(pp_);
        Eigen::Vector3d r2 = v->getOpticalAxis();

        return acos(fmin(fmax(double(r1.dot(r2)),-1.0),1.0));
    }

    //------------------------------------------------------------------------------
    float View::distanceVisualNeighborScore(L3DPP::View::Ptr v)
    {
        // bring tgt camera center to src coordinate frame
        Eigen::Vector3d Ctgt_t = R_*v->C()+t_;

        // define two planes trough the camera center
        Eigen::Vector3d n1(1,0,0);
        Eigen::Vector3d n2(0,1,0);

        // compute distances to the planes
        float dist1 = fabs(n1.dot(Ctgt_t));
        float dist2 = fabs(n2.dot(Ctgt_t));

        return dist1+dist2;
    }

    //------------------------------------------------------------------------------
    float View::baseLine(L3DPP::View::Ptr v)
    {
        return (C_ - v->C()).norm();
    }

    //------------------------------------------------------------------------------
    void View::translate(const Eigen::Vector3d& t)
    {
        C_ += t;
        t_ = -R_ * C_;
    }
}
