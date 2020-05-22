#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <assert.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <fstream>

#include <podofo/podofo.h>

using namespace PoDoFo;

using namespace cv;
using namespace std;

int recursive_markers = 3;
const int subsampling = 1;
const int ss_border = 2;

void setnumber(Mat &m, int n)
{
  m.at<uchar>(1, 1) = (n/1 & 0x01) * 255;
  m.at<uchar>(1, 2) = (n/2 & 0x01) * 255;
  m.at<uchar>(1, 3) = (n/4 & 0x01) * 255;
  m.at<uchar>(2, 1) = (n/8 & 0x01) * 255;
  m.at<uchar>(2, 2) = (n/16 & 0x01) * 255;
  m.at<uchar>(2, 3) = (n/32 & 0x01) * 255;
  m.at<uchar>(3, 1) = (n/64 & 0x01) * 255;
  m.at<uchar>(3, 2) = (n/128 & 0x01) * 255;
  m.at<uchar>(3, 3) = (n/256 & 0x01) * 255;
}

int smallidtomask(int id, int x, int y)
{
  int j = (id / 32) * 2 + (id % 2) + y;
  int i = (id % 32) + x;
  
  return (j*13 + i * 7) % 512;
}

int idtomask(int id)
{
  if ((id&2==2)) 
    return id ^ 170;
  else
    return id ^ 340;
}

int masktoid(int mask)
{
  if ((mask&2==2)) 
    return mask ^ 170;
  else
    return mask ^ 340;
}

void writemarker(Mat &img, int page, int offx = 0, int offy = 0)
{
  Mat marker = Mat::zeros(5, 5, CV_8UC1);
  marker.at<uchar>(2, 4) = 255;
  
  for(int j=0;j<16;j++)
    for(int i=0;i<32;i++) {
      setnumber(marker, idtomask(j*32+i));
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+(i%2)*5+offy, j*10+(i%2)*5+5+offy));
      
      setnumber(marker, page ^ smallidtomask(j*32+i, 0, 2*((i+1)%2)-1));
      marker = 255 - marker;
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+((i+1)%2)*5+offy, j*10+((i+1)%2)*5+5+offy));
      marker = 255 - marker;
    }
}

void checker_recurse(Mat &img, Mat &checker)
{
  Mat hr;
  int w = img.size().width;
  int h = img.size().height;
  int ws = subsampling+2*ss_border;
  int w_hr = w*ws;;
  uint8_t *ptr_hr, *ptr_img;
  
  if (!recursive_markers) {
    checker = img;
    return;
  }
  
  resize(img, hr, Point2i(img.size())*ws, 0, 0, INTER_NEAREST);
  
  ptr_img = img.ptr<uchar>(0);
  ptr_hr = hr.ptr<uchar>(0);
  
  for(int y=0;y<h;y++)
    for(int x=0;x<w;x++) {
      for(int j=ss_border;j<ws-ss_border;j++)
        for(int i=j%2+ss_border;i<ws-ss_border;i+=2)
          ptr_hr[(y*ws+j)*w_hr+x*ws+i] = 255-ptr_hr[(y*ws+j)*w_hr+x*ws+i];
    }
    
  checker = hr;
}

void checker_add_recursive(Mat &img, Mat &checker)
{
  for(int i=0;i<recursive_markers;i++)
    checker_recurse(img, img);
}

class SVGMarker {
public:
    cv::Mat_<uint8_t> &img;

    int limit_width = 32;
    int limit_height = 32;

    int scale = 1;

    int recursive = 0;

    int dpi = 600;

    double bleed = 0;

    void setBleed(double const _bleed) {
        bleed = _bleed;
    }

    void setDPI(int const _dpi) {
        dpi = _dpi;
    }

    void setLimitWidth(int const _limit) {
        limit_width = _limit;
    }

    void setLimitHeight(int const _limit) {
        limit_height = _limit;
    }


    SVGMarker(cv::Mat_<uint8_t> & _img) : img(_img) {}

    class SVGRect : public cv::Rect {
    public:
        int level = 0;
        bool black = true;

        SVGRect(int const x,
                int const y,
                int const width = 1,
                int const height = 1,
                int const _level = 0,
                bool const _black = true) : cv::Rect(x, y, width, height), level(_level), black(_black) {
            if (width <= 0 || height <= 0) {
                std::cout << "Invalid rect:" << std::endl;
            }
        }
    };
    std::vector<SVGRect> markers;

    bool inLimit(SVGRect const& r) const {
        double scaled_limit_width = limit_width*5*scale;
        double scaled_limit_height = limit_height*5*scale;
        for (int ii = 0; ii < recursive; ++ii) {
            scaled_limit_width *= 5;
            scaled_limit_height *= 5;
        }
        return (r.x + r.width <= scaled_limit_width + .5) && (r.y + r.height <= scaled_limit_height + .5);
    }

    void convertMainMarkers() {
        markers.clear();
        for (int ii = 0; ii < img.rows; ++ii) {
            for (int jj = 0; jj < img.cols; ++jj) {
                if (img(ii,jj) == 0) {
                    markers.push_back(SVGRect(jj, ii, 1, 1, 0));
                }
            }
        }
    }

    void convertMainMarkers2() {
        markers.clear();
        // Add overlaps in y-direction to avoid rendering artifacts where two rectangles have zero distance
        // but no overlap.
        for (int jj = 0; jj < 5*limit_width; ++jj) {
            int ii_start = -1;
            for (int ii = 0; ii < 5*limit_height; ++ii) {
                if (img(ii,jj) == 0) {
                    if (ii_start < 0) {
                        ii_start = ii;
                    }
                }
                else {
                    if (ii_start >= 0 && ii - ii_start > 1) {
                        markers.push_back(SVGRect(jj, ii_start, 1, ii - ii_start, 0));
                    }
                    ii_start = -1;
                }
            }
            if (ii_start >= 0 && 5*limit_width - ii_start > 1) {
                markers.push_back(SVGRect(jj, ii_start, 1, 5*limit_height - ii_start, 0));
            }
        }
        // If a black pixel is found check neighbours and make rectangle as large as possible.
        for (int ii = 0; ii < 5*limit_height; ++ii) {
            int jj_start = -1;
            for (int jj = 0; jj < 5*limit_width; ++jj) {
                if (img(ii,jj) == 0) {
                    if (jj_start < 0) {
                        jj_start = jj;
                    }
                }
                else {
                    if (jj_start >= 0) {
                        markers.push_back(SVGRect(jj_start, ii, jj - jj_start, 1, 0));
                    }
                    jj_start = -1;
                }
            }
            if (jj_start >= 0) {
                markers.push_back(SVGRect(jj_start, ii, 5*limit_width - jj_start, 1, 0));
            }
        }
    }

    void scaleMarkers(double const factor) {
        for (auto& marker: markers) {
            marker.x *= factor;
            marker.y *= factor;
            marker.width *= factor;
            marker.height *= factor;
        }
    }

    void addSubmarkers(int const level) {
        int scale = 1;
        for (int ii = 1; ii < level; ++ii) {
            scale *= 5;
        }

        int w = img.size().width;
        int h = img.size().height;
        int ws = subsampling+2*ss_border;
        int w_hr = w*ws;;

        cv::Mat_<uint8_t> hr;
        resize(img, hr, Point2i(img.size())*ws, 0, 0, INTER_NEAREST);

        uint8_t * ptr_img = img.ptr<uchar>(0);
        uint8_t * ptr_hr = hr.ptr<uchar>(0);

        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++) {
                for(int j=ss_border;j<ws-ss_border;j++) {
                    for(int i=j%2+ss_border;i<ws-ss_border;i+=2){
                        ptr_hr[(y*ws+j)*w_hr+x*ws+i] = 255-ptr_hr[(y*ws+j)*w_hr+x*ws+i];
                        markers.push_back(SVGRect(x*scale, y*scale, 1, 1, level, ptr_hr[(y*ws+j)*w_hr+x*ws+i] != 0));
                    }
                }
            }
        }
        img = hr;
    }

    void addBG(std::ostream& out) {
        int min_x = markers.front().x;
        int min_y = markers.front().y;
        int max_x = min_x;
        int max_y = min_y;
        for (auto const& it : markers) {
            if (inLimit(it)) {
                min_x = std::min(min_x, it.x);
                min_y = std::min(min_y, it.y);
                max_x = std::max(max_x, it.x + it.width);
                max_y = std::max(max_y, it.y + it.width);
            }
        }
        out << std::string("<rect x='") + std::to_string(min_x)
               + "' y='" + std::to_string(min_y)
               + "' width='" + std::to_string(max_x - min_x)
               + "' height='" + std::to_string(max_y - min_y)
               + "' style='fill:#ffffff' />\n";
    }

    void addBGLimit(std::ostream& out) {
        int width = scale*limit_width*5;
        int height = scale*limit_height*5;
        for (int ii = 0; ii < recursive; ++ii) {
            width *= 5;
            height *= 5;
        }
        out << std::string("<rect x='0' y='0'")
               + " width='" + std::to_string(width)
               + "' height='" + std::to_string(height)
               + "' style='fill:#ffffff' />\n";
    }



    void writeSVG(std::string const filename) {
        std::ofstream svg(filename);
        svg <<
                "<?xml version='1.0' encoding='iso-8859-1'?>"
                "<!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 20001102//EN'"
                " 'http://www.w3.org/TR/2000/CR-SVG-20001102/DTD/svg-20001102.dtd'>"

                "<svg width='100%' height='100%'>\n";

        addBGLimit(svg);

        std::stringstream out[recursive+1];

        for (int ii = 0; ii <= recursive; ++ii) {
            out[ii] << "\n<g id='layer" << ii << "' inkscape:groupmode='layer' inkscape:label='Layer " << ii << "'>"
                    << "\n<g id='g" << ii << "'>\n";
        }

        for (auto const& rect : markers) {
            if (inLimit(rect)) {
                out[rect.level] << std::string("<rect x='") + std::to_string(rect.x)
                                   + "' y='" + std::to_string(rect.y)
                                   + "' width='" + std::to_string(rect.width)
                                   + "' height='" + std::to_string(rect.height)
                                   + "' style='fill:#" + (rect.black ? "000000" : "ffffff") + "' />\n";
            }
        }

        for (int ii = 0; ii <= recursive; ++ii) {
            out[ii] << "\n</g></g>\n";
            svg << out[ii].str();
        }

        svg << "</svg>\n";
    }

    void copySubmarkers(int const level) {
        int counter = 0;
        int scale = 25;
        for (int ii = 1; ii < level; ++ii) {
            scale *= 5;
        }
        for (int ii = 2; ii < scale*limit_height && ii < img.cols; ii += 5) {
            for (int jj = 2; jj < scale*limit_width && jj < img.rows; jj += 5) {
                bool const black = img(ii, jj) == 0;
                markers.push_back(SVGRect(jj, ii, 1, 1, level, black));
                counter++;
            }
        }
        std::cout << "copySubmarker: " << counter << " markers at level " << level << std::endl;
    }

    void enlargeSubmarkers(int const level, int const border, bool const black) {
        if (border < 1) {
            std::cout << "No white submarker enlargement" << std::endl;
            return;
        }
        size_t enlarged = 0;
        for (auto& it : markers) {
            if ((it.black == black) && it.level == level) {
                it.x -= border;
                it.y -= border;
                it.width += 2*border;
                it.height += 2*border;
                enlarged++;
            }
        }
        std::cout << "Enlarged " << enlarged << " " << (black ? "black": "white") << " submarkers" << std::endl;
    }

    static int origPxPerMarker(int const recurs) {
        int result = 5;
        for (int ii = 0; ii < recurs; ++ii) {
            result *= 5;
        }
        return result;
    }

    int origPxPerMarker() {
        return origPxPerMarker(recursive);
    }

    int offset_x = 0;
    int offset_y = 0;

    void writePDF(std::string const filename) {

        double const pdf_units_per_dot = 72.0/double(dpi);
        double const dots_per_cm = double(dpi)/2.54;

        double const width_cm  = double(scale * limit_width  * origPxPerMarker()) * 2.54 / dpi;
        double const height_cm = double(scale * limit_height * origPxPerMarker()) * 2.54 / dpi;

        std::cout << "Width in cm: " << width_cm << std::endl
                  << "Height in cm: " << height_cm << std::endl;

        double const width_cm_print  = 2*bleed + std::ceil(width_cm);
        double const height_cm_print = 2*bleed + std::ceil(height_cm);

        std::cout << "Print width in cm: " << width_cm_print << std::endl
                  << "Print height in cm: " << height_cm_print << std::endl;

        double const offset_x_cm = (width_cm_print - width_cm)/2;
        double const offset_y_cm = (height_cm_print - height_cm)/2;

        std::cout << "Print offset: " << offset_x_cm << "cm in x, " << offset_y_cm << "cm in y direction." << std::endl;

        double const offset_x_dot = offset_x_cm * dots_per_cm;
        double const offset_y_dot = offset_y_cm * dots_per_cm;

        std::cout << "Print offset: " << offset_x_dot << "dots in x, " << offset_y_dot << "dots in y direction." << std::endl;

        double const final_offset_x = std::round(offset_x_dot);
        double const final_offset_y = std::round(offset_y_dot);

        offset_x = int(final_offset_x);
        offset_y = int(final_offset_y);

        std::cout << "Final print offset: " << final_offset_x << "dots in x, " << final_offset_y << "dots in y direction." << std::endl;

        double const width_pdf_units  =  width_cm_print*72.0/2.54;
        double const height_pdf_units =  height_cm_print*72.0/2.54;

        int counter[5] = {0,0,0,0,0};

        PdfPainter painter;
        try {
            PdfStreamedDocument document(filename.c_str());
            PdfPage * page = document.CreatePage( PdfRect(0, 0, width_pdf_units, height_pdf_units) );

            if (!page) {
                throw std::runtime_error(std::string("Could not create pdf page: ") + std::to_string(ePdfError_InvalidHandle));
            }

            painter.SetPage( page );

            for (auto const& r : markers) {
                if (r.black) {
                    painter.SetColor(0,0,0);
                }
                else {
                    painter.SetColor(1,1,1);
                }
                painter.Rectangle(
                            pdf_units_per_dot * (r.x + final_offset_x),
                            height_pdf_units - pdf_units_per_dot * (r.y + r.height + final_offset_y),
                            pdf_units_per_dot * r.width,
                            pdf_units_per_dot * r.height);
                painter.Fill();
                counter[r.level]++;

            }

            for (size_t ii = 0; ii < 4; ++ii) {
                std::cout << "Rectangles at level #" << ii << ": " << counter[ii] << std::endl;
            }

            painter.FinishPage();
            document.Close();
        }
        catch ( PdfError & e ) {
            try {
                painter.FinishPage();
            } catch( ... ) {
            }
            throw e;
        }
    }

    void run(int const recursive, int const scale, int const border, std::string const filename) {
        convertMainMarkers2();

        this->recursive = recursive;
        this->scale = scale;

        for (int level = 1; level <= recursive; ++level) {
            scaleMarkers(5);
            //addSubmarkers(level);
            cv::Size const before = img.size();
            checker_recurse(img,img);
            copySubmarkers(level);
            std::cout << "Adding recursive submarkers, level " << level
                      << ". Image size before: " << before << ", after: " << img.size() << std::endl;
        }

        scaleMarkers(scale);

        enlargeSubmarkers(recursive, border, true);
        enlargeSubmarkers(recursive, border, false);

        enlargeSubmarkers(recursive-1, border*5, true);
        enlargeSubmarkers(recursive-1, border*5, false);

        writeSVG(filename + ".svg");

        writePDF(filename + ".pdf");

        img = img(cv::Rect(0,0,limit_width*origPxPerMarker(),limit_height*origPxPerMarker()));

        cv::resize(img, img, cv::Size(), scale, scale, INTER_NEAREST);


        size_t counter = 0;
        // Make white submarkers larger in the raster image.
        for (int ii = 2*scale; ii < img.rows; ii += 5*scale) {
            for (int jj = 2*scale; jj < img.cols; jj += 5*scale) {
                int const color = img(ii, jj);
                if (color == 255 || true) {
                    counter++;
                    for (int border_level = 1; border_level <= border; ++border_level) {
                        for (int kk = -border_level; kk < scale+border_level; ++kk) {
                            img(ii-border_level, jj+kk) = color;
                            img(ii+scale+border_level-1, jj+kk) = color;
                            img(ii+kk, jj-border_level) = color;
                            img(ii+kk, jj+scale+border_level-1) = color;
                        }
                    }
                }
            }
        }
        std::cout << "Enlarged " << counter << " white submarkers in raster image" << std::endl;

        cv::copyMakeBorder(img, img, offset_y, offset_y, offset_x, offset_x, cv::BORDER_CONSTANT, cv::Scalar(255));
    }
};



int main(int argc, char* argv[])
{
  if (argc != 4 && argc != 6 && argc <= 8) {
    std::cerr << "Usage:" << std::endl
              << "3-parameter mode: " << (argc > 0 ? argv[0] : "program_name")
              << " <page nr.> <number of recursive markers> <output image filename>"
              << std::endl
              << "5-parameter mode: " << (argc > 0 ? argv[0] : "program_name")
              << " <page nr.> <number of recursive markers> <width> <height> <output image filename>"
              << std::endl
              << "7+-parameter (SVG+pdf) mode: " << (argc > 0 ? argv[0] : "program_name")
              << " <page nr.>" << std::endl
              << " <number of recursive markers, 0-3>" << std::endl
              << " <width (in pages)> <height (in pages)>"
              << " <output image filename prefix>" << std::endl
              << " <scale, i.e. width of the smallest submarker in dots>" << std::endl
              << " <added border of the smallest white submarkers in dots. This helps when the printer makes black areas larger than it should>" << std::endl
              << " <limit width counted in level-0 markers>" << std::endl
              << " <limit height counted in level-0 markers>" << std::endl
              << " <printer resolution in dots per inch>" << std::endl
              << " <bleed in cm>"
              << std::endl
              << std::endl;
    return EXIT_FAILURE;
  }
  int page = atoi(argv[1]); 
  
  if (argc == 4) {
    Mat img = Mat::zeros(16*10, 32*5, CV_8UC1);
    img += 255;
    writemarker(img, page);
    //resize(img, img, Size(16*10*8, 16*10*8), 0, 0, INTER_NEAREST);
    recursive_markers = atoi(argv[2]);
    checker_add_recursive(img, img);
    imwrite(argv[3], img);
  }
  else if (argc == 6) {
    int w = atoi(argv[3]);
    int h = atoi(argv[4]);
    assert(w && h);
    Mat img = Mat::zeros(h*32*5, w*32*5, CV_8UC1);
    img += 255;
    
    for(int j=0;j<h;j++)
      for(int i=0;i<w;i++)
	writemarker(img, page+j*w+i, 32*5*i, 32*5*j);
    //resize(img, img, Size(w*32*5*1, h*32*5*1), 0, 0, INTER_NEAREST); 
    recursive_markers = atoi(argv[2]);
    checker_add_recursive(img, img);
    imwrite(argv[5], img);
  }
  else if (argc >= 8) {
      int w = atoi(argv[3]);
      int h = atoi(argv[4]);
      if (w <= 0) {
          std::cout << "width is " << w << " <= 0." << std::endl;
      }
      if (h <= 0) {
          std::cout << "height is " << h << " <= 0." << std::endl;
      }
      assert(w && h);
      Mat_<uint8_t> img(h*32*5, w*32*5, uint8_t(255));
      int limit_width = 32;
      int limit_height = 32;
      if (argc >= 9) {
          limit_width = limit_height = atoi(argv[8]);
      }
      if (argc >= 10) {
          limit_height = atoi(argv[9]);
      }

      for(int j=0;j<h;j++)
        for(int i=0;i<w;i++)
      writemarker(img, page+j*w+i, 32*5*i, 32*5*j);
      //resize(img, img, Size(w*32*5*1, h*32*5*1), 0, 0, INTER_NEAREST);
      recursive_markers = atoi(argv[2]);
      SVGMarker svg(img);
      svg.setLimitWidth(limit_width);
      svg.setLimitHeight(limit_height);
      int const svg_scale = atoi(argv[6]);
      int const added_border = atoi(argv[7]);
      int const dpi = argc >= 11 ? atoi(argv[10]) : 100;
      svg.setDPI(dpi);
      double const bleed = argc >= 12 ? atof(argv[11]) : 0;
      svg.setBleed(bleed);
      std::string const prefix(argv[5]);
      std::cout << "Parameters: " << std::endl
                << "Recursive levels: " << recursive_markers << std::endl
                << "svg scale: " << svg_scale << std::endl
                << "added border for smallest white submarkers: " << added_border << std::endl
                << "output file prefix: " << prefix << std::endl
                << "width in main markers: " << limit_width << std::endl
                << "height in main markers: " << limit_height << std::endl
                << "resolution in dpi (dots per inch): " << dpi << std::endl
                << "bleed in mm: " << bleed << std::endl
                << std::endl;
      svg.run(recursive_markers, // recursive levels
              svg_scale,
              added_border,
              prefix);


      cv::Mat channels[3] = {img, img, img};
      cv::Mat colored;
      cv::merge(channels, 3, colored);

      imwrite(std::string(argv[5]) + ".png", colored);
      imwrite(std::string(argv[5]) + ".jpg", colored, {
          cv::IMWRITE_JPEG_QUALITY, 100,
                  cv::IMWRITE_JPEG_OPTIMIZE, 1
      });
  }

  
  return 0;
  
}
