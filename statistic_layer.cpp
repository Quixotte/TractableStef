#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TestStatisticLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.teststatistic_param().top_k();
  phase_ = Caffe::phase();
}


template <typename Dtype>
void TestStatisticLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void TestStatisticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  phase_ = Caffe::phase();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

    // layout of line
    // true_label prob_0 label_0 prob_1 label_1 prob_2 label_2 prob_3 label_3
    if (phase_ == Caffe::TEST) {
      std::cout << "[>>] ";
    } else {
      std::cout << "[<<] ";
    }
    std::cout << static_cast<float>(bottom_label[i]) << "; ";
    for (int k = 0; k < top_k_; k ++) {
      std::cout << bottom_data_vector[k].first << "; ";
      std::cout << bottom_data_vector[k].second << "; ";
    }
    std::cout << std::endl;
  }
}

INSTANTIATE_CLASS(TestStatisticLayer);
REGISTER_LAYER_CLASS(TESTSTATISTIC, TestStatisticLayer);
}  // namespace caffe