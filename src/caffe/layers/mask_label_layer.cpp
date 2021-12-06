// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/mask_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_out = bottom[0]->num();
  start_index = this->layer_param_.mask_label_param().start() - 1;
  end_index = this->layer_param_.mask_label_param().end() - 1;
  num_label = end_index - start_index + 1;
}

template <typename Dtype>
void MaskLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape;
  shape.push_back(num_out);
  shape.push_back(num_label);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MaskLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num_bottom = count/num_out;
    for (int i = 0; i < num_out; ++i) {
      for(int j=0; j< num_label; ++j)
            top_data[i*num_label + j] = bottom_data[i*num_bottom+j];
    }
 // for (int i = 0; i < num_out; ++i) {
   //   for(int j=0; j< num_label; ++j)
       //     LOG(INFO)<<"Top data is "<<top_data[i*num_label + j] \
     // <<" Bottom data is "<<bottom_data[i*num_bottom+j];
   // }
}

template <typename Dtype>
void MaskLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaskLabelLayer);
#endif

INSTANTIATE_CLASS(MaskLabelLayer);
REGISTER_LAYER_CLASS(MaskLabel);

}  // namespace caffe
