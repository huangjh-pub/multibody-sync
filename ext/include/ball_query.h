#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);

at::Tensor ball_query_var_radius(at::Tensor new_xyz, at::Tensor xyz, at::Tensor radius, const int nsample);
