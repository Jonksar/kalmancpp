/*
 * --------------------------------------------------
 * File Name :
 * Creation Date : 2019-10-26 Sat 06:42 am
 * Last Modified : 2019-12-26 Thu 09:30 pm
 * Created By : Joonatan Samuel
 * --------------------------------------------------
 */


#include <stdio.h>
#include <iostream>
#include "ExtendedKalmanFilterStatic.cpp"
#include "UnscentedKalmanFilterStatic.cpp"

static const int dim_x = 2;


template <class T>
Eigen::Matrix<T, dim_x, 1> update(const Eigen::Matrix<T, dim_x, 1>& x, const T& dt)
{
    Eigen::Matrix<double, dim_x, 1>  x_copy = x;
    x_copy(0) = x_copy(0) + x_copy(1) * dt;

    return x_copy;
};

int main(int argc, char *argv[])
{
    UnscentedKalmanFilter<double, dim_x, 2, 0> filter;
    filter.x << 1, 1;

    std::cout << std::endl << "x: " << std::endl << filter.x << std::endl;
    std::cout << std::endl << "P: " << std::endl << filter.P << std::endl;

    UnscentedTransform<double, 2, double> UT;

    auto result = UT.unscented_transform(
            filter.x,
            filter.P,
            update<double>,
            1.0f);

    std::cout << "After update: " << std::endl << std::endl;
    std::cout << "result.x" << std::endl;
    std::cout << result.mean << std::endl;
    std::cout << "result.P" << std::endl;
    std::cout << result.covariance << std::endl;


    result = UT.unscented_transform(
            result.mean,
            result.covariance,
            update<double>,
            1.0f);

    std::cout << "After update: " << std::endl << std::endl;
    std::cout << "result.x" << std::endl;
    std::cout << result.mean << std::endl;
    std::cout << "result.P" << std::endl;
    std::cout << result.covariance << std::endl;

    // UnscentedTransform::Result unscented_transform(state_matrix_size_t x,
    //         state_square_size_t P,
    //         transform_func_t Hx,
    //         fargs...)
    // typedef std::function< state_matrix_size_t(const state_matrix_size_t& x, fargs...)> transform_func_t;

    return 0;
}

