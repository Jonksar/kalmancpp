/*
 * --------------------------------------------------
 * File Name : ExtendedKalmanFilter.hpp
 * Creation Date : 2019-10-26 Sat 06:40 am
 * Last Modified : 2019-10-26 Sat 01:33 pm
 * Created By : Joonatan Samuel
 * --------------------------------------------------
 */


#ifndef EXTENDEDKALMANFILTERSTATIC_HPP_S9PB4JQT
#define EXTENDEDKALMANFILTERSTATIC_HPP_S9PB4JQT

#include <math.h>
#include <functional>

#ifdef ENABLE_EIGEN_ASSERTS
#undef NDEBUG
#endif
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

template <class T, size_t dim_x, size_t dim_z, size_t dim_u>
class ExtendedKalmanFilter {
    typedef Eigen::Matrix<T, dim_x, 1>  state_matrix_size_t;
    typedef Eigen::Matrix<T, dim_x, dim_x>  state_square_size_t;
    typedef Eigen::Matrix<T, dim_x, 1>  measurement_matrix_t;
    typedef Eigen::Matrix<T, dim_z, dim_z>  measurement_square_size_t;
    typedef Eigen::Matrix<T, dim_x, dim_z>  state_to_measurement_size_t;

    public:
        ExtendedKalmanFilter::state_matrix_size_t       x;            // state
        ExtendedKalmanFilter::state_matrix_size_t       U;            // control vector
        ExtendedKalmanFilter::state_square_size_t       P;            // uncertainty covariance
        ExtendedKalmanFilter::state_square_size_t       F;            // state transition matrix
        ExtendedKalmanFilter::measurement_square_size_t R;            // state uncertainty
        ExtendedKalmanFilter::state_square_size_t       Q;            // process uncertainty
        ExtendedKalmanFilter::measurement_matrix_t      y;            // residual
        ExtendedKalmanFilter::measurement_square_size_t S;            // measurement covariance
        Eigen::Matrix<T, dim_x, dim_u>                  B;            // control transition matrix

        ExtendedKalmanFilter() {
            // Public variable nulling
            x.setZero();                    // state
            P.setIdentity();                // state uncertainty covariance
            U.setZero();                    // control vector
            B.setZero();                    // control transition matrix
            F.setIdentity();                // state transition matrix

            R.setIdentity();                // measurement uncertainty
            Q.setIdentity();                // process uncertainty

            S.setIdentity();                // measurement covariance

            y.setZero();                    // residual


            // Private memory allocation
            _I   .setIdentity();          // identity
            _PHT .setZero();              // Allocate memory for intermediate values
            _K   .setZero();              // Allocate memory for intermediate values
            _I_KH.setZero();              // Allocate memory for intermediate values
            _H   .setZero();              // Allocate memory for intermediate values
        };

        ~ExtendedKalmanFilter() {};

        // Predict phase of Kalman filter, in place
        void predict() {
            x = F * x + U;
            P = F * P * F.transpose() + Q;

            U.setZero();
        };

        // Predict phase of Kalman filter
        void _predict(state_matrix_size_t& _x, state_square_size_t& _P)
        {
            _x = F * x + U;
            _P = F * P * F.transpose() + Q;
        };

        // Update phase of Kalman filter
        void update(const ExtendedKalmanFilter::measurement_matrix_t& z,
                    std::function<ExtendedKalmanFilter::state_square_size_t(ExtendedKalmanFilter::state_matrix_size_t x)> HJacobian,
                    std::function<ExtendedKalmanFilter::state_matrix_size_t(ExtendedKalmanFilter::state_matrix_size_t x)> Hx)
        {
            _H = HJacobian(x);

            _PHT = P * _H.transpose();
            S = (_H * _PHT + R).inverse();

            // optimal kalman gain
            _K = _PHT * S;
            _I_KH = _I - _K * _H;

            y = z - Hx(x);

            // update state & covariance
            x = x + _K * y;
            P = _I_KH * P * _I_KH.transpose() + _K * R * _K.transpose();
        };

        float distance(const ExtendedKalmanFilter::measurement_matrix_t& z,
            std::function<ExtendedKalmanFilter::state_square_size_t(ExtendedKalmanFilter::state_matrix_size_t x)> HJacobian,
            std::function<ExtendedKalmanFilter::state_matrix_size_t(ExtendedKalmanFilter::state_matrix_size_t x)> Hx)
        {
            auto _x = F * x + U;
            auto _P = F * P * F.transpose() + Q;

            _H = HJacobian(_x);
            _PHT = _P * _H.transpose();

            S = (_H * _PHT + R).inverse();

            // optimal kalman gain
            _K = _PHT * S;
            _I_KH = _I - _K * _H;
            y = z - Hx(_x);

            return sqrt( (y.transpose() * S * y)(0, 0) );
        };

        float measurement_distance(ExtendedKalmanFilter::measurement_matrix_t z,
            std::function<ExtendedKalmanFilter::state_matrix_size_t(ExtendedKalmanFilter::state_matrix_size_t x)> Hx)
        {
            return (z - Hx(x)).norm();
        };

    private:

        // Private memory allocation
        ExtendedKalmanFilter::state_square_size_t             _I   ;
        ExtendedKalmanFilter::state_to_measurement_size_t     _PHT ;
        ExtendedKalmanFilter::state_to_measurement_size_t     _K   ;
        ExtendedKalmanFilter::state_square_size_t             _I_KH;
        ExtendedKalmanFilter::state_square_size_t             _H   ;
};


#endif /* end of include guard: EXTENDEDKALMANFILTERSTATIC_HPP_S9PB4JQT */
