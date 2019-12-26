/*
 * --------------------------------------------------
 * File Name : UnscentedKalmanFilter.hpp
 * Creation Date : 2019-12-26 Sat 09:31 am
 * Last Modified : 2019-12-26 Thu 11:13 pm
 * Created By : Joonatan Samuel
 * --------------------------------------------------
 */

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <stdio.h>
#include <iostream>
#include "ExtendedKalmanFilterStatic.cpp"
#include "UnscentedKalmanFilterStatic.cpp"
#include "catch.hpp"

static const int dim_x = 2;

TEST_CASE("MerweScaledSigmaPoints", "Weights") {
    /*
     *
     * Python:
     * >>> from filterpy.kalman import MerweScaledSigmaPoints
     * >>> MerweScaledSigmaPoints(2, 0.1, 2., 0.)
     * MerweScaledSigmaPoints object
     * n = 2
     * alpha = 0.1
     * beta = 2.0
     * kappa = 0.0
     * Wm = [-99.  25.  25.  25.  25.]
     * Wc = [-96.01  25.    25.    25.    25.  ]
     * subtract = <ufunc 'subtract'>
     * sqrt = <function cholesky at 0x1104ee8c0>
     *
     * Inputs:
     *   n = 2
     *   alpha = 0.1
     *   beta = 2.0
     *   kappa = 0.0
     *
     * Weights:
     *   Wm = [-99.  25.  25.  25.  25.]
     *   Wc = [-96.01  25.    25.    25.    25.  ]
    */
    MerweScaledSigmaPoints<double, dim_x> pts_generator(0.1, 2.0, 0.0);

    REQUIRE(pts_generator.Wm[0] == Approx(-99.  ));
    REQUIRE(pts_generator.Wc[0] == Approx(-96.01));

    for (int i = 1; i < 2*dim_x+1; ++i) {
        REQUIRE(pts_generator.Wm[i] == Approx(25.));
        REQUIRE(pts_generator.Wc[i] == Approx(25.));
    }
};

Eigen::Matrix<double, dim_x, 1> update(const Eigen::Matrix<double, dim_x, 1>& x, const double& dt)
{
    Eigen::Matrix<double, dim_x, 1>  x_copy = x;
    x_copy(0) = x_copy(0);
    x_copy(1) = x_copy(0) + x_copy(1) * dt;

    return x_copy;
};

TEST_CASE("UnscentedTransform", "Basic update") {
    /* Python:
    >>> import numpy as np
    >>> from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    >>>
    >>> dt = 0.1
    >>> points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=0.)
    >>> def fx(x, dt):
    ...  return np.array([x[0], x[0] + dt * x[1]])
    ...
    >>> kf = UnscentedKalmanFilter(dim_x=2, dim_z=2, dt=dt, fx=fx, hx=None, points=points)
    >>> kf.x = np.array([1, 1])
    >>> kf.P
    array([[1., 0.],
           [0., 1.]])
    >>> kf.predict()
    >>> kf.x
    array([1. , 1.1])

    THIS DIFFERS FROM ASSERT BELOW, BECAUSE +Q is missing.
    >>> kf.P
    array([[2.  , 1.  ],
           [1.  , 2.01]])
   */
    UnscentedKalmanFilter<double, dim_x, 2, 0> filter;
    filter.x << 1, 1;
    REQUIRE(filter.x(0) == Approx(1.));
    REQUIRE(filter.x(1) == Approx(1.));
    REQUIRE(filter.P(0, 0) == Approx(1.));
    REQUIRE(filter.P(1, 1) == Approx(1.));
    REQUIRE(filter.P(0, 1) == Approx(0.));
    REQUIRE(filter.P(1, 0) == Approx(0.));

    UnscentedTransform<double, 2, double> UT;

    auto result = UT.unscented_transform(
            filter.x,
            filter.P,
            update,
            0.1);

    REQUIRE(result.mean(0) == Approx(1.));
    REQUIRE(result.mean(1) == Approx(1.1));
    REQUIRE(result.covariance(0, 0) == Approx(1.));
    REQUIRE(result.covariance(1, 1) == Approx(1.01));
    REQUIRE(result.covariance(0, 1) == Approx(1.));
    REQUIRE(result.covariance(1, 0) == Approx(1.));
};

