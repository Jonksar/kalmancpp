/*
 * --------------------------------------------------
 * File Name : UnscentedKalmanFilter.hpp
 * Creation Date : 2019-10-26 Sat 06:40 am
 * Last Modified : 2019-12-26 Thu 08:59 pm
 * Created By : Joonatan Samuel
 * --------------------------------------------------
 */

#include <eigen3/Eigen/Cholesky>
/**
 * Generates sigma points and weights according to Van der Merwe’s 2004 dissertation[1] for the UnscentedKalmanFilter class.
 * It parametizes the sigma points using alpha, beta, kappa terms, and is the version seen in most publications.
 * Unless you know better, this should be your default choice
 *
 * @tparam T numeric type, usually double or float
 * @tparam n number of dimensions sigma points are sampled from
 */
template <class T, int n>
class MerweScaledSigmaPoints {
    public:
        typedef Eigen::Matrix<T, n, 1> state_matrix_size_t;
        typedef Eigen::Matrix<T, n, n>  state_square_size_t;

        T alpha;
        T beta;
        T kappa;

        T Wc[2*n+1];
        T Wm[2*n+1];
        state_matrix_size_t sigma_points_array[2*n+1];

        /**
         * Initialization function that by that will compute weights for sigma points
         *
         * @param[in] alpha [T numeric] Determins the spread of the sigma points around the mean. Usually a small positive value (1e-3) according to [3].
         * @param[in] beta  [T numeric] Incorporates prior knowledge of the distribution of the mean. For Gaussian x beta=2 is optimal, according to [3].
         * @param[in] kappa [T numeric, default=0.0] Secondary scaling parameter usually set to 0 according to [4], or to 3-n according to [5].
         */
        MerweScaledSigmaPoints(T alpha=0.01, T beta=2, T kappa=0) : alpha(alpha), beta(beta), kappa(kappa) {
            compute_weights();
        };

        ~MerweScaledSigmaPoints() {
        };

        /** Computes the weights for the scaled unscented Kalman filter.
         */
        void compute_weights() {
            T lambda_ = std::pow(alpha, 2) * (T(n)+kappa) - T(n);

            T c = .5 / (T(n) + lambda_);
            // std::cout << "MerweScaledSigmaPoints.compute_weights: " << c << " " << lambda_ << std::endl;
            Wc[0] = lambda_ / (T(n) + lambda_) + (1 - std::pow(alpha, 2) + beta);
            Wm[0] = lambda_ / (T(n) + lambda_);

            for (int i = 1; i < 2*n+1; i++) {
                Wc[i] = c;
                Wm[i] = c;
            }

            // std::cout << "MerweScaledSigmaPoints weights computed" << std::endl;
            return;
        };

        inline size_t num_sigmas( )
        {
            return 2*n + 1;
        };

        /** Main compute function of the class. Given a vector and covariance, generate 2*n + 1 sigma points.
         *
         * @param[in] x input state vector
         * @param[in] P covariance of x
         *
         * @return the sigma points
         */
        state_matrix_size_t* sigma_points(const state_matrix_size_t& x, const state_square_size_t& P)
        {
            T lambda_ = std::pow(this->alpha, 2) * (n + this->kappa) - n;
            state_square_size_t U = ((lambda_ + n)*P).llt().matrixL();

            sigma_points_array[0] = x;

            for (size_t i = 0; i < n; i++) {
                sigma_points_array[i+1  ]  = (x + U.block(0, i, n, 1));
                sigma_points_array[n+i+1]  = (x - U.block(0, i, n, 1));
            };

            return sigma_points_array;
        };
};


/**
 * Printing operator for MerweScaledSigmaPoints
 *
 * @tparam T numeric type, usually double or float
 * @tparam n number of dimensions sigma points are sampled from
 */
template <class T, size_t n>
std::ostream & operator << (std::ostream &out, const MerweScaledSigmaPoints<T, n>& c)
{
    return out <<
        "MerweScaledSigmaPoints(" <<
        "n="     <<  n << ", " <<
        "alpha=" <<  c.alpha << ", " <<
        "beta="  <<  c.beta << ", " <<
        "kappa=" <<  c.kappa << ")" << std::endl;
    return out;
};


/**
 * Implements Unscented Transform that is more robust to non-linearities than linear approximation used in ExtendedKalmanFilters.
 * Given current state, covariance and transform_function; Calculate new state and covariance.
 *
 * @tparam T numeric type, usually double or float
 * @tparam n state dimensions size.
 * @tparam argTs arguments for update functions, usually time dimension will get passed here.
 */
template <class T, int n, class ... argTs>
class UnscentedTransform {
    public:
    MerweScaledSigmaPoints<T, n> sigma_point_generator;
    typedef typename MerweScaledSigmaPoints<T, n>::state_matrix_size_t state_matrix_size_t;;
    typedef typename MerweScaledSigmaPoints<T, n>::state_square_size_t state_square_size_t;
    typedef std::function< state_matrix_size_t(const state_matrix_size_t& x, argTs...)> transform_func_t;

    struct Result {
        typename MerweScaledSigmaPoints<T, n>::state_matrix_size_t mean;
        typename MerweScaledSigmaPoints<T, n>::state_square_size_t covariance;
    };

    UnscentedTransform() {};
    ~UnscentedTransform() {};

    /**
     * Main working function of the class.
     * Given current state, covariance and transform_function; Calculate new state and covariance.
     *
     * @param[in] x state
     * @param[in] P covariance of state
     * @param[in] Hx Transform of x
     * @param[in] function parameters of Hx, usually time delta will get passed here
     *
     * @return UnscentedTransform::Result that holds mean and covariance.
     */
    UnscentedTransform::Result unscented_transform(state_matrix_size_t x,
            state_square_size_t P,
            transform_func_t Hx,
            argTs... fargs)
    {
        // Initialize result
        Result result;

        // Sigma points before transform
        auto pts = sigma_point_generator.sigma_points(x, P);

        // Memory alloc for points after transform
        // TODO: Make static
        state_matrix_size_t transformed[sigma_point_generator.num_sigmas()];

        // Pass each state through transform of Hx
        for (size_t i = 0; i < sigma_point_generator.num_sigmas(); i++)
        {
            auto pt  = pts[i];
            std::cout << "i: " << i << std::endl << pt << std::endl;
            transformed[i] = Hx(pt, fargs...);
            std::cout << "i_t: " << i << std::endl << transformed[i] << std::endl;
        };

        // 1. Calculate the mean,
        //               as the weighted sum of evolved states.
        result.mean.setZero();
        for (size_t i = 0; i < sigma_point_generator.num_sigmas(); i++)
        {
            result.mean += sigma_point_generator.Wm[i] * transformed[i];
        };

        // 2. Calculate the covariance,
        //              as the weighted covariance of the points vs the mean
        result.covariance.setZero();
        for (size_t i = 0; i < sigma_point_generator.num_sigmas(); i++)
        {
            // P += Wc[k] * np.outer(y, y)
            auto y = transformed[i] - result.mean;
            result.covariance += (y * y.transpose()) * sigma_point_generator.Wc[i];

            std::cout << "result.covariance" << std::endl;
            std::cout << result.covariance << std::endl;
        };

        return result;
    };
};

/**
 * Implements the Scaled Unscented Kalman filter (UKF) as defined by Simon Julier in [1], using the formulation
 * provided by Wan and Merle in [2]. This filter scales the sigma points to avoid strong nonlinearities.
 *
 * @tparam T numeric type, usually double or float
 * @tparam dim_x state dimensions size.
 * @tparam dim_z measurement dimension size.
 * @tparam dim_u control dimension size.
 */
template <class T, size_t dim_x, size_t dim_z, size_t dim_u>
class UnscentedKalmanFilter {
    // Static types handles
    typedef Eigen::Matrix<T, dim_x, 1>  state_matrix_size_t;
    typedef Eigen::Matrix<T, dim_x, dim_x>  state_square_size_t;
    typedef Eigen::Matrix<T, dim_x, 1>  measurement_matrix_t;
    typedef Eigen::Matrix<T, dim_z, dim_z>  measurement_square_size_t;
    typedef Eigen::Matrix<T, dim_x, dim_z>  state_to_measurement_size_t;

    private:
        UnscentedKalmanFilter::state_square_size_t             _I   ;
        UnscentedKalmanFilter::state_to_measurement_size_t     _PHT ;
        UnscentedKalmanFilter::state_to_measurement_size_t     _K   ;
        UnscentedKalmanFilter::state_square_size_t             _I_KH;
        UnscentedKalmanFilter::state_square_size_t             _H   ;

    public:
        UnscentedKalmanFilter::state_matrix_size_t       x;            // state
        UnscentedKalmanFilter::state_matrix_size_t       U;            // control vector
        UnscentedKalmanFilter::state_square_size_t       P;            // uncertainty covariance
        UnscentedKalmanFilter::state_square_size_t       F;            // state transition matrix
        UnscentedKalmanFilter::measurement_square_size_t R;            // state uncertainty
        UnscentedKalmanFilter::state_square_size_t       Q;            // process uncertainty
        UnscentedKalmanFilter::measurement_matrix_t      y;            // residual
        UnscentedKalmanFilter::measurement_square_size_t S;            // measurement covariance
        Eigen::Matrix<T, dim_x, dim_u>                   B;            // control transition matrix

        // state_matrix_size_t stateTransforms[n_sigma_pt];

        UnscentedKalmanFilter() {
            // Public variable nulling
            // TODO: Can Eigen provide allocation with zeros?
            x.setZero();                    // state
            P.setIdentity();                // state uncertainty covariance
            U.setZero();                    // control vector
            B.setZero();                    // control transition matrix
            F.setIdentity();                // state transition matrix

            R.setIdentity();                // measurement uncertainty
            Q.setIdentity();                // process uncertainty

            S.setIdentity();                // measurement covariance

            y.setZero();                    // residual


            // Private variable nulling
            _I   .setIdentity();          // identity
            _PHT .setZero();              // Allocate memory for intermediate values
            _K   .setZero();              // Allocate memory for intermediate values
            _I_KH.setZero();              // Allocate memory for intermediate values
            _H   .setZero();              // Allocate memory for intermediate values
        };

        ~UnscentedKalmanFilter() {};
};
