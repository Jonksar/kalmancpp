/*
 * --------------------------------------------------
 * File Name : UnscentedKalmanFilter.hpp
 * Creation Date : 2019-10-26 Sat 06:40 am
 * Last Modified : 2019-11-02 Sat 07:32 pm
 * Created By : Joonatan Samuel
 * --------------------------------------------------
 */

#include <eigen3/Eigen/Cholesky>

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

        MerweScaledSigmaPoints(T alpha=0.01, T beta=2, T kappa=0) : alpha(alpha), beta(beta), kappa(kappa) {
            compute_weights();
        };

        ~MerweScaledSigmaPoints() {
        };

        // Computes the weights for the scaled unscented Kalman filter.
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

        size_t num_sigmas( )
        {
            return 2*n + 1;
        };

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

    UnscentedTransform::Result unscented_transform(state_matrix_size_t x,
            state_square_size_t P,
            transform_func_t Hx,
            argTs... fargs)
    {
        // Initialize result
        Result result;

        // Sigma points before transform
        auto pts = sigma_point_generator.sigma_points(x, P);

        // Dynamic memory for points after transform
        state_matrix_size_t transformed[sigma_point_generator.num_sigmas()];

        // Pass each state through transform of Hx
        for (size_t i = 0; i < sigma_point_generator.num_sigmas(); i++)
        {
            auto pt  = pts[i];
            std::cout << "i: " << i << std::endl << pt << std::endl;
            transformed[i] = Hx(pt, fargs...);
            std::cout << "i_t: " << i << std::endl << transformed[i] << std::endl;
        };

        // 1. Calculate the mean
        result.mean.setZero();
        for (size_t i = 0; i < sigma_point_generator.num_sigmas(); i++)
        {
            result.mean += sigma_point_generator.Wm[i] * transformed[i];
        };

        // 2. Calculate the covariance
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

template <class T, size_t dim_x, size_t dim_z, size_t dim_u, size_t n_sigma_pt>
class UnscentedKalmanFilter {
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
        Eigen::Matrix<T, dim_x, dim_u>                  B;            // control transition matrix


        state_matrix_size_t stateTransforms[n_sigma_pt];


        UnscentedKalmanFilter() {
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

        ~UnscentedKalmanFilter() {};
};
