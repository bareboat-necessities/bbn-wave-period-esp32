#pragma once

/*
  Stable drop-in replacement for Kalman3D_Wave_II.
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#else
#include <ArduinoEigenDense.h>
#endif

#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using Eigen::Matrix;

template<typename T>
inline Eigen::Quaternion<T> quat_from_delta_theta(const Eigen::Matrix<T,3,1>& dtheta) {
    const T theta = dtheta.norm();
    const T half_theta = T(0.5) * theta;

    T w, k;
    if (theta < T(1e-2)) {
        const T t2 = theta * theta;
        const T t4 = t2 * t2;

        w = T(1);
        w = std::fma(-t2, T(1)/T(8), w);
        w = std::fma( t4, T(1)/T(384), w);

        k = T(0.5);
        k = std::fma(-t2, T(1)/T(48), k);
        k = std::fma( t4, T(1)/T(3840), k);
    } else {
        w = std::cos(half_theta);
        k = std::sin(half_theta) / theta;
    }

    const Eigen::Matrix<T,3,1> v = k * dtheta;
    Eigen::Quaternion<T> q(w, v.x(), v.y(), v.z());
    q.normalize();
    return q;
}

template<typename T, int N>
static inline void project_psd(Eigen::Matrix<T,N,N>& S,
                               T rel_floor = T(1e-12)) {
    S = T(0.5) * (S + S.transpose());

    T scale = T(0);
    for (int i = 0; i < N; ++i) {
        const T d = S(i,i);
        if (std::isfinite(d)) scale = std::max(scale, std::abs(d));
    }
    if (!(scale > T(0)) || !std::isfinite(scale)) scale = T(1);

    const T lam_floor = std::max(rel_floor * scale,
                                 std::numeric_limits<T>::min());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!std::isfinite(S(i,j))) S(i,j) = (i == j) ? lam_floor : T(0);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N>> es(S);
    if (es.info() != Eigen::Success) {
        S.setZero();
        S.diagonal().array() = lam_floor;
        return;
    }

    Eigen::Matrix<T,N,1> lam = es.eigenvalues();
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(lam(i)) || lam(i) < lam_floor) lam(i) = lam_floor;
    }

    S = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
    S = T(0.5) * (S + S.transpose());
}

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class Kalman3D_Wave_II {
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;
    static constexpr int LIN_N  = 6 + (with_accel_bias ? 3 : 0);
    static constexpr int NX     = BASE_N + LIN_N;

    static constexpr int OFF_V    = BASE_N + 0;
    static constexpr int OFF_P    = BASE_N + 3;
    static constexpr int OFF_BAW  = with_accel_bias ? (BASE_N + 6) : -1;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, BASE_N, BASE_N> MatrixBaseN;
    typedef Matrix<T, NX, NX> MatrixNX;
    typedef Matrix<T, NX, 3> MatrixNX3;

    static constexpr T STD_GRAVITY = T(9.80665);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct MeasDiag3 {
        Vector3 r = Vector3::Zero();
        Matrix3 S = Matrix3::Zero();
        T nis = std::numeric_limits<T>::quiet_NaN();
        bool accepted = false;
    };

    const MeasDiag3& lastAccDiag() const noexcept { return last_acc_diag_; }
    const MeasDiag3& lastMagDiag() const noexcept { return last_mag_diag_; }

    Kalman3D_Wave_II(Vector3 const& sigma_a,
                     Vector3 const& sigma_g,
                     Vector3 const& sigma_m,
                     T Pq0 = T(5e-4),
                     T Pb0 = T(1e-6),
                     T b0  = T(1e-11),
                     T R_p0_noise_var = T(1.5),
                     T R_v0_noise_var = T(0.3),
                     T gravity_magnitude = T(STD_GRAVITY))
      : gravity_magnitude_(gravity_magnitude),
        qref(Eigen::Quaternion<T>::Identity()),
        Rmag(sigma_m.array().square().matrix().asDiagonal()),
        Qbase(initialize_Q(sigma_g, b0)),
        Racc(sigma_a.array().square().matrix().asDiagonal()) {

        Rmag = T(0.5) * (Rmag + Rmag.transpose());
        Racc = T(0.5) * (Racc + Racc.transpose());

        R_p0 = Matrix3::Identity() * std::max(T(1e-6), R_p0_noise_var);
        R_v0 = Matrix3::Identity() * std::max(T(1e-6), R_v0_noise_var);

        MatrixBaseN Pbase; Pbase.setZero();
        Pbase.template topLeftCorner<3,3>() =
            Matrix3::Identity() * std::max(T(1e-10), Pq0);
        if constexpr (with_gyro_bias) {
            Pbase.template block<3,3>(3,3) =
                Matrix3::Identity() * std::max(T(1e-12), Pb0);
        }

        xext.setZero();
        Pext.setZero();
        Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

        if constexpr (with_accel_bias) {
            Pext.template block<3,3>(OFF_BAW, OFF_BAW) =
                Matrix3::Identity() * initial_baw_std_(0) * initial_baw_std_(0);
        }

        // More conservative than the previous 10 m/s initial velocity uncertainty.
        const T sigma_v0 = T(1.0);
        const T sigma_p0 = T(20.0);
        set_initial_linear_uncertainty(sigma_v0, sigma_p0);

        {
            Vector3 sigma_acc0 = get_Racc_std();

            const T sigma_p00 = std::sqrt(std::max(T(1e-12), R_p0_noise_var));
            const T sigma_v00 = std::sqrt(std::max(T(1e-12), R_v0_noise_var));
            Vector3 sigma_p00v = Vector3::Constant(sigma_p00);
            Vector3 sigma_v00v = Vector3::Constant(sigma_v00);

            log_sigma_acc_f_.x = clamp_pos_vec_(sigma_acc0, sigma_acc_min_, sigma_acc_max_).array().log().matrix();
            log_sigma_p0_f_.x  = clamp_pos_vec_(sigma_p00v, sigma_p0_min_, sigma_p0_max_).array().log().matrix();
            log_sigma_v0_f_.x  = clamp_pos_vec_(sigma_v00v, sigma_v0_min_, sigma_v0_max_).array().log().matrix();

            log_sigma_acc_f_.P = Vector3::Constant(T(0.10) * T(0.10));
            log_sigma_p0_f_.P  = Vector3::Constant(T(0.15) * T(0.15));
            log_sigma_v0_f_.P  = Vector3::Constant(T(0.15) * T(0.15));

            const T rw_sig_acc = T(0.02);
            const T rw_sig_p0  = T(0.02);
            const T rw_sig_v0  = T(0.02);
            log_sigma_acc_f_.q = Vector3::Constant(rw_sig_acc * rw_sig_acc);
            log_sigma_p0_f_.q  = Vector3::Constant(rw_sig_p0 * rw_sig_p0);
            log_sigma_v0_f_.q  = Vector3::Constant(rw_sig_v0 * rw_sig_v0);

            const T cmd_sig_acc = T(0.10);
            const T cmd_sig_p0  = T(0.15);
            const T cmd_sig_v0  = T(0.15);
            log_sigma_acc_f_.r = Vector3::Constant(cmd_sig_acc * cmd_sig_acc);
            log_sigma_p0_f_.r  = Vector3::Constant(cmd_sig_p0 * cmd_sig_p0);
            log_sigma_v0_f_.r  = Vector3::Constant(cmd_sig_v0 * cmd_sig_v0);
        }

        refresh_baw_rw_from_sigma_acc_();
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    void initialize_from_acc_mag(Vector3 const& acc_body, Vector3 const& mag_body) {
        const Vector3 acc = deheel_vector_(acc_body);
        const Vector3 mag = deheel_vector_(mag_body);

        const T anorm = acc.norm();
        if (!(anorm > T(1e-8))) {
            throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
        }
        const Vector3 z_world = -(acc / anorm);

        Vector3 x_world = mag - (mag.dot(z_world)) * z_world;
        if (!(x_world.norm() > T(1e-8))) {
            throw std::runtime_error("Magnetometer vector parallel to gravity — cannot initialize yaw");
        }
        x_world.normalize();
        Vector3 y_world = z_world.cross(x_world);
        if (!(y_world.norm() > T(1e-8))) {
            throw std::runtime_error("Degenerate initialization basis");
        }
        y_world.normalize();
        x_world = y_world.cross(z_world).normalized();

        Matrix3 R_wb0;
        R_wb0.col(0) = x_world;
        R_wb0.col(1) = y_world;
        R_wb0.col(2) = z_world;

        qref = Eigen::Quaternion<T>(R_wb0);
        qref.normalize();

        v2ref = R_bw() * mag;
        last_acc_body_cached_ = acc_body;
        auto_zero_pseudo_elapsed_sec_ = T(0);
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    void initialize_from_acc(Vector3 const& acc_body) {
        const Vector3 acc = deheel_vector_(acc_body);
        const T anorm = acc.norm();
        if (!(anorm > T(1e-8))) {
            throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
        }
        qref = quaternion_from_acc(acc / anorm);
        qref.normalize();
        last_acc_body_cached_ = acc_body;
        auto_zero_pseudo_elapsed_sec_ = T(0);
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc) {
        Vector3 an = acc.normalized();
        Vector3 zb = Vector3::UnitZ();

        Vector3 target = -an;
        T cos_theta = zb.dot(target);
        Vector3 axis = zb.cross(target);
        T norm_axis = axis.norm();

        if (norm_axis < T(1e-8)) {
            if (cos_theta > 0) return Eigen::Quaternion<T>::Identity();
            return Eigen::Quaternion<T>(0, 1, 0, 0);
        }

        axis /= norm_axis;
        cos_theta = std::max(T(-1), std::min(T(1), cos_theta));
        const T angle = std::acos(cos_theta);

        Eigen::AngleAxis<T> aa(angle, axis);
        Eigen::Quaternion<T> q(aa);
        q.normalize();
        return q;
    }

    void set_mag_world_ref(const Vector3& B_world) {
        if (B_world.allFinite() && B_world.norm() > T(1e-9)) v2ref = B_world;
    }

    void time_update(Vector3 const& gyr_body,
                     Vector3 const& acc_body,
                     T Ts) {
        if (!(Ts > T(1e-5)) || !std::isfinite(Ts)) return;
        if (Ts > T(0.05)) Ts = T(0.05);
        if (!gyr_body.allFinite() || !acc_body.allFinite()) return;

        last_dt_ = Ts;
        last_acc_body_cached_ = acc_body;

        param_rw_predict_(Ts);

        const Vector3 gyr = deheel_vector_(gyr_body);

        Vector3 gyro_bias = Vector3::Zero();
        if constexpr (with_gyro_bias) gyro_bias = xext.template segment<3>(3);
        last_gyr_bias_corrected = gyr - gyro_bias;
        if (!last_gyr_bias_corrected.allFinite()) {
            last_gyr_bias_corrected.setZero();
            return;
        }

        const Vector3 omega_b = last_gyr_bias_corrected;
        if (have_prev_omega_ && Ts > T(0)) {
            const Vector3 alpha_raw = (omega_b - prev_omega_b_) / Ts;
            if (alpha_raw.allFinite()) {
                if (alpha_smooth_tau_ > T(0)) {
                    const T a = T(1) - std::exp(-Ts / alpha_smooth_tau_);
                    alpha_b_ = (T(1) - a) * alpha_b_ + a * alpha_raw;
                } else {
                    alpha_b_ = alpha_raw;
                }
            } else {
                alpha_b_.setZero();
            }
        } else {
            alpha_b_.setZero();
            have_prev_omega_ = true;
        }
        prev_omega_b_ = omega_b;
        if (!alpha_b_.allFinite()) alpha_b_.setZero();
        if (!prev_omega_b_.allFinite()) prev_omega_b_.setZero();

        Eigen::Quaternion<T> dq = quat_from_delta_theta((last_gyr_bias_corrected * Ts).eval());
        qref = qref * dq;
        qref.normalize();

        const Matrix3 I = Matrix3::Identity();
        const Vector3 w = last_gyr_bias_corrected;
        const T omega = w.norm();
        const T theta = omega * Ts;

        MatrixBaseN F_AA = MatrixBaseN::Identity();
        if (theta < T(1e-5)) {
            const Matrix3 Wx = skew_symmetric_matrix(w);
            F_AA.template topLeftCorner<3,3>() = I - Wx * Ts + (Wx * Wx) * (Ts * Ts / T(2));
        } else {
            const Matrix3 Wn = skew_symmetric_matrix(w / (omega + std::numeric_limits<T>::epsilon()));
            const T s = std::sin(theta), c = std::cos(theta);
            F_AA.template topLeftCorner<3,3>() = I - s * Wn + (T(1) - c) * (Wn * Wn);
        }

        if constexpr (with_gyro_bias) {
            Matrix3 Rstep, Bstep;
            rot_and_B_from_wt_(w, Ts, Rstep, Bstep);
            F_AA.template topLeftCorner<3,3>() = Rstep;
            F_AA.template block<3,3>(0,3) = Bstep;
        }

        MatrixBaseN Q_AA = MatrixBaseN::Zero();
        if (!use_exact_att_bias_Qd_) {
            Q_AA = Qbase * Ts;
        } else {
            const Matrix3 Qg = Qbase.template topLeftCorner<3,3>();
            Matrix3 Qbg = Matrix3::Zero();
            if constexpr (with_gyro_bias) Qbg = Qbase.template bottomRightCorner<3,3>();

            Matrix3 I_R;
            if (is_isotropic3_(Qg)) I_R = Matrix3::Identity() * (Qg(0,0) * Ts);
            else I_R = simpson_R_Q_RT_(w, Ts, Qg);

            Matrix3 I_BB = Matrix3::Zero();
            if constexpr (with_gyro_bias) I_BB = simpson_B_Q_BT_(w, Ts, Qbg);

            Q_AA.template topLeftCorner<3,3>() = I_R + I_BB;
            if constexpr (with_gyro_bias) {
                Matrix3 Qbb = Qbg * Ts;
                Matrix3 IB; integral_B_ds_(w, Ts, IB);
                Matrix3 Qtb = IB * Qbg;
                Q_AA.template topRightCorner<3,3>() = Qtb;
                Q_AA.template bottomLeftCorner<3,3>() = Qtb.transpose();
                Q_AA.template bottomRightCorner<3,3>() = Qbb;
            }
            Q_AA = T(0.5) * (Q_AA + Q_AA.transpose());
            if constexpr (with_gyro_bias) project_psd<T,6>(Q_AA, T(1e-12));
            else project_psd<T,3>(Q_AA, T(1e-12));
        }

        MatrixNX Fext = MatrixNX::Identity();
        MatrixNX Qext = MatrixNX::Zero();

        Fext.template block<BASE_N,BASE_N>(0,0) = F_AA;
        Qext.template block<BASE_N,BASE_N>(0,0) = Q_AA;

        if (linear_block_enabled_) {
            const Vector3 acc_cmd = deheel_vector_(acc_body);

            Vector3 lever = Vector3::Zero();
            Matrix3 Jlever_dbg = Matrix3::Zero();

            if (use_imu_lever_arm_) {
                const Vector3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
                lever.noalias() += alpha_b_.cross(r_imu_bprime)
                                +  omega_b.cross(omega_b.cross(r_imu_bprime));

                if constexpr (with_gyro_bias) {
                    T k_alpha = T(0);
                    if (have_prev_omega_ && Ts > T(0)) {
                        if (alpha_smooth_tau_ > T(0)) {
                            const T a = T(1) - std::exp(-Ts / alpha_smooth_tau_);
                            k_alpha = a / Ts;
                        } else {
                            k_alpha = T(1) / Ts;
                        }
                    }
                    const Matrix3 J_alpha_part = k_alpha * skew_symmetric_matrix(r_imu_bprime);
                    const Matrix3 J_omega_part = d_omega_x_omega_x_r_domega_(omega_b, r_imu_bprime);
                    Jlever_dbg = J_alpha_part - J_omega_part;
                }
            }

            if (!lever.allFinite()) {
                lever.setZero();
                Jlever_dbg.setZero();
            }

            const Matrix3 Rbw = R_bw();
            const Vector3 g_world(0,0,+gravity_magnitude_);

            Vector3 u_rot = Rbw * (acc_cmd - lever);
            Vector3 u_w   = u_rot + g_world;
            if (!u_rot.allFinite() || !u_w.allFinite()) return;

            // Clamp commanded inertial acceleration to avoid one-frame catastrophic injection.
            clamp_vec_(u_w, linear_accel_limit_);

            Vector3 v = xext.template segment<3>(OFF_V);
            Vector3 p = xext.template segment<3>(OFF_P);

            if constexpr (with_accel_bias) {
                Vector3 b_aw_prev = xext.template segment<3>(OFF_BAW);
                clamp_vec_(b_aw_prev, T(3.0));

                Matrix3 Fbb = Matrix3::Identity();
                for (int i = 0; i < 3; ++i) {
                    const T tau = std::max(T(1e-3), baw_leak_tau_sec_(i));
                    const T a = std::exp(-Ts / tau);
                    b_aw_prev(i) *= a;
                    Fbb(i,i) = a;
                }

                xext.template segment<3>(OFF_BAW) = b_aw_prev;

                Vector3 a_eff = u_w - b_aw_prev;
                clamp_vec_(a_eff, linear_accel_limit_);

                xext.template segment<3>(OFF_V) = v + Ts * a_eff;
                xext.template segment<3>(OFF_P) = p + Ts * v + (T(0.5) * Ts * Ts) * a_eff;

                Fext.template block<3,3>(OFF_BAW, OFF_BAW) = Fbb;
                Fext.template block<3,3>(OFF_V, OFF_BAW)   = -(Ts) * Fbb;
                Fext.template block<3,3>(OFF_P, OFF_BAW)   = -(T(0.5) * Ts * Ts) * Fbb;
            } else {
                Vector3 a_eff = u_w;
                clamp_vec_(a_eff, linear_accel_limit_);

                xext.template segment<3>(OFF_V) = v + Ts * a_eff;
                xext.template segment<3>(OFF_P) = p + Ts * v + (T(0.5) * Ts * Ts) * a_eff;
            }

            Vector3 vv = xext.template segment<3>(OFF_V);
            Vector3 pp = xext.template segment<3>(OFF_P);
            clamp_vec_(vv, velocity_limit_);
            clamp_vec_(pp, position_limit_);
            xext.template segment<3>(OFF_V) = vv;
            xext.template segment<3>(OFF_P) = pp;

            Fext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity();
            Fext.template block<3,3>(OFF_P, OFF_V) = Matrix3::Identity() * Ts;
            Fext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity();

            const Matrix3 J_u_att = skew_symmetric_matrix(u_rot);
            Fext.template block<3,3>(OFF_V, 0) = J_u_att * Ts;
            Fext.template block<3,3>(OFF_P, 0) = J_u_att * (T(0.5) * Ts * Ts);

            if constexpr (with_gyro_bias) {
                if (use_imu_lever_arm_) {
                    const Matrix3 J_u_bg = -Rbw * Jlever_dbg;
                    Fext.template block<3,3>(OFF_V, 3) = J_u_bg * Ts;
                    Fext.template block<3,3>(OFF_P, 3) = J_u_bg * (T(0.5) * Ts * Ts);
                }
            }

            Matrix3 Ru_world = Rbw * Racc * Rbw.transpose();
            Ru_world = T(0.5) * (Ru_world + Ru_world.transpose());
            project_psd<T,3>(Ru_world, T(1e-12));

            const T dt  = Ts;
            const T dt2 = dt * dt;
            const T dt3 = dt2 * dt;
            const T dt4 = dt2 * dt2;
            const T dt5 = dt4 * dt;

            Qext.template block<3,3>(OFF_V, OFF_V).noalias() += dt2 * Ru_world;
            Qext.template block<3,3>(OFF_V, OFF_P).noalias() += (T(0.5) * dt3) * Ru_world;
            Qext.template block<3,3>(OFF_P, OFF_V) = Qext.template block<3,3>(OFF_V, OFF_P).transpose();
            Qext.template block<3,3>(OFF_P, OFF_P).noalias() += (T(0.25) * dt4) * Ru_world;

            if constexpr (with_accel_bias) {
                for (int axis = 0; axis < 3; ++axis) {
                    const T q = acc_bias_updates_enabled_ ? Q_baw_rw_(axis, axis) : T(0);
                    const T q_limited = std::min(q, T(1.0));

                    Qext(OFF_V + axis, OFF_V + axis)     += q_limited * dt3 / T(3);
                    Qext(OFF_V + axis, OFF_P + axis)     += q_limited * dt4 / T(8);
                    Qext(OFF_P + axis, OFF_V + axis)      = Qext(OFF_V + axis, OFF_P + axis);
                    Qext(OFF_P + axis, OFF_P + axis)     += q_limited * dt5 / T(20);

                    Qext(OFF_V + axis, OFF_BAW + axis)   += -q_limited * dt2 / T(2);
                    Qext(OFF_BAW + axis, OFF_V + axis)    = Qext(OFF_V + axis, OFF_BAW + axis);

                    Qext(OFF_P + axis, OFF_BAW + axis)   += -q_limited * dt3 / T(6);
                    Qext(OFF_BAW + axis, OFF_P + axis)    = Qext(OFF_P + axis, OFF_BAW + axis);

                    Qext(OFF_BAW + axis, OFF_BAW + axis) += q_limited * dt;
                }
            }
        }

        Qext = T(0.5) * (Qext + Qext.transpose());
        MatrixNX Pnew = Fext * Pext * Fext.transpose() + Qext;
        Pext = Pnew;

        sanitize_and_clamp_state_();
        sanitize_covariance_();

        maybe_run_auto_zero_pseudos_(Ts);
    }

    void time_update(Vector3 const& gyr_body, T Ts) {
        time_update(gyr_body, last_acc_body_cached_, Ts);
    }

    void measurement_update_acc_only(Vector3 const& acc_meas_body) {
        last_acc_diag_ = MeasDiag3{};
        last_acc_diag_.accepted = false;
        last_acc_body_cached_ = acc_meas_body;

        const Vector3 acc_meas = deheel_vector_(acc_meas_body);
        if (!acc_meas.allFinite()) return;

        const T anorm = acc_meas.norm();
        if (!(anorm > T(1e-6))) return;

        // Direction-only accelerometer update.
        const Vector3 z_meas = acc_meas / anorm;

        const Matrix3 Rwb = R_wb();
        const Vector3 g_body = Rwb * Vector3(T(0), T(0), -gravity_magnitude_);
        const T gnorm = g_body.norm();
        if (!(gnorm > T(1e-6))) return;

        const Vector3 zhat = g_body / gnorm;
        const Vector3 innov = z_meas - zhat;
        if (!innov.allFinite()) return;

        last_acc_diag_.r = innov;

        const Matrix3 J_att = -skew_symmetric_matrix(zhat);

        const T sigma_acc = std::sqrt(std::max(T(1e-12), Racc.trace() / T(3)));
        const T sigma_dir_base =
            std::max(T(1e-3),
            std::min(T(0.25), sigma_acc / std::max(T(1e-3), gravity_magnitude_)));

        const T mag_dev = std::abs(anorm - gravity_magnitude_);
        const T dyn_mag = mag_dev / std::max(T(0.25), sigma_acc);

        T rate = last_gyr_bias_corrected.norm();
        if (!std::isfinite(rate)) rate = T(0);

        const T infl = T(1)
                     + T(0.35) * dyn_mag * dyn_mag
                     + T(0.10) * rate * rate;

        const T sigma_dir = std::min(T(0.5), sigma_dir_base * infl);
        const Matrix3 R_dir = Matrix3::Identity() * (sigma_dir * sigma_dir);

        Matrix3& S_mat = S_scratch_;
        S_mat = R_dir;

        {
            const Matrix3 P_th_th = Pext.template block<3,3>(0,0);
            S_mat.noalias() += J_att * P_th_th * J_att.transpose();
        }

        S_mat = T(0.5) * (S_mat + S_mat.transpose());

        MatrixNX3& PCt = PCt_scratch_;
        PCt.setZero();
        PCt.noalias() = Pext.template block<NX,3>(0,0) * J_att.transpose();

        // Freeze only v,p. Allow b_aw to move indirectly via cross-covariance.
        freeze_vp_rows_(PCt);
        if constexpr (with_accel_bias) {
            if (!acc_bias_updates_enabled_) freeze_baw_rows_(PCt);
        }

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, S_mat.norm())) {
            last_acc_diag_.S = S_mat;
            last_acc_diag_.nis = std::numeric_limits<T>::quiet_NaN();
            last_acc_diag_.accepted = false;
            return;
        }

        last_acc_diag_.S = S_mat;
        last_acc_diag_.nis = nis3_from_ldlt_(ldlt, innov);

        const T nis_gate = T(25);
        if (!std::isfinite(last_acc_diag_.nis) || last_acc_diag_.nis > nis_gate) {
            last_acc_diag_.accepted = false;
            return;
        }

        MatrixNX3& K = K_scratch_;
        K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

        freeze_vp_rows_(K);
        if constexpr (with_accel_bias) {
            if (!acc_bias_updates_enabled_) freeze_baw_rows_(K);
        }

        xext.noalias() += K * innov;
        joseph_update3_(K, S_mat, PCt);

        applyQuaternionCorrectionFromErrorState();
        last_acc_diag_.accepted = true;
    }

    void measurement_update_mag_only(const Vector3& mag_meas_body) {
        last_mag_diag_ = MeasDiag3{};
        last_mag_diag_.accepted = false;

        const Vector3 mag_meas = deheel_vector_(mag_meas_body);
        if (!mag_meas.allFinite()) return;
        const T mag_norm = mag_meas.norm();
        if (!(mag_norm > T(1e-6))) return;

        Vector3 v2hat = R_wb() * v2ref;
        if (v2hat.dot(mag_meas) < T(0)) v2hat = -v2hat;

        const Vector3 r = mag_meas - v2hat;
        last_mag_diag_.r = r;

        const Matrix3 J_att = -skew_symmetric_matrix(v2hat);

        Matrix3& S_mat = S_scratch_;
        S_mat = Rmag;
        const Matrix3 P_th_th = Pext.template block<3,3>(0,0);
        S_mat.noalias() += J_att * P_th_th * J_att.transpose();

        MatrixNX3& PCt = PCt_scratch_;
        PCt.setZero();
        PCt.noalias() += Pext.template block<NX,3>(0,0) * J_att.transpose();

        // Freeze only v,p. Allow b_aw indirect correction unless bias updates are disabled.
        freeze_vp_rows_(PCt);
        if constexpr (with_accel_bias) {
            if (!acc_bias_updates_enabled_) freeze_baw_rows_(PCt);
        }

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, Rmag.norm())) {
            last_mag_diag_.S = S_mat;
            last_mag_diag_.nis = std::numeric_limits<T>::quiet_NaN();
            last_mag_diag_.accepted = false;
            return;
        }

        last_mag_diag_.S = S_mat;
        last_mag_diag_.nis = nis3_from_ldlt_(ldlt, r);

        MatrixNX3& K = K_scratch_;
        K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

        freeze_vp_rows_(K);
        if constexpr (with_accel_bias) {
            if (!acc_bias_updates_enabled_) freeze_baw_rows_(K);
        }

        xext.noalias() += K * r;
        joseph_update3_(K, S_mat, PCt);
        applyQuaternionCorrectionFromErrorState();
        last_mag_diag_.accepted = true;
    }

    void measurement_update_position_pseudo(const Vector3& p_meas, const Vector3& sigma_meas) {
        if (!linear_block_enabled_) return;

        const Vector3 p_pred = xext.template segment<3>(OFF_P);
        const Vector3 r = p_meas - p_pred;
        if (!r.allFinite()) return;

        Matrix3& S_mat = S_scratch_;
        S_mat = Pext.template block<3,3>(OFF_P, OFF_P);

        Matrix3 R_meas = Matrix3::Zero();
        const T sx = std::max(T(1e-6), std::abs(sigma_meas.x()));
        const T sy = std::max(T(1e-6), std::abs(sigma_meas.y()));
        const T sz = std::max(T(1e-6), std::abs(sigma_meas.z()));
        R_meas(0,0) = sx * sx;
        R_meas(1,1) = sy * sy;
        R_meas(2,2) = sz * sz;
        S_mat.noalias() += R_meas;

        MatrixNX3& PCt = PCt_scratch_;
        PCt.noalias() = Pext.template block<NX,3>(0, OFF_P);

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, R_meas.norm())) return;

        MatrixNX3& K = K_scratch_;
        K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * r;
        joseph_update3_(K, S_mat, PCt);
        applyQuaternionCorrectionFromErrorState();
    }

    void measurement_update_velocity_pseudo(const Vector3& v_meas, const Vector3& sigma_meas) {
        if (!linear_block_enabled_) return;

        const Vector3 v_pred = xext.template segment<3>(OFF_V);
        const Vector3 r = v_meas - v_pred;
        if (!r.allFinite()) return;

        Matrix3& S_mat = S_scratch_;
        S_mat = Pext.template block<3,3>(OFF_V, OFF_V);

        Matrix3 R_meas = Matrix3::Zero();
        const T sx = std::max(T(1e-6), std::abs(sigma_meas.x()));
        const T sy = std::max(T(1e-6), std::abs(sigma_meas.y()));
        const T sz = std::max(T(1e-6), std::abs(sigma_meas.z()));
        R_meas(0,0) = sx * sx;
        R_meas(1,1) = sy * sy;
        R_meas(2,2) = sz * sz;
        S_mat.noalias() += R_meas;

        MatrixNX3& PCt = PCt_scratch_;
        PCt.noalias() = Pext.template block<NX,3>(0, OFF_V);

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, R_meas.norm())) return;

        MatrixNX3& K = K_scratch_;
        K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * r;
        joseph_update3_(K, S_mat, PCt);
        applyQuaternionCorrectionFromErrorState();
    }

    void measurement_update_vert_velocity_pseudo(T vz_meas, T sigma_meas) {
        if (!linear_block_enabled_) return;

        constexpr int idx_vz = OFF_V + 2;
        if (!std::isfinite(vz_meas)) return;

        const T r = vz_meas - xext(idx_vz);
        if (!std::isfinite(r)) return;

        const T sigma = std::max(T(1e-6), std::abs(sigma_meas));
        const T R = sigma * sigma;
        const T S = Pext(idx_vz, idx_vz) + R;
        if (!(S > T(0)) || !std::isfinite(S)) return;

        Eigen::Matrix<T, NX, 1> PCt;
        PCt.noalias() = Pext.col(idx_vz);

        Eigen::Matrix<T, NX, 1> K;
        K.noalias() = PCt / S;

        xext.noalias() += K * r;

        Pext.noalias() -= K * PCt.transpose();
        Pext.noalias() -= PCt * K.transpose();
        Pext.noalias() += K * S * K.transpose();

        sanitize_and_clamp_state_();
        sanitize_covariance_();
        applyQuaternionCorrectionFromErrorState();
    }

    [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    [[nodiscard]] MatrixBaseN covariance_base() const { return Pext.topLeftCorner(BASE_N, BASE_N); }
    [[nodiscard]] MatrixNX covariance_full() const { return Pext; }

    [[nodiscard]] Eigen::Quaternion<T> quaternion_boat() const {
        const Eigen::Quaternion<T> q_WBprime = quaternion();
        const T half = -wind_heel_rad_ * T(0.5);
        const T c = std::cos(half);
        const T s = std::sin(half);
        const Eigen::Quaternion<T> q_BprimeB(c, s, 0, 0);
        return q_WBprime * q_BprimeB;
    }

    [[nodiscard]] Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) return xext.template segment<3>(3);
        return Vector3::Zero();
    }

    [[nodiscard]] Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) return xext.template segment<3>(OFF_BAW);
        return Vector3::Zero();
    }

    [[nodiscard]] Vector3 get_world_accel_bias() const { return get_acc_bias(); }
    [[nodiscard]] Vector3 get_velocity() const { return xext.template segment<3>(OFF_V); }
    [[nodiscard]] Vector3 get_position() const { return xext.template segment<3>(OFF_P); }

    void set_linear_block_enabled(bool on) {
        if (linear_block_enabled_ && !on) {
            zero_AL_cross_cov_once_();
            auto_zero_pseudo_elapsed_sec_ = T(0);
        }
        linear_block_enabled_ = on;
        if (!linear_block_enabled_) auto_zero_pseudo_elapsed_sec_ = T(0);
    }

    bool linear_block_enabled() const { return linear_block_enabled_; }

    void set_warmup_mode(bool on) {
        set_linear_block_enabled(!on);
        set_acc_bias_updates_enabled(!on);
        if (on) clear_imu_lever_arm();
    }

    void set_acc_bias_updates_enabled(bool en) {
        if (acc_bias_updates_enabled_ == en) return;
        if constexpr (with_accel_bias) {
            if (!en) {
                Pext.template block<3,BASE_N>(OFF_BAW, 0).setZero();
                Pext.template block<BASE_N,3>(0, OFF_BAW).setZero();

                if (linear_block_enabled_) {
                    Pext.template block<3,6>(OFF_BAW, OFF_V).setZero();
                    Pext.template block<6,3>(OFF_V, OFF_BAW).setZero();
                }
            } else {
                auto Pba = Pext.template block<3,3>(OFF_BAW, OFF_BAW);
                for (int i = 0; i < 3; ++i) {
                    Pba(i,i) = std::max(Pba(i,i), initial_baw_std_(i) * initial_baw_std_(i));
                }
                Pext.template block<3,3>(OFF_BAW, OFF_BAW) = Pba;
            }
            sanitize_covariance_();
        }
        acc_bias_updates_enabled_ = en;
    }

    void set_Rp0_noise_std(const Vector3& sigma_p0) {
        if (param_rw_enabled_) { param_rw_update_sigma_p0_cmd_(sigma_p0); return; }
        R_p0 = sigma_p0.cwiseMax(T(1e-6)).array().square().matrix().asDiagonal();
        R_p0 = T(0.5) * (R_p0 + R_p0.transpose());
    }

    void set_Rp0_noise_matrix(const Matrix3& R) {
        Matrix3 S = T(0.5) * (R + R.transpose());
        project_psd<T,3>(S, T(1e-12));
        if (param_rw_enabled_) {
            Vector3 sig;
            sig.x() = std::sqrt(std::max(T(0), S(0,0)));
            sig.y() = std::sqrt(std::max(T(0), S(1,1)));
            sig.z() = std::sqrt(std::max(T(0), S(2,2)));
            param_rw_update_sigma_p0_cmd_(sig);
            return;
        }
        R_p0 = S;
    }

    void set_Rv0_noise_std(const Vector3& sigma_v0) {
        if (param_rw_enabled_) { param_rw_update_sigma_v0_cmd_(sigma_v0); return; }
        R_v0 = sigma_v0.cwiseMax(T(1e-6)).array().square().matrix().asDiagonal();
        R_v0 = T(0.5) * (R_v0 + R_v0.transpose());
    }

    void set_Rv0_noise_matrix(const Matrix3& R) {
        Matrix3 S = T(0.5) * (R + R.transpose());
        project_psd<T,3>(S, T(1e-12));
        if (param_rw_enabled_) {
            Vector3 sig;
            sig.x() = std::sqrt(std::max(T(0), S(0,0)));
            sig.y() = std::sqrt(std::max(T(0), S(1,1)));
            sig.z() = std::sqrt(std::max(T(0), S(2,2)));
            param_rw_update_sigma_v0_cmd_(sig);
            return;
        }
        R_v0 = S;
    }

    void set_Racc_std(const Vector3& sigma_acc) {
        if (param_rw_enabled_) { param_rw_update_sigma_acc_cmd_(sigma_acc); return; }
        Racc = sigma_acc.cwiseMax(T(1e-6)).array().square().matrix().asDiagonal();
        Racc = T(0.5) * (Racc + Racc.transpose());
        refresh_baw_rw_from_sigma_acc_();
    }

    void set_Rmag_std(const Vector3& sigma_mag) {
        Rmag = sigma_mag.cwiseMax(T(1e-6)).array().square().matrix().asDiagonal();
        Rmag = T(0.5) * (Rmag + Rmag.transpose());
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0) {
        sigma_v0 = std::max(T(1e-6), std::abs(sigma_v0));
        sigma_p0 = std::max(T(1e-6), std::abs(sigma_p0));
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0 * sigma_v0);
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0 * sigma_p0);
        sanitize_covariance_();
    }

    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) {
            const T ss = std::max(T(0), s);
            initial_baw_std_ = Vector3::Constant(ss);
            Pext.template block<3,3>(OFF_BAW, OFF_BAW) = Matrix3::Identity() * ss * ss;
            sanitize_covariance_();
        }
    }

    void set_initial_acc_bias(const Vector3& b0_world) {
        if constexpr (with_accel_bias) {
            xext.template segment<3>(OFF_BAW) = b0_world;
            sanitize_and_clamp_state_();
        }
    }

    void set_world_accel_bias_rw_gain(const Vector3& gain) {
        baw_rw_gain_ = gain.cwiseMax(T(0));
        refresh_baw_rw_from_sigma_acc_();
    }

    void set_world_accel_bias_rw_floor(const Vector3& floor_std_per_sqrt_s) {
        baw_rw_floor_ = floor_std_per_sqrt_s.cwiseMax(T(0));
        refresh_baw_rw_from_sigma_acc_();
    }

    void set_world_accel_bias_rw_std(const Vector3& std_per_sqrt_s) {
        Vector3 s = std_per_sqrt_s;
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(s(i)) || s(i) < T(0)) s(i) = T(0);
        }
        Q_baw_rw_ = s.array().square().matrix().asDiagonal();
        Q_baw_rw_ = T(0.5) * (Q_baw_rw_ + Q_baw_rw_.transpose());
    }

    [[nodiscard]] Vector3 get_world_accel_bias_rw_std() const {
        Vector3 s;
        s.x() = std::sqrt(std::max(T(0), Q_baw_rw_(0,0)));
        s.y() = std::sqrt(std::max(T(0), Q_baw_rw_(1,1)));
        s.z() = std::sqrt(std::max(T(0), Q_baw_rw_(2,2)));
        return s;
    }

    void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }

    void initialize_from_truth(const Vector3 &p_ned,
                               const Vector3 &v_ned,
                               const Eigen::Quaternion<T> &q_bw,
                               const Vector3 &b_aw_ned) {
        xext.setZero();
        xext.template segment<3>(OFF_V) = v_ned;
        xext.template segment<3>(OFF_P) = p_ned;
        if constexpr (with_accel_bias) xext.template segment<3>(OFF_BAW) = b_aw_ned;
        if constexpr (with_gyro_bias) xext.template segment<3>(3).setZero();

        qref = q_bw.conjugate();
        qref.normalize();

        Pext.setZero();
        const T p_0 = T(1e-5);
        Pext.diagonal().array() = p_0;

        auto_zero_pseudo_elapsed_sec_ = T(0);
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    void set_param_rw_enabled(bool on) {
        param_rw_enabled_ = on;
        if (param_rw_enabled_) refresh_model_params_from_filtered_();
        else refresh_baw_rw_from_sigma_acc_();
    }

    void set_param_rw_process_std_log(Vector3 sigma_acc_rw_log,
                                      Vector3 sigma_p0_rw_log,
                                      Vector3 sigma_v0_rw_log) {
        log_sigma_acc_f_.q = sigma_acc_rw_log.array().square().matrix();
        log_sigma_p0_f_.q  = sigma_p0_rw_log.array().square().matrix();
        log_sigma_v0_f_.q  = sigma_v0_rw_log.array().square().matrix();
    }

    void set_param_cmd_std_log(Vector3 sigma_acc_cmd_log,
                               Vector3 sigma_p0_cmd_log,
                               Vector3 sigma_v0_cmd_log) {
        log_sigma_acc_f_.r = sigma_acc_cmd_log.array().square().matrix();
        log_sigma_p0_f_.r  = sigma_p0_cmd_log.array().square().matrix();
        log_sigma_v0_f_.r  = sigma_v0_cmd_log.array().square().matrix();
    }

    void command_sigma_acc(const Vector3& sigma_acc_cmd) { param_rw_update_sigma_acc_cmd_(sigma_acc_cmd); }
    void command_sigma_p0 (const Vector3& sigma_p0_cmd)  { param_rw_update_sigma_p0_cmd_(sigma_p0_cmd); }
    void command_sigma_v0 (const Vector3& sigma_v0_cmd)  { param_rw_update_sigma_v0_cmd_(sigma_v0_cmd); }

    [[nodiscard]] Vector3 get_Racc_std() const {
        Vector3 s;
        s.x() = std::sqrt(std::max(T(0), Racc(0,0)));
        s.y() = std::sqrt(std::max(T(0), Racc(1,1)));
        s.z() = std::sqrt(std::max(T(0), Racc(2,2)));
        return s;
    }

    [[nodiscard]] Vector3 get_Rp0_noise_std() const {
        Vector3 s;
        s.x() = std::sqrt(std::max(T(0), R_p0(0,0)));
        s.y() = std::sqrt(std::max(T(0), R_p0(1,1)));
        s.z() = std::sqrt(std::max(T(0), R_p0(2,2)));
        return s;
    }

    [[nodiscard]] Vector3 get_Rv0_noise_std() const {
        Vector3 s;
        s.x() = std::sqrt(std::max(T(0), R_v0(0,0)));
        s.y() = std::sqrt(std::max(T(0), R_v0(1,1)));
        s.z() = std::sqrt(std::max(T(0), R_v0(2,2)));
        return s;
    }

    void apply_adaptive_params(const Vector3& sigma_acc_cmd) {
        command_sigma_acc(sigma_acc_cmd);
    }

    void set_imu_lever_arm_body(const Vector3& r_b) {
        r_imu_wrt_cog_body_phys_ = r_b;
        use_imu_lever_arm_ = (r_b.squaredNorm() > T(0));
    }

    void clear_imu_lever_arm() {
        r_imu_wrt_cog_body_phys_.setZero();
        use_imu_lever_arm_ = false;
    }

    void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

    void update_wind_heel(T heel_rad) {
        const T old = wind_heel_rad_;
        if (heel_rad == old) return;
        retarget_bodyprime_frame_(heel_rad - old);
        wind_heel_rad_ = heel_rad;
        update_unheel_trig_();
    }

    static Eigen::Matrix<T,3,1> ned_field_from_decl_incl(T D_rad, T I_rad, T B = T(1)) {
        const T cI = std::cos(I_rad), sI = std::sin(I_rad);
        const T cD = std::cos(D_rad), sD = std::sin(D_rad);
        return (Eigen::Matrix<T,3,1>() <<
            B * cI * cD,
            B * cI * sD,
            B * sI
        ).finished();
    }

    void set_auto_zero_pseudo_updates(bool position_on, bool velocity_on, T period_sec) {
        auto_zero_position_pseudo_enabled_ = position_on;
        auto_zero_velocity_pseudo_enabled_ = velocity_on;
        auto_zero_pseudo_period_sec_ = std::max(T(0), period_sec);
        auto_zero_pseudo_elapsed_sec_ = T(0);
    }

    void set_auto_zero_position_pseudo_enabled(bool on) {
        auto_zero_position_pseudo_enabled_ = on;
        auto_zero_pseudo_elapsed_sec_ = T(0);
    }

    void set_auto_zero_velocity_pseudo_enabled(bool on) {
        auto_zero_velocity_pseudo_enabled_ = on;
        auto_zero_pseudo_elapsed_sec_ = T(0);
    }

    void set_auto_zero_pseudo_period(T period_sec) {
        auto_zero_pseudo_period_sec_ = std::max(T(0), period_sec);
        auto_zero_pseudo_elapsed_sec_ = T(0);
    }

    [[nodiscard]] bool auto_zero_position_pseudo_enabled() const { return auto_zero_position_pseudo_enabled_; }
    [[nodiscard]] bool auto_zero_velocity_pseudo_enabled() const { return auto_zero_velocity_pseudo_enabled_; }
    [[nodiscard]] T auto_zero_pseudo_period() const { return auto_zero_pseudo_period_sec_; }

  private:
    const T gravity_magnitude_ = T(STD_GRAVITY);

    Eigen::Quaternion<T> qref;
    Vector3 v2ref = Vector3::UnitX();

    Matrix<T, NX, 1> xext;
    MatrixNX Pext;

    Vector3 last_gyr_bias_corrected = Vector3::Zero();
    Vector3 last_acc_body_cached_   = Vector3::Zero();

    Matrix3 Q_baw_rw_ = Matrix3::Identity() * T(1e-4);
    Vector3 initial_baw_std_ = Vector3::Constant(T(0.02));

    Matrix3 Rmag;
    MatrixBaseN Qbase;

    Matrix3 Racc;
    Matrix3 R_p0;
    Matrix3 R_v0;

    bool auto_zero_position_pseudo_enabled_ = true;
    bool auto_zero_velocity_pseudo_enabled_ = true;
    T    auto_zero_pseudo_period_sec_       = T(0.015);
    T    auto_zero_pseudo_elapsed_sec_      = T(0);

    bool linear_block_enabled_ = true;
    bool acc_bias_updates_enabled_ = true;
    bool use_exact_att_bias_Qd_ = true;

    bool   use_imu_lever_arm_ = false;
    Vector3 r_imu_wrt_cog_body_phys_ = Vector3::Zero();

    Vector3 prev_omega_b_ = Vector3::Zero();
    Vector3 alpha_b_      = Vector3::Zero();
    bool    have_prev_omega_ = false;

    T last_dt_ = T(1.0/200);
    T alpha_smooth_tau_ = T(0.05);

    Vector3 baw_rw_gain_  = Vector3::Constant(T(0.25));
    Vector3 baw_rw_floor_ = Vector3::Constant(T(1e-4));

    // Weak leak on the world accel bias state. Z is shorter by default.
    Vector3 baw_leak_tau_sec_ = (Vector3() << T(12.0), T(12.0), T(4.0)).finished();

    // Hard safety clamps for catastrophic one-frame spikes.
    T linear_accel_limit_ = T(30.0);
    T velocity_limit_     = T(50.0);
    T position_limit_     = T(500.0);

    MatrixNX3    PCt_scratch_;
    MatrixNX3    K_scratch_;
    Matrix3      S_scratch_;

    MeasDiag3 last_acc_diag_;
    MeasDiag3 last_mag_diag_;

    EIGEN_STRONG_INLINE void symmetrize_Pext_() {
        for (int i = 0; i < NX; ++i) {
            for (int j = i + 1; j < NX; ++j) {
                const T v = T(0.5) * (Pext(i,j) + Pext(j,i));
                Pext(i,j) = v;
                Pext(j,i) = v;
            }
        }
    }

    EIGEN_STRONG_INLINE Matrix3 R_wb() const { return qref.toRotationMatrix(); }
    EIGEN_STRONG_INLINE Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }

    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
        Matrix3 M;
        M << T(0),    -vec(2),  vec(1),
             vec(2),  T(0),    -vec(0),
            -vec(1),  vec(0),   T(0);
        return M;
    }

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0) {
        MatrixBaseN Q; Q.setZero();
        if constexpr (with_gyro_bias) {
            Q.template topLeftCorner<3,3>() = sigma_g.array().square().matrix().asDiagonal();
            Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * std::max(T(0), b0);
        } else {
            Q = sigma_g.array().square().matrix().asDiagonal();
        }
        return Q;
    }

    void applyQuaternionCorrectionFromErrorState() {
        Vector3 dtheta = xext.template segment<3>(0);
        if (!dtheta.allFinite()) {
            xext.template head<3>().setZero();
            sanitize_and_clamp_state_();
            sanitize_covariance_();
            return;
        }

        const T max_corr = T(0.35);
        const T n = dtheta.norm();
        if (n > max_corr) dtheta *= (max_corr / n);

        const Matrix3 Gtheta = Matrix3::Identity() - T(0.5) * skew_symmetric_matrix(dtheta);

        Eigen::Quaternion<T> corr = quat_from_delta_theta(dtheta);
        qref = qref * corr;
        qref.normalize();

        {
            Eigen::Matrix<T,3,3> Ptt = Pext.template block<3,3>(0,0);
            Pext.template block<3,3>(0,0) = Gtheta * Ptt * Gtheta.transpose();
        }
        if constexpr (NX > 3) {
            Eigen::Matrix<T,3,NX-3> Ptx = Pext.template block<3,NX-3>(0,3);
            Pext.template block<3,NX-3>(0,3) = Gtheta * Ptx;
            Pext.template block<NX-3,3>(3,0) = Pext.template block<3,NX-3>(0,3).transpose();
        }

        xext.template head<3>().setZero();
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    EIGEN_STRONG_INLINE bool safe_ldlt3_(Matrix3& S, Eigen::LDLT<Matrix3>& ldlt, T noise_scale) const {
        S = T(0.5) * (S + S.transpose());

        const T ns = std::isfinite(noise_scale) ? std::abs(noise_scale) : T(0);
        T bump = std::max(T(1e-9), T(1e-6) * (ns + T(1)));

        for (int pass = 0; pass < 3; ++pass) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (!std::isfinite(S(i,j))) S(i,j) = (i == j) ? bump : T(0);
                }
                if (!(S(i,i) > T(0)) || !std::isfinite(S(i,i))) S(i,i) = bump;
            }

            S = T(0.5) * (S + S.transpose());
            ldlt.compute(S);
            if (ldlt.info() == Eigen::Success) return true;

            S.diagonal().array() += bump;
            bump *= T(10);
        }

        return false;
    }

    EIGEN_STRONG_INLINE T nis3_from_ldlt_(const Eigen::LDLT<Matrix3>& ldlt,
                                          const Vector3& r) const {
        Vector3 x = ldlt.solve(r);
        if (!x.allFinite()) return std::numeric_limits<T>::quiet_NaN();
        const T v = r.dot(x);
        return std::isfinite(v) ? v : std::numeric_limits<T>::quiet_NaN();
    }

    EIGEN_STRONG_INLINE void sanitize_and_clamp_state_() {
        if (!qref.coeffs().allFinite()) qref = Eigen::Quaternion<T>::Identity();
        else qref.normalize();

        for (int i = 0; i < NX; ++i) {
            if (!std::isfinite(xext(i))) xext(i) = T(0);
        }

        if constexpr (with_gyro_bias) {
            Vector3 bg = xext.template segment<3>(3);
            clamp_vec_(bg, T(0.5));
            xext.template segment<3>(3) = bg;
        }

        if constexpr (with_accel_bias) {
            Vector3 ba = xext.template segment<3>(OFF_BAW);
            clamp_vec_(ba, T(3.0));
            xext.template segment<3>(OFF_BAW) = ba;
        }
    }

    EIGEN_STRONG_INLINE void sanitize_covariance_() {
        const T diag_floor = T(1e-12);
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NX; ++j) {
                if (!std::isfinite(Pext(i,j))) Pext(i,j) = (i == j) ? diag_floor : T(0);
            }
        }
        symmetrize_Pext_();
        for (int i = 0; i < NX; ++i) {
            if (!(Pext(i,i) > diag_floor)) Pext(i,i) = diag_floor;
        }
        symmetrize_Pext_();
    }

    EIGEN_STRONG_INLINE void joseph_update3_(const Eigen::Matrix<T,NX,3>& K,
                                             const Matrix3& S,
                                             const Eigen::Matrix<T,NX,3>& PCt) {
        for (int i = 0; i < NX; ++i) {
            for (int j = i; j < NX; ++j) {
                T KCP_ij = T(0);
                T KCP_ji = T(0);
                for (int l = 0; l < 3; ++l) {
                    const T Ki_l = K(i,l);
                    const T Kj_l = K(j,l);
                    const T Pj_l = PCt(j,l);
                    const T Pi_l = PCt(i,l);

                    KCP_ij += Ki_l * Pj_l;
                    if (j != i) KCP_ji += Kj_l * Pi_l;
                }
                if (j == i) KCP_ji = KCP_ij;

                T KSK_ij = T(0);
                for (int a = 0; a < 3; ++a) {
                    const T Kia = K(i,a);
                    for (int b = 0; b < 3; ++b) {
                        const T Kjb = K(j,b);
                        KSK_ij += Kia * S(a,b) * Kjb;
                    }
                }
                const T delta = -(KCP_ij + KCP_ji) + KSK_ij;
                Pext(i,j) += delta;
                if (j != i) Pext(j,i) = Pext(i,j);
            }
        }
        sanitize_covariance_();
    }

    void zero_AL_cross_cov_once_() {
        constexpr int NA = BASE_N;
        constexpr int NL = LIN_N;
        Pext.template block<NA,NL>(0, OFF_V).setZero();
        Pext.template block<NL,NA>(OFF_V, 0).setZero();
    }

    EIGEN_STRONG_INLINE void freeze_linear_rows_(MatrixNX3& M) const {
        M.template block<LIN_N,3>(OFF_V, 0).setZero();
    }

    // Freeze only v,p. Leave b_aw unfrozen.
    EIGEN_STRONG_INLINE void freeze_vp_rows_(MatrixNX3& M) const {
        M.template block<6,3>(OFF_V, 0).setZero();
    }

    EIGEN_STRONG_INLINE void freeze_baw_rows_(MatrixNX3& M) const {
        if constexpr (with_accel_bias) {
            M.template block<3,3>(OFF_BAW, 0).setZero();
        }
    }

    EIGEN_STRONG_INLINE void freeze_baw_rows_(Eigen::Matrix<T, NX, 1>& v) const {
        if constexpr (with_accel_bias) {
            v.template segment<3>(OFF_BAW).setZero();
        }
    }

    EIGEN_STRONG_INLINE void freeze_base_rows_(MatrixNX3& M) const {
        M.template block<BASE_N,3>(0, 0).setZero();
    }

    EIGEN_STRONG_INLINE void freeze_base_rows_(Eigen::Matrix<T, NX, 1>& v) const {
        v.template segment<BASE_N>(0).setZero();
    }

    template<typename Vec3Like>
    static inline void clamp_vec_(Vec3Like& v, T lim) {
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(v(i))) v(i) = T(0);
            v(i) = std::max(-lim, std::min(lim, v(i)));
        }
    }

    T wind_heel_rad_ = T(0);
    T cos_unheel_x_  = T(1);
    T sin_unheel_x_  = T(0);

    EIGEN_STRONG_INLINE void update_unheel_trig_() {
        if (std::abs(wind_heel_rad_) < T(1e-9)) {
            cos_unheel_x_ = T(1);
            sin_unheel_x_ = T(0);
        } else {
            const T angle = -wind_heel_rad_;
            cos_unheel_x_ = std::cos(angle);
            sin_unheel_x_ = std::sin(angle);
        }
    }

    EIGEN_STRONG_INLINE Vector3 deheel_vector_(const Vector3& v_body) const {
        if (std::abs(wind_heel_rad_) < T(1e-9)) return v_body;
        Vector3 v;
        v.x() = v_body.x();
        v.y() = cos_unheel_x_ * v_body.y() - sin_unheel_x_ * v_body.z();
        v.z() = sin_unheel_x_ * v_body.y() + cos_unheel_x_ * v_body.z();
        return v;
    }

    EIGEN_STRONG_INLINE Matrix3 Rx_(T angle_rad) const {
        const T c = std::cos(angle_rad);
        const T s = std::sin(angle_rad);
        Matrix3 R;
        R << T(1), T(0), T(0),
             T(0), c,   -s,
             T(0), s,    c;
        return R;
    }

    void retarget_bodyprime_frame_(T delta_heel_rad) {
        if (std::abs(delta_heel_rad) < T(1e-12)) return;

        const Matrix3 R = Rx_(-delta_heel_rad);
        const Eigen::Quaternion<T> qR(R);
        qref = qR * qref;
        qref.normalize();

        xext.template segment<3>(0) = R * xext.template segment<3>(0);
        if constexpr (with_gyro_bias) {
            xext.template segment<3>(3) = R * xext.template segment<3>(3);
        }

        last_gyr_bias_corrected = R * last_gyr_bias_corrected;
        prev_omega_b_           = R * prev_omega_b_;
        alpha_b_                = R * alpha_b_;

        MatrixNX Tm = MatrixNX::Identity();
        Tm.template block<3,3>(0,0) = R;
        if constexpr (with_gyro_bias) Tm.template block<3,3>(3,3) = R;

        Pext = Tm * Pext * Tm.transpose();
        sanitize_and_clamp_state_();
        sanitize_covariance_();
    }

    EIGEN_STRONG_INLINE void rot_and_B_from_wt_(const Vector3& w, T t, Matrix3& R, Matrix3& B) const {
        const T wnorm = w.norm();
        const Matrix3 W = skew_symmetric_matrix(w);

        if (wnorm < T(1e-7)) {
            const T t2 = t*t, t3 = t2*t;
            R = Matrix3::Identity() - W * t + T(0.5) * (W*W) * t2;
            B = -( Matrix3::Identity()*t - T(0.5)*W*t2 + (W*W)*(t3/T(6)) );
            return;
        }

        const T theta = wnorm * t;
        const T s = std::sin(theta), c = std::cos(theta);
        const T invw = T(1) / wnorm;
        const Matrix3 K = W * invw;

        R = Matrix3::Identity() - s*K + (T(1)-c)*(K*K);

        const T invw2 = invw * invw;
        const Matrix3 term1 = Matrix3::Identity() * t;
        const Matrix3 term2 = ((T(1)-c) * invw2) * W;
        const Matrix3 term3 = ((t - s*invw) * invw2) * (W*W);
        B = -( term1 - term2 + term3 );
    }

    EIGEN_STRONG_INLINE void integral_B_ds_(const Vector3& w, T Tstep, Matrix3& IB) const {
        const T wnorm = w.norm();
        const Matrix3 W = skew_symmetric_matrix(w);

        if (wnorm < T(1e-7)) {
            const T T2 = Tstep*Tstep, T3 = T2*Tstep, T4 = T3*Tstep;
            IB = -( Matrix3::Identity()*(T(0.5)*T2)
                  - W*(T(1.0/6.0)*T3)
                  + (W*W)*(T(1.0/24.0)*T4) );
            return;
        }

        const T theta = wnorm * Tstep;
        const T s = std::sin(theta), c = std::cos(theta);
        const T invw  = T(1) / wnorm;
        const T invw2 = invw * invw;

        const Matrix3 termI = Matrix3::Identity() * (T(0.5) * Tstep*Tstep);
        const Matrix3 termW = ((Tstep - s*invw) * invw2) * W;
        const Matrix3 termW2 = ( (T(0.5)*Tstep*Tstep) + ((c - T(1)) * invw2) ) * invw2 * (W*W);

        IB = -( termI - termW + termW2 );
    }

    EIGEN_STRONG_INLINE Matrix3 d_omega_x_omega_x_r_domega_(const Vector3& w, const Vector3& r) const {
        const T s = w.dot(r);
        return Matrix3::Identity() * s + (w * r.transpose()) - T(2) * (r * w.transpose());
    }

    EIGEN_STRONG_INLINE Matrix3 simpson_R_Q_RT_(const Vector3& w, T Tstep, const Matrix3& Q) const {
        Matrix3 R0, Btmp, Rm, R1;
        rot_and_B_from_wt_(w, T(0), R0, Btmp);
        rot_and_B_from_wt_(w, T(0.5)*Tstep, Rm, Btmp);
        rot_and_B_from_wt_(w, Tstep, R1, Btmp);

        const Matrix3 f0 = R0 * Q * R0.transpose();
        const Matrix3 f1 = Rm * Q * Rm.transpose();
        const Matrix3 f2 = R1 * Q * R1.transpose();
        return (Tstep / T(6)) * (f0 + T(4)*f1 + f2);
    }

    EIGEN_STRONG_INLINE Matrix3 simpson_B_Q_BT_(const Vector3& w, T Tstep, const Matrix3& Q) const {
        Matrix3 Rtmp, B0, Bm, B1;
        rot_and_B_from_wt_(w, T(0), Rtmp, B0);
        rot_and_B_from_wt_(w, T(0.5)*Tstep, Rtmp, Bm);
        rot_and_B_from_wt_(w, Tstep, Rtmp, B1);

        const Matrix3 g0 = B0 * Q * B0.transpose();
        const Matrix3 g1 = Bm * Q * Bm.transpose();
        const Matrix3 g2 = B1 * Q * B1.transpose();
        return (Tstep / T(6)) * (g0 + T(4)*g1 + g2);
    }

    EIGEN_STRONG_INLINE bool is_isotropic3_(const Matrix3& S, T tol = T(1e-9)) const {
        const T a = S(0,0), b = S(1,1), c = S(2,2);
        Matrix3 Off = S;
        Off.diagonal().setZero();
        const T off = Off.cwiseAbs().sum();
        const T mean = (a + b + c) / T(3);
        return (std::abs(a-mean) + std::abs(b-mean) + std::abs(c-mean) + off)
               <= tol * (T(1) + std::abs(mean));
    }

    struct RWVec3Diag {
        Vector3 x = Vector3::Zero();
        Vector3 P = Vector3::Constant(T(1e-6));
        Vector3 q = Vector3::Zero();
        Vector3 r = Vector3::Zero();

        void predict(T dt) {
            if (dt > T(0)) P.array() += q.array() * dt;
            for (int i = 0; i < 3; ++i) {
                if (!(P(i) > T(0)) || !std::isfinite(P(i))) P(i) = T(1e-6);
            }
        }

        void update(const Vector3& z) {
            for (int i = 0; i < 3; ++i) {
                const T S = P(i) + r(i);
                if (!(S > T(0)) || !std::isfinite(S)) continue;
                const T K = P(i) / S;
                x(i) = x(i) + K * (z(i) - x(i));
                P(i) = (T(1) - K) * P(i);
                if (!(P(i) > T(0)) || !std::isfinite(P(i))) P(i) = T(1e-9);
            }
        }
    };

    static inline T clamp_pos_(T v, T lo, T hi) {
        if (!std::isfinite(v)) return lo;
        return std::max(lo, std::min(v, hi));
    }

    static inline Vector3 clamp_pos_vec_(const Vector3& v, T lo, T hi) {
        Vector3 out = v;
        out.x() = clamp_pos_(out.x(), lo, hi);
        out.y() = clamp_pos_(out.y(), lo, hi);
        out.z() = clamp_pos_(out.z(), lo, hi);
        return out;
    }

    bool param_rw_enabled_ = false;
    RWVec3Diag log_sigma_acc_f_;
    RWVec3Diag log_sigma_p0_f_;
    RWVec3Diag log_sigma_v0_f_;

    T sigma_acc_min_ = T(1e-4), sigma_acc_max_ = T(50);
    T sigma_p0_min_  = T(1e-6), sigma_p0_max_  = T(1e6);
    T sigma_v0_min_  = T(1e-6), sigma_v0_max_  = T(1e6);

    EIGEN_STRONG_INLINE void refresh_model_params_from_filtered_() {
        Vector3 sigma_acc = log_sigma_acc_f_.x.array().exp().matrix();
        sigma_acc = clamp_pos_vec_(sigma_acc, sigma_acc_min_, sigma_acc_max_);
        Racc = sigma_acc.array().square().matrix().asDiagonal();
        Racc = T(0.5) * (Racc + Racc.transpose());

        Vector3 sigma_p0 = log_sigma_p0_f_.x.array().exp().matrix();
        sigma_p0 = clamp_pos_vec_(sigma_p0, sigma_p0_min_, sigma_p0_max_);
        R_p0 = sigma_p0.array().square().matrix().asDiagonal();
        R_p0 = T(0.5) * (R_p0 + R_p0.transpose());

        Vector3 sigma_v0 = log_sigma_v0_f_.x.array().exp().matrix();
        sigma_v0 = clamp_pos_vec_(sigma_v0, sigma_v0_min_, sigma_v0_max_);
        R_v0 = sigma_v0.array().square().matrix().asDiagonal();
        R_v0 = T(0.5) * (R_v0 + R_v0.transpose());

        refresh_baw_rw_from_sigma_acc_();
    }

    EIGEN_STRONG_INLINE void refresh_baw_rw_from_sigma_acc_() {
        const Vector3 sigma_acc = get_Racc_std();
        const Vector3 sigma_rw =
            (baw_rw_gain_.cwiseProduct(sigma_acc) + baw_rw_floor_).cwiseMax(T(1e-9));
        Q_baw_rw_ = sigma_rw.array().square().matrix().asDiagonal();
        Q_baw_rw_ = T(0.5) * (Q_baw_rw_ + Q_baw_rw_.transpose());
    }

    EIGEN_STRONG_INLINE void param_rw_predict_(T dt) {
        if (!param_rw_enabled_) return;
        log_sigma_acc_f_.predict(dt);
        log_sigma_p0_f_.predict(dt);
        log_sigma_v0_f_.predict(dt);
        refresh_model_params_from_filtered_();
    }

    EIGEN_STRONG_INLINE void param_rw_update_sigma_acc_cmd_(const Vector3& sigma_acc_cmd) {
        if (!param_rw_enabled_) {
            set_Racc_std(sigma_acc_cmd);
            return;
        }
        Vector3 s = clamp_pos_vec_(sigma_acc_cmd, sigma_acc_min_, sigma_acc_max_);
        log_sigma_acc_f_.update(s.array().log().matrix());
        refresh_model_params_from_filtered_();
    }

    EIGEN_STRONG_INLINE void param_rw_update_sigma_p0_cmd_(const Vector3& sigma_p0_cmd) {
        if (!param_rw_enabled_) {
            set_Rp0_noise_std(sigma_p0_cmd);
            return;
        }
        Vector3 s = clamp_pos_vec_(sigma_p0_cmd, sigma_p0_min_, sigma_p0_max_);
        log_sigma_p0_f_.update(s.array().log().matrix());
        refresh_model_params_from_filtered_();
    }

    EIGEN_STRONG_INLINE void param_rw_update_sigma_v0_cmd_(const Vector3& sigma_v0_cmd) {
        if (!param_rw_enabled_) {
            set_Rv0_noise_std(sigma_v0_cmd);
            return;
        }
        Vector3 s = clamp_pos_vec_(sigma_v0_cmd, sigma_v0_min_, sigma_v0_max_);
        log_sigma_v0_f_.update(s.array().log().matrix());
        refresh_model_params_from_filtered_();
    }

    EIGEN_STRONG_INLINE void maybe_run_auto_zero_pseudos_(T Ts) {
        if (!linear_block_enabled_) {
            auto_zero_pseudo_elapsed_sec_ = T(0);
            return;
        }
        if (!(auto_zero_position_pseudo_enabled_ || auto_zero_velocity_pseudo_enabled_)) {
            auto_zero_pseudo_elapsed_sec_ = T(0);
            return;
        }
        if (!(auto_zero_pseudo_period_sec_ > T(0)) || !std::isfinite(auto_zero_pseudo_period_sec_)) {
            return;
        }
        if (!(Ts > T(0)) || !std::isfinite(Ts)) return;

        auto_zero_pseudo_elapsed_sec_ += Ts;
        if (auto_zero_pseudo_elapsed_sec_ + T(1e-12) < auto_zero_pseudo_period_sec_) return;
        auto_zero_pseudo_elapsed_sec_ = T(0);

        if (auto_zero_position_pseudo_enabled_) {
            measurement_update_position_pseudo(Vector3::Zero(), get_Rp0_noise_std());
        }
        if (auto_zero_velocity_pseudo_enabled_) {
            measurement_update_velocity_pseudo(Vector3::Zero(), get_Rv0_noise_std());
        }
    }
};
