#pragma once
#include <cmath>
#include <Eigen/Dense>

// Simple WGS84 constants
constexpr double WGS84_A = 6378137.0;       // semi-major axis [m]
constexpr double WGS84_F = 1.0/298.257223563;
constexpr double WGS84_B = WGS84_A * (1.0 - WGS84_F);
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double KNOTS2MPS = 0.514444;

class GPS_to_INS {
public:
    using Vector3 = Eigen::Matrix<double,3,1>;
    using Matrix3 = Eigen::Matrix<double,3,3>;

    // Reference origin (lat0, lon0 in deg, alt0 in m)
    GPS_to_INS(double lat0_deg, double lon0_deg, double alt0_m)
    {
        lat0_rad_ = lat0_deg * DEG2RAD;
        lon0_rad_ = lon0_deg * DEG2RAD;
        alt0_m_   = alt0_m;

        // Precompute rotation ECEF->NED
        double sLat = std::sin(lat0_rad_), cLat = std::cos(lat0_rad_);
        double sLon = std::sin(lon0_rad_), cLon = std::cos(lon0_rad_);
        R_ecef2ned_ <<
            -sLat*cLon, -sLat*sLon,  cLat,
             -sLon,      cLon,       0.0,
            -cLat*cLon, -cLat*sLon, -sLat;

        // Store reference ECEF
        ref_ecef_ = lla_to_ecef(lat0_rad_, lon0_rad_, alt0_m_);
    }

    // Convert GPS fix → NED
    Vector3 lla_to_ned(double lat_deg, double lon_deg, double alt_m) const {
        Vector3 ecef = lla_to_ecef(lat_deg * DEG2RAD, lon_deg * DEG2RAD, alt_m);
        Vector3 d_ecef = ecef - ref_ecef_;
        return R_ecef2ned_ * d_ecef;
    }

    // Convert COG/SOG → NED velocity
    Vector3 cog_sog_to_ned(double cog_deg, double sog_value, bool sog_in_knots=true,
                           double climb_rate_mps=0.0) const {
        double sog_mps = sog_in_knots ? sog_value*KNOTS2MPS : sog_value;
        double cog_rad = cog_deg * DEG2RAD;
        double vN = sog_mps * std::cos(cog_rad);
        double vE = sog_mps * std::sin(cog_rad);
        double vD = -climb_rate_mps; // Down positive
        return Vector3(vN, vE, vD);
    }

    // Position covariance from HDOP/VDOP
    Matrix3 pos_cov_from_dops(double hdop, double vdop,
                              double sigma_horiz_base=1.5,
                              double sigma_vert_base=3.0) const {
        Matrix3 R = Matrix3::Zero();
        R(0,0) = std::pow(hdop * sigma_horiz_base,2);
        R(1,1) = std::pow(hdop * sigma_horiz_base,2);
        R(2,2) = std::pow(vdop * sigma_vert_base,2);
        return R;
    }

    // Velocity covariance
    Matrix3 vel_cov(double sigma_vn=0.1, double sigma_ve=0.1, double sigma_vd=0.3) const {
        Matrix3 R = Matrix3::Zero();
        R(0,0) = sigma_vn*sigma_vn;
        R(1,1) = sigma_ve*sigma_ve;
        R(2,2) = sigma_vd*sigma_vd;
        return R;
    }

    // χ² gating test (returns true if measurement is accepted)
    bool chi2_gate(const Vector3& z, const Matrix3& R,
                   double chi2_thresh=16.27) const {
        // 16.27 ≈ chi2inv(0.99,3) → 99% confidence gate for 3 DOF
        Eigen::LLT<Matrix3> llt(R);
        if(llt.info()!=Eigen::Success) return false; // bad covariance
        Vector3 y = llt.matrixL().solve(z); // R^{-1/2} z
        double d2 = y.squaredNorm();
        return d2 <= chi2_thresh;
    }

private:
    double lat0_rad_, lon0_rad_, alt0_m_;
    Vector3 ref_ecef_;
    Eigen::Matrix<double,3,3> R_ecef2ned_;

    static Vector3 lla_to_ecef(double lat_rad, double lon_rad, double alt_m) {
        double a = WGS84_A;
        double f = WGS84_F;
        double e2 = f*(2-f);
        double sLat = std::sin(lat_rad), cLat = std::cos(lat_rad);
        double sLon = std::sin(lon_rad), cLon = std::cos(lon_rad);

        double N = a / std::sqrt(1 - e2*sLat*sLat);
        double x = (N+alt_m) * cLat * cLon;
        double y = (N+alt_m) * cLat * sLon;
        double z = (N*(1-e2)+alt_m) * sLat;
        return Vector3(x,y,z);
    }
};
