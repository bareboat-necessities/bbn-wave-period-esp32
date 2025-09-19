#pragma once
#include <cmath>
#include <Eigen/Dense>

// WGS84 constants
constexpr double WGS84_A = 6378137.0;       // semi-major axis [m]
constexpr double WGS84_F = 1.0/298.257223563;
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double KNOTS2MPS = 0.514444;

class GPS_to_INS {
public:
    using Vector3 = Eigen::Matrix<double,3,1>;
    using Matrix3 = Eigen::Matrix<double,3,3>;

    struct Result {
        Vector3 measurement = Vector3::Zero();   // NED measurement (pos or vel)
        Matrix3 covariance = Matrix3::Identity();
        bool valid = false;                      // accepted by quality checks
    };

    // Constructor: set reference origin (deg, deg, m)
    GPS_to_INS(double lat0_deg, double lon0_deg, double alt0_m)
    {
        lat0_rad_ = lat0_deg * DEG2RAD;
        lon0_rad_ = lon0_deg * DEG2RAD;
        alt0_m_   = alt0_m;

        // Rotation ECEF->NED
        double sLat = std::sin(lat0_rad_), cLat = std::cos(lat0_rad_);
        double sLon = std::sin(lon0_rad_), cLon = std::cos(lon0_rad_);
        R_ecef2ned_ <<
            -sLat*cLon, -sLat*sLon,  cLat,
             -sLon,      cLon,       0.0,
            -cLat*cLon, -cLat*sLon, -sLat;

        ref_ecef_ = lla_to_ecef(lat0_rad_, lon0_rad_, alt0_m_);
    }

    // === Build position measurement (lat/lon/alt → NED + covariance) ===
    Result build_position_measurement(double lat_deg, double lon_deg, double alt_m,
                                      int fixType, double hdop, double vdop,
                                      double max_hdop=5.0, double max_vdop=10.0) const
    {
        Result out;
        if (fixType <= 0) return out;               // no fix
        if (hdop > max_hdop || vdop > max_vdop) return out;

        out.measurement = lla_to_ned(lat_deg, lon_deg, alt_m);

        // Covariance from DOPs
        out.covariance = Matrix3::Zero();
        double sigma_h = hdop * 1.5;   // base 1.5 m
        double sigma_v = vdop * 3.0;   // base 3.0 m
        out.covariance(0,0) = sigma_h*sigma_h;
        out.covariance(1,1) = sigma_h*sigma_h;
        out.covariance(2,2) = sigma_v*sigma_v;

        out.valid = true;
        return out;
    }

    // === Build velocity measurement (COG/SOG → NED + covariance) ===
    Result build_velocity_measurement(double cog_deg, double sog, bool sog_in_knots,
                                      double climb_rate_mps,
                                      int fixType, double sAcc=0.2,
                                      double max_sAcc=2.0) const
    {
        Result out;
        if (fixType <= 0) return out;
        if (sAcc > max_sAcc) return out;

        double sog_mps = sog_in_knots ? sog*KNOTS2MPS : sog;
        double cog_rad = cog_deg * DEG2RAD;
        double vN = sog_mps * std::cos(cog_rad);
        double vE = sog_mps * std::sin(cog_rad);
        double vD = -climb_rate_mps; // Down positive
        out.measurement = Vector3(vN, vE, vD);

        // Covariance from reported sAcc
        out.covariance = Matrix3::Zero();
        out.covariance(0,0) = sAcc*sAcc;
        out.covariance(1,1) = sAcc*sAcc;
        out.covariance(2,2) = std::max(0.5, sAcc*2.0) * std::max(0.5, sAcc*2.0);

        out.valid = true;
        return out;
    }

private:
    double lat0_rad_, lon0_rad_, alt0_m_;
    Vector3 ref_ecef_;
    Eigen::Matrix<double,3,3> R_ecef2ned_;

    Vector3 lla_to_ned(double lat_deg, double lon_deg, double alt_m) const {
        Vector3 ecef = lla_to_ecef(lat_deg*DEG2RAD, lon_deg*DEG2RAD, alt_m);
        Vector3 d_ecef = ecef - ref_ecef_;
        return R_ecef2ned_ * d_ecef;
    }

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
