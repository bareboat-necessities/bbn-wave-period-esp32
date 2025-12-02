#pragma once
#include <cmath>
#include <algorithm>

template <typename Real = float>
class FirstOrderIIRSmoother {
public:
    /**
     * @param dt                  Sample period in seconds (e.g. 1.0f / 240.0f)
     * @param settle_time_sec90   Time to reach ~90% of a step (seconds)
     */
    FirstOrderIIRSmoother(Real dt,
                          Real settle_time_sec90)
        : y_(Real(0)),
          initialized_(false)
    {
        setGainFromSettleTime(dt, settle_time_sec90);
    }

    void setGainFromSettleTime(Real dt, Real settle_time_sec90) {
        Real N = settle_time_sec90 / dt;
        if (N <= Real(1)) {
            K_ = Real(1); // effectively no smoothing
        } else {
            // K = 1 - 0.1^(1/N)  →  90% of step in settle_time_sec90
            K_ = Real(1) - std::pow(Real(0.1), Real(1) / N);
        }
    }

    void setInitial(Real x0) {
        y_ = x0;
        initialized_ = true;
    }

    Real update(Real x) {
        if (!initialized_) {
            y_ = x;
            initialized_ = true;
            return y_;
        }
        y_ += K_ * (x - y_);
        return y_;
    }

    Real value() const { return y_; }
    Real gain()  const { return K_; }

private:
    Real y_;
    Real K_;
    bool initialized_;
};
